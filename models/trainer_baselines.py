import torch
import torch.nn.functional as F
from collections import OrderedDict

import utils.util as util
from models.models import BaseRunnerModel, BaseModel, BaseTrainerModel
import networks
import networks.losses as losses


class ConcatTrainer(BaseTrainerModel):
    def __init__(self, opt):
        super(ConcatTrainer, self).__init__(opt)
        self._name = 'ConcatTrainer'

        # create networks
        self._init_create_networks()

        # init train variables and losses
        if self._is_train:
            self._init_train_vars()
            self._init_losses()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # define inputs tensors
        self.head_bbox = None       # (N, 4)
        self.valid_bbox = None      # (N,)
        self.images = None          # (N, 2, 3, h, w)
        self.pseudo_masks = None    # (2N, 1, h, w)
        self.T = None               # (N, h, w, 2)
        self.bg_inputs = None       # (N, 4, h, w)
        self.src_inputs = None
        self.tsf_inputs = None
        self.inputs = None          # (N, 9, h, w)

        # define output/visualization tensors
        self._vis_src_img = None       # source image
        self._vis_gen_bg = None        # generated bg
        self._vis_gen_color = None     # generated color image
        self._vis_gen_mask = None      # generated mask image
        self._vis_gen_img = None       # generated target image
        self._vis_batch_src_img = None  # batch source images
        self._vis_batch_gen_img = None  # batch generated images

    def _init_create_networks(self):
        multi_gpus = len(self._gpu_ids) > 1

        # generator network
        self._G = self._create_generator()
        self._G.init_weights()
        if multi_gpus:
            self._G = torch.nn.DataParallel(self._G)
        self._G.cuda()

        # discriminator network
        self._D = self._create_discriminator()
        self._D.init_weights()
        if multi_gpus:
            self._D = torch.nn.DataParallel(self._D)
        self._D.cuda()

    def _create_generator(self):
        return networks.create_by_name('concat', bg_dim=4, src_dim=self._G_cond_nc,
                                       tsf_dim=self._G_cond_nc, norm_type=self._opt.norm_type)

    def _create_discriminator(self):
        return networks.create_by_name('patch', input_nc=3+2*self._G_cond_nc, ndf=64,
                                       n_layers=self._opt.D_layers, use_sigmoid=False)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=(self._opt.G_adam_b1, self._opt.G_adam_b2))
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=(self._opt.D_adam_b1, self._opt.D_adam_b2))

    def _init_losses(self):
        # define loss functions
        multi_gpus = len(self._gpu_ids) > 1

        if self._opt.use_vgg:

            self._criterion_vgg = losses.VGGLoss(before_relu=True)
            if multi_gpus:
                self._criterion_vgg = torch.nn.DataParallel(self._criterion_vgg)
            self._criterion_vgg.cuda()

        if self._opt.use_face:
            self._criterion_face = losses.SphereFaceLoss()
            if multi_gpus:
                self._criterion_face = torch.nn.DataParallel(self._criterion_face)
            self._criterion_face.cuda()

        # init losses G
        self._loss_g_vgg = self._Tensor([0])
        self._loss_g_face = self._Tensor([0])
        self._loss_g_adv = self._Tensor([0])
        self._loss_g_smooth = self._Tensor([0])
        self._loss_g_mask = self._Tensor([0])
        self._loss_g_mask_smooth = self._Tensor([0])

        # init losses D
        self._d_real = self._Tensor([0])
        self._d_fake = self._Tensor([0])

    def set_input(self, sample):
        with torch.no_grad():
            self.head_bbox = sample['head_bbox']            # (N, 4)
            self.valid_bbox = sample['valid_bbox'].cuda()   # (N, 4)
            self.T = sample['T'].cuda()                     # (N, h, w, 2)
            self.bg_inputs = sample['bg_inputs'].cuda()     # (N, 4, h, w)

            inputs = torch.cat([sample['src_inputs'], sample['tsf_inputs'][:, 3:]], dim=1)
            self.inputs = inputs.cuda()
            self.images = sample['images'][:, 1, ...].cuda()
            self.pseudo_masks = sample['pseudo_masks'][:, 1, ...].cuda()

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        # generate fake images
        fake_bg, fake_tsf_color, fake_tsf_mask = \
            self._G.forward(self.bg_inputs, self.inputs)

        fake_tsf_imgs = fake_tsf_mask * fake_bg + (1 - fake_tsf_mask) * fake_tsf_color

        # keep data for visualization
        if keep_data_for_visuals:
            self.put_to_visuals(fake_tsf_color, fake_bg, fake_tsf_imgs, fake_tsf_mask)

        return fake_tsf_imgs, fake_tsf_mask, None

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
        if self._is_train:
            # run
            fake_tsf_imgs, fake_tsf_mask, _ = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            loss_G = self._optimize_G(fake_tsf_imgs, fake_tsf_mask)

            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

            # train D
            if trainable:
                loss_D = self._optimize_D(fake_tsf_imgs)
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

    def _optimize_G(self, fake_tsf_imgs, fake_tsf_mask):
        fake_input_D = torch.cat([fake_tsf_imgs, self.inputs[:, 3:, ...]], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        if self._opt.use_vgg:
            self._loss_g_vgg = torch.mean(self._criterion_vgg(fake_tsf_imgs, self.images)) * self._opt.lambda_vgg

        if self._opt.use_face:
            self._loss_g_face = torch.mean(
                self._criterion_face(fake_tsf_imgs, self.images, weights=self.valid_bbox,
                                     bbox1=self.head_bbox, bbox2=self.head_bbox)) * self._opt.lambda_face
        # loss mask
        self._loss_g_mask = torch.mean((fake_tsf_mask - self.pseudo_masks) ** 2) * self._opt.lambda_mask
        self._loss_g_mask_smooth = self._compute_loss_smooth(fake_tsf_mask) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_vgg + self._loss_g_face + \
               self._loss_g_mask + self._loss_g_mask_smooth

    def _optimize_D(self, fake_tsf_imgs):
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), self.inputs[:, 3:, ...]], dim=1)
        real_input_D = self.inputs

        d_real_outs = self._D.forward(real_input_D)
        d_fake_outs = self._D.forward(fake_input_D)

        _loss_d_real = self._compute_loss_D(d_real_outs, 1) * self._opt.lambda_D_prob
        _loss_d_fake = self._compute_loss_D(d_fake_outs, -1) * self._opt.lambda_D_prob

        self._d_real = torch.mean(d_real_outs)
        self._d_fake = torch.mean(d_fake_outs)

        # combine losses
        return _loss_d_real + _loss_d_fake

    def _compute_loss_D(self, x, y):
        return torch.mean((x - y) ** 2)

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_vgg', self._loss_g_vgg.item()),
                                 ('g_face', self._loss_g_face.item()),
                                 ('g_adv', self._loss_g_adv.item()),
                                 ('g_mask', self._loss_g_mask.item()),
                                 ('g_mask_smooth', self._loss_g_mask_smooth.item()),
                                 ('d_real', self._d_real.item()),
                                 ('d_fake', self._d_fake.item())])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # inputs
        visuals['1_src_img'] = self._vis_src_img

        # outputs
        visuals['2_gen_bg'] = self._vis_gen_bg
        visuals['3_gen_color'] = self._vis_gen_color
        visuals['4_gen_mask'] = self._vis_gen_mask
        visuals['5_gen_img'] = self._vis_gen_img

        # batch outputs
        visuals['6_batch_src_img'] = self._vis_batch_src_img
        visuals['7_batch_gen_img'] = self._vis_batch_gen_img

        return visuals

    def put_to_visuals(self, fake_color, fake_bg, fake_imgs, fake_masks):
        with torch.no_grad():
            self._vis_src_img = util.tensor2im(self.images[0])
            self._vis_gen_bg = util.tensor2im(fake_bg[0].detach())
            self._vis_gen_color = util.tensor2im(fake_color[0].detach())
            self._vis_gen_mask = util.tensor2maskim(fake_masks[0].detach())
            self._vis_gen_img = util.tensor2im(fake_imgs[0].detach())

            self._vis_batch_src_img = util.tensor2im(self.images, idx=-1)
            self._vis_batch_gen_img = util.tensor2im(fake_imgs.detach(), idx=-1)

    def save(self, label):
        # save networks
        self._save_network(self._G, 'G', label)
        self._save_network(self._D, 'D', label)

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_D, 'D', label)

    def load(self):
        multi_gpus = len(self._gpu_ids) > 1
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch, need_module=multi_gpus)

        if self._is_train:
            # load D
            self._load_network(self._D, 'D', load_epoch, need_module=multi_gpus)

            # load optimizers
            self._load_optimizer(self._optimizer_G, 'G', load_epoch)
            self._load_optimizer(self._optimizer_D, 'D', load_epoch)

    def update_learning_rate(self):
        # updated learning rate G
        final_lr = self._opt.final_lr

        lr_decay_G = (self._opt.lr_G - final_lr) / self._opt.nepochs_decay
        self._current_lr_G -= lr_decay_G
        for param_group in self._optimizer_G.param_groups:
            param_group['lr'] = self._current_lr_G
        print('update G learning rate: %f -> %f' % (self._current_lr_G + lr_decay_G, self._current_lr_G))

        # update learning rate D
        lr_decay_D = (self._opt.lr_D - final_lr) / self._opt.nepochs_decay
        self._current_lr_D -= lr_decay_D
        for param_group in self._optimizer_D.param_groups:
            param_group['lr'] = self._current_lr_D
        print('update D learning rate: %f -> %f' % (self._current_lr_D + lr_decay_D, self._current_lr_D))


class TextureWarpingTrainer(ConcatTrainer):
    def __init__(self, opt):
        super(TextureWarpingTrainer, self).__init__(opt)
        self._name = 'TextureWarpingTrainer'

    def _create_generator(self):
        return networks.create_by_name('texture', bg_dim=4, src_dim=0,
                                       tsf_dim=self._G_cond_nc, norm_type=self._opt.norm_type)

    def _create_discriminator(self):
        return networks.create_by_name('patch', input_nc=3+self._G_cond_nc, ndf=64,
                                       n_layers=self._opt.D_layers, use_sigmoid=False)

    def set_input(self, sample):
        with torch.no_grad():
            self.head_bbox = sample['head_bbox']            # (N, 4)
            self.valid_bbox = sample['valid_bbox'].cuda()   # (N, 4)
            self.T = sample['T'].cuda()                     # (N, h, w, 2)
            self.bg_inputs = sample['bg_inputs'].cuda()     # (N, 4, h, w)
            self.inputs = sample['tsf_inputs'].cuda()
            self.images = sample['images'][:, 1, ...].cuda()
            self.pseudo_masks = sample['pseudo_masks'][:, 1, ...].cuda()

    def _optimize_G(self, fake_tsf_imgs, fake_tsf_mask):
        fake_input_D = torch.cat([fake_tsf_imgs, self.inputs[:, 3:, ...]], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        if self._opt.use_vgg:
            self._loss_g_vgg = torch.mean(self._criterion_vgg(fake_tsf_imgs, self.images)) * self._opt.lambda_vgg

        if self._opt.use_face:
            self._loss_g_face = torch.mean(
                self._criterion_face(fake_tsf_imgs, self.images, weights=self.valid_bbox,
                                     bbox1=self.head_bbox, bbox2=self.head_bbox)) * self._opt.lambda_face
        # loss mask
        self._loss_g_mask = torch.mean((fake_tsf_mask - self.pseudo_masks) ** 2) * self._opt.lambda_mask
        self._loss_g_mask_smooth = self._compute_loss_smooth(fake_tsf_mask) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_vgg + self._loss_g_face + \
               self._loss_g_mask + self._loss_g_mask_smooth

    def _optimize_D(self, fake_tsf_imgs):
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), self.inputs[:, 3:, ...]], dim=1)
        real_input_D = torch.cat([self.images, self.inputs[:, 3:, ...]], dim=1)

        d_real_outs = self._D.forward(real_input_D)
        d_fake_outs = self._D.forward(fake_input_D)

        _loss_d_real = self._compute_loss_D(d_real_outs, 1) * self._opt.lambda_D_prob
        _loss_d_fake = self._compute_loss_D(d_fake_outs, -1) * self._opt.lambda_D_prob

        self._d_real = torch.mean(d_real_outs)
        self._d_fake = torch.mean(d_fake_outs)

        # combine losses
        return _loss_d_real + _loss_d_fake


class FeatureWarpingTrainer(ConcatTrainer):
    def __init__(self, opt):
        super(FeatureWarpingTrainer, self).__init__(opt)
        self._name = 'FeatureWarpingTrainer'

    def _create_generator(self):
        return networks.create_by_name('feature', bg_dim=4, src_dim=self._G_cond_nc,
                                       tsf_dim=self._G_cond_nc, norm_type=self._opt.norm_type)

    def _create_discriminator(self):
        return networks.create_by_name('patch', input_nc=3+self._D_cond_nc, ndf=64,
                                       n_layers=self._opt.D_layers, use_sigmoid=False)

    def set_input(self, sample):
        with torch.no_grad():
            self.head_bbox = sample['head_bbox']            # (N, 4)
            self.valid_bbox = sample['valid_bbox'].cuda()   # (N, 4)
            self.T = sample['T'].cuda()                     # (N, h, w, 2)
            self.bg_inputs = sample['bg_inputs'].cuda()     # (N, 4, h, w)
            self.src_inputs = sample['src_inputs'][:, 0:3, ...].cuda()
            self.tsf_inputs = sample['tsf_inputs'].cuda()
            self.images = sample['images'][:, 1, ...].cuda()
            self.pseudo_masks = sample['pseudo_masks'][:, 1, ...].cuda()

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        # generate fake images
        fake_bg, fake_tsf_color, fake_tsf_mask = \
            self._G.forward(self.bg_inputs, self.src_inputs, self.tsf_inputs)

        fake_tsf_imgs = fake_tsf_mask * fake_bg + (1 - fake_tsf_mask) * fake_tsf_color

        # keep data for visualization
        if keep_data_for_visuals:
            self.put_to_visuals(fake_tsf_color, fake_bg, fake_tsf_imgs, fake_tsf_mask)

        return fake_tsf_imgs, fake_tsf_mask, None

    def _optimize_G(self, fake_tsf_imgs, fake_tsf_mask):
        fake_input_D = torch.cat([fake_tsf_imgs, self.tsf_inputs[:, 3:, ...]], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        if self._opt.use_vgg:
            self._loss_g_vgg = torch.mean(self._criterion_vgg(fake_tsf_imgs, self.images)) * self._opt.lambda_vgg

        if self._opt.use_face:
            self._loss_g_face = torch.mean(
                self._criterion_face(fake_tsf_imgs, self.images, weights=self.valid_bbox,
                                     bbox1=self.head_bbox, bbox2=self.head_bbox)) * self._opt.lambda_face
        # loss mask
        self._loss_g_mask = torch.mean((fake_tsf_mask - self.pseudo_masks) ** 2) * self._opt.lambda_mask
        self._loss_g_mask_smooth = self._compute_loss_smooth(fake_tsf_mask) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_vgg + self._loss_g_face + \
               self._loss_g_mask + self._loss_g_mask_smooth

    def _optimize_D(self, fake_tsf_imgs):
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), self.tsf_inputs[:, 3:, ...]], dim=1)
        real_input_D = torch.cat([self.images, self.tsf_inputs[:, 3:, ...]], dim=1)

        d_real_outs = self._D.forward(real_input_D)
        d_fake_outs = self._D.forward(fake_input_D)

        _loss_d_real = self._compute_loss_D(d_real_outs, 1) * self._opt.lambda_D_prob
        _loss_d_fake = self._compute_loss_D(d_fake_outs, -1) * self._opt.lambda_D_prob

        self._d_real = torch.mean(d_real_outs)
        self._d_fake = torch.mean(d_fake_outs)

        # combine losses
        return _loss_d_real + _loss_d_fake