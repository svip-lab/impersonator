import torch
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import os

import utils.util as util
import utils.cv_utils as cv_utils
from models.models import BaseRunnerModel, BaseTrainerModel
import networks

import ipdb


class ImperTrainer(BaseTrainerModel):
    def __init__(self, opt):
        super(ImperTrainer, self).__init__(opt)
        self._name = 'ImperTrainer'

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
        self.src_inputs = None      # (N, 6, h, w)
        self.tsf_inputs = None      # (N, 6, h, w)

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
        return networks.create_by_name(self._opt.gen_name, bg_dim=4, src_dim=3+self._G_cond_nc,
                                       tsf_dim=3+self._G_cond_nc, norm_type=self._opt.norm_type, replace=True)

    def _create_discriminator(self):
        return networks.create_by_name('patch', input_nc=3+self._D_cond_nc, ndf=64,
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
        import networks.losses as losses
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
            self.src_inputs = sample['src_inputs'].cuda()   # (N, 6, h, w)
            self.tsf_inputs = sample['tsf_inputs'].cuda()   # (N, 6, h, w)
            # self.images = sample['images'].cuda()         # (N, 2, 3, h, w)
            images = sample['images']
            # ipdb.set_trace()
            self.images = torch.cat([images[:, 0, ...], images[:, 1, ...]], dim=0).cuda()   # (2N, 3, h, w)
            pseudo_masks = sample['pseudo_masks']
            self.pseudo_masks = torch.cat([pseudo_masks[:, 0, ...], pseudo_masks[:, 1, ...]], dim=0).cuda()  # (2N, 1, h, w)

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        # generate fake images
        fake_bg, fake_src_color, fake_src_mask, fake_tsf_color, fake_tsf_mask = \
            self._G.forward(self.bg_inputs, self.src_inputs, self.tsf_inputs, T=self.T)

        fake_src_imgs = fake_src_mask * fake_bg + (1 - fake_src_mask) * fake_src_color
        fake_tsf_imgs = fake_tsf_mask * fake_bg + (1 - fake_tsf_mask) * fake_tsf_color

        fake_masks = torch.cat([fake_src_mask, fake_tsf_mask], dim=0)
        fake_imgs = torch.cat([fake_src_imgs, fake_tsf_imgs], dim=0)

        # keep data for visualization
        if keep_data_for_visuals:
            self.put_to_visuals(fake_tsf_color, fake_bg, fake_imgs, fake_masks)

        return fake_tsf_imgs, fake_imgs, fake_masks

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
        if self._is_train:
            # run
            fake_tsf_imgs, fake_imgs, fake_masks = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            loss_G = self._optimize_G(fake_tsf_imgs, fake_imgs, fake_masks)

            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

            # train D
            if trainable:
                loss_D = self._optimize_D(fake_tsf_imgs)
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

    def _optimize_G(self, fake_tsf_imgs, fake_imgs, fake_masks):
        bs = fake_tsf_imgs.shape[0]
        fake_input_D = torch.cat([fake_tsf_imgs, self.tsf_inputs[:, 3:, ...]], dim=1)

        front_mask = 1 - self.pseudo_masks[bs:]
        fake_input_D = torch.cat([fake_tsf_imgs * front_mask,
                                  self.tsf_inputs[:, 3:, ...]], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        if self._opt.use_vgg:
            self._loss_g_vgg = torch.mean(self._criterion_vgg(fake_imgs, self.images)) * self._opt.lambda_vgg

        if self._opt.use_face:
            self._loss_g_face = torch.mean(
                self._criterion_face(fake_tsf_imgs, self.images[bs:, ...], weights=self.valid_bbox,
                                     bbox1=self.head_bbox, bbox2=self.head_bbox)) * self._opt.lambda_face
        # # loss mask
        # self._loss_g_mask = torch.mean((fake_masks - self.pseudo_masks) ** 2) * self._opt.lambda_mask
        self._loss_g_mask = torch.mean(torch.abs(fake_masks - self.pseudo_masks)) * self._opt.lambda_mask
        self._loss_g_mask_smooth = self._compute_loss_smooth(fake_masks) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_vgg + self._loss_g_face + \
               self._loss_g_mask + self._loss_g_mask_smooth

    def _optimize_D(self, fake_tsf_imgs):
        # fake_input_D = torch.cat([fake_tsf_imgs.detach(), self.tsf_inputs[:, 3:, ...]], dim=1)
        # real_input_D = self.tsf_inputs

        bs = fake_tsf_imgs.shape[0]
        front_mask = 1 - self.pseudo_masks[bs:]
        fake_input_D = torch.cat([fake_tsf_imgs.detach() * front_mask,
                                  self.tsf_inputs[:, 3:, ...]], dim=1)

        real_input_D = torch.cat([self.images[bs:] * front_mask,
                                  self.tsf_inputs[:, 3:, ...]], dim=1)

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
            ids = fake_imgs.shape[0] // 2
            self._vis_src_img = util.tensor2im(self.images[0])
            self._vis_gen_bg = util.tensor2im(fake_bg[0].detach())
            self._vis_gen_color = util.tensor2im(fake_color[0].detach())
            self._vis_gen_mask = util.tensor2maskim(fake_masks[ids].detach())
            self._vis_gen_img = util.tensor2im(fake_imgs[ids].detach())

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

    def debug(self, visualizer):
        import torch.nn.functional as F
        import utils.util as util
        pass


class ImperTrainerFixBG(BaseTrainerModel):
    def __init__(self, opt):
        super(ImperTrainerFixBG, self).__init__(opt)
        self._name = 'ImperTrainerFixBG'

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
        self.src_inputs = None      # (N, 6, h, w)
        self.tsf_inputs = None      # (N, 6, h, w)
        self.bg_imgs = None         # (2N, 3, h, w)

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

        # bgnet network
        self._bg = self._create_bgnet()
        if multi_gpus:
            self._bg = torch.nn.DataParallel(self._bg)
        self._bg.cuda()

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

    def _create_bgnet(self):
        net = networks.create_by_name('isag', c_dim=4)
        self._load_params(net, self._opt.bg_model, need_module=False)
        net.eval()
        return net

    def _create_generator(self):
        return networks.create_by_name(self._opt.gen_name, bg_dim=4, src_dim=3+self._G_cond_nc,
                                       tsf_dim=3+self._G_cond_nc, norm_type=self._opt.norm_type)

    def _create_discriminator(self):
        return networks.create_by_name('patch', input_nc=3+self._D_cond_nc, ndf=64, n_layers=4, use_sigmoid=False)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=(self._opt.G_adam_b1, self._opt.G_adam_b2))
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=(self._opt.D_adam_b1, self._opt.D_adam_b2))

    def _init_losses(self):
        import networks.losses as losses
        # define loss functions
        multi_gpus = len(self._gpu_ids) > 1

        if self._opt.use_vgg:

            self._criterion_vgg = losses.VGGLoss(before_relu=False)
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
            self.valid_bbox = sample['valid_bbox'].cuda()   # (N, 1)
            self.T = sample['T'].cuda()                     # (N, h, w, 2)
            self.src_inputs = sample['src_inputs'].cuda()   # (N, 6, h, w)
            self.tsf_inputs = sample['tsf_inputs'].cuda()   # (N, 6, h, w)

            images = sample['images']
            self.images = torch.cat([images[:, 0, ...], images[:, 1, ...]], dim=0).cuda()   # (2N, 3, h, w)

            pseudo_masks = sample['pseudo_masks']
            self.pseudo_masks = torch.cat([pseudo_masks[:, 0, ...], pseudo_masks[:, 1, ...]], dim=0).cuda()  # (2N, 1, h, w)

            bg_inputs = sample['bg_inputs']
            bg_inputs = torch.cat([bg_inputs[:, 0, ...], bg_inputs[:, 1, ...]], dim=0)  # (2N, 3+1, h, w)
            bg_imgs = bg_inputs[:, 0:3, ...].cuda()
            bg_masks = bg_inputs[:, 3:, ...].cuda()
            # ipdb.set_trace()

            if self._is_train:
                self.bg_imgs = self._bg(bg_imgs, masks=bg_masks, only_out=True)
            else:
                self.bg_imgs = self._bg(bg_imgs, masks=bg_masks, only_x=True)
            # print(self.bg_imgs.shape)

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_G_train(self):
        self._G.train()
        self._is_train = False

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        # generate fake images
        fake_src_color, fake_src_mask, fake_tsf_color, fake_tsf_mask = \
            self._G.forward(self.src_inputs, self.tsf_inputs, T=self.T)

        bs = fake_src_color.shape[0]
        if self._is_train:
            fake_src_imgs = fake_src_mask * self.bg_imgs[0:bs] + (1 - fake_src_mask) * fake_src_color
            fake_tsf_imgs = fake_tsf_mask * self.bg_imgs[bs:] + (1 - fake_tsf_mask) * fake_tsf_color
        else:
            fake_src_imgs = fake_src_mask * self.bg_imgs[0:bs] + (1 - fake_src_mask) * fake_src_color
            fake_tsf_imgs = fake_tsf_mask * self.bg_imgs[0:bs] + (1 - fake_tsf_mask) * fake_tsf_color

        fake_masks = torch.cat([fake_src_mask, fake_tsf_mask], dim=0)
        fake_imgs = torch.cat([fake_src_imgs, fake_tsf_imgs], dim=0)

        # keep data for visualization
        if keep_data_for_visuals:
            self.put_to_visuals(fake_tsf_color, self.bg_imgs, fake_imgs, fake_masks)

        # return fake_tsf_imgs, fake_imgs, fake_masks
        return fake_tsf_imgs, fake_imgs, fake_tsf_color

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
        if self._is_train:
            # run
            fake_tsf_imgs, fake_imgs, fake_masks = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            loss_G = self._optimize_G(fake_tsf_imgs, fake_imgs, fake_masks)

            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

            # train D
            if trainable:
                loss_D = self._optimize_D(fake_tsf_imgs)
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

    def _optimize_G(self, fake_tsf_imgs, fake_imgs, fake_masks):
        bs = fake_tsf_imgs.shape[0]

        # fake_input_D = torch.cat([fake_tsf_imgs, self.tsf_inputs[:, 3:, ...]], dim=1)
        front_mask = 1 - self.pseudo_masks[bs:]
        # front_mask = 1 - self.tsf_inputs[:, -1:, ...]
        fake_input_D = torch.cat([fake_tsf_imgs * front_mask,
                                  self.tsf_inputs[:, 3:, ...]], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        if self._opt.use_vgg:
            self._loss_g_vgg = torch.mean(self._criterion_vgg(fake_imgs, self.images)) * self._opt.lambda_vgg

        if self._opt.use_face:
            self._loss_g_face = torch.mean(self._criterion_face(
                fake_tsf_imgs, self.images[bs:, ...], weights=self.valid_bbox,
                bbox1=self.head_bbox, bbox2=self.head_bbox)) * self._opt.lambda_face
        # loss mask
        # self._loss_g_mask = torch.mean((fake_masks - self.pseudo_masks) ** 2) * self._opt.lambda_mask
        self._loss_g_mask = torch.mean(torch.abs(fake_masks - self.pseudo_masks)) * self._opt.lambda_mask

        if self._opt.lambda_mask_smooth != 0:
            self._loss_g_mask_smooth = self._compute_loss_smooth(fake_masks) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_vgg + self._loss_g_face + \
               self._loss_g_mask + self._loss_g_mask_smooth

    def _optimize_D(self, fake_tsf_imgs):
        # fake_input_D = torch.cat([fake_tsf_imgs.detach(), self.tsf_inputs[:, 3:, ...]], dim=1)
        # real_input_D = self.tsf_inputs

        bs = fake_tsf_imgs.shape[0]
        front_mask = 1 - self.pseudo_masks[bs:]
        # front_mask = 1 - self.tsf_inputs[:, -1:, ...]
        fake_input_D = torch.cat([fake_tsf_imgs.detach() * front_mask,
                                  self.tsf_inputs[:, 3:, ...]], dim=1)

        real_input_D = torch.cat([self.images[bs:] * front_mask,
                                  self.tsf_inputs[:, 3:, ...]], dim=1)

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
            ids = fake_imgs.shape[0] // 2
            self._vis_src_img = util.tensor2im(self.images[0])
            self._vis_gen_bg = util.tensor2im(fake_bg[0].detach())
            self._vis_gen_color = util.tensor2im(fake_color[0].detach())
            self._vis_gen_mask = util.tensor2maskim(fake_masks[ids].detach())
            self._vis_gen_img = util.tensor2im(fake_imgs[ids].detach())

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

    def debug(self, visualizer):
        import torch.nn.functional as F
        import utils.util as util
        pass


class Imitator(BaseRunnerModel):
    def __init__(self, opt):
        super(Imitator, self).__init__(opt)

        self._create_networks()

        # prefetch variables
        self.src_info = None
        self.tsf_info = None
        self.T = None
        self.first_cam = None

    def _create_networks(self):
        # 0. create bgnet
        self.bgnet = self._create_bgnet().cuda()

        # 1. create generator
        self.generator = self._create_generator().cuda()

        # 2. create hmr
        self.hmr = self._create_hmr().cuda()

        # 3. create render
        self.render = networks.SMPLRendererV2(image_size=self._opt.image_size, tex_size=self._opt.tex_size,
                                              has_front=False, fill_back=False).cuda()
        # 4. pre-processor
        if self._opt.has_detector:
            self.detector = networks.PersonMaskRCNNDetector(ks=self._opt.bg_ks, threshold=0.5, to_gpu=True)
        else:
            self.detector = None
        # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        # model.eval()
        # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        # predictions = model(x)

    def _create_bgnet(self):
        net = networks.create_by_name('isag', c_dim=4)
        self._load_params(net, self._opt.bg_model, need_module=False)
        net.eval()
        return net

    def _create_generator(self):
        net = networks.create_by_name(self._opt.gen_name, bg_dim=4, src_dim=3+self._G_cond_nc,
                                      tsf_dim=3+self._G_cond_nc, norm_type=self._opt.norm_type)

        if self._opt.load_path:
            self._load_params(net, self._opt.load_path)
        elif self._opt.load_epoch > 0:
            self._load_network(net, 'G', self._opt.load_epoch)
        else:
            raise ValueError('load_path {} is empty and load_epoch {} is 0'.format(
                self._opt.load_path, self._opt.load_epoch))

        net.eval()
        return net

    def _create_hmr(self):
        hmr = networks.bodymesh.HumanModelRecovery(smpl_data=util.load_pickle_file(self._opt.smpl_model))
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def visualize(self, *args, **kwargs):
        pass

    def personalize(self, src_path, src_smpl=None, output_path='', visualizer=None):

        with torch.no_grad():
            ori_img = cv_utils.read_cv2_img(src_path)

            # resize image and convert the color space from [0, 255] to [-1, 1]
            img = cv_utils.transform_img(ori_img, self._opt.image_size, transpose=True) * 2 - 1.0
            img = torch.tensor(img, dtype=torch.float32).cuda()[None, ...]

            if src_smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.tensor(img_hmr, dtype=torch.float32).cuda()[None, ...]
                src_smpl = self.hmr(img_hmr)
            else:
                src_smpl = torch.tensor(src_smpl, dtype=torch.float32).cuda()[None, ...]

            # source process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
            src_info = self.hmr.get_details(src_smpl)
            src_f2verts, src_fim, src_wim = self.render.render_fim_wim(src_info['cam'], src_info['verts'])
            # src_f2pts = src_f2verts[:, :, :, 0:2]
            src_info['fim'] = src_fim
            src_info['wim'] = src_wim
            src_info['cond'], _ = self.render.encode_fim(src_info['cam'], src_info['verts'], fim=src_fim,
                                                         transpose=True)
            src_info['f2verts'] = src_f2verts
            src_info['p2verts'] = src_f2verts[:, :, :, 0:2]
            src_info['p2verts'][:, :, :, 1] *= -1
            # add image to source info
            src_info['img'] = img
            src_info['image'] = ori_img

            # 2. process the src inputs
            if self.detector is not None:
                bbox, src_bg_mask = self.detector.inference(img[0])
            else:
                src_bg_mask = util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.bg_ks, mode='dilate')

            src_info['bg'] = self.bgnet(img, masks=src_bg_mask, only_x=True)
            src_ft_mask = util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.ft_ks, mode='dilate')
            src_inputs = torch.cat([img * src_ft_mask, src_info['cond']], dim=1)
            src_info['feats'] = self.generator.src_model.inference(src_inputs)

            self.src_info = src_info

            if visualizer is not None:
                visualizer.vis_named_img('src', img)
                visualizer.vis_named_img('src_fim', src_info['fim'])
                visualizer.vis_named_img('bg', src_info['bg'])
                visualizer.vis_named_img('src_bg_mask', src_bg_mask)

            if output_path:
                cv_utils.save_cv2_img(src_info['image'], output_path, image_size=self._opt.image_size)

    def _extract_smpls(self, input_file):
        img = cv_utils.read_cv2_img(input_file)
        img = cv_utils.transform_img(img, image_size=224) * 2 - 1.0  # hmr receive [-1, 1]
        img = img.transpose((2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32).cuda()[None, ...]
        theta = self.hmr(img)[-1]

        return theta

    def inference(self, tgt_paths, tgt_smpls=None, cam_strategy='smooth', output_dir='', visualizer=None):
        length = len(tgt_paths)

        outputs = []
        for t in range(length):
            tgt_path = tgt_paths[t]
            tgt_smpl = tgt_smpls[t] if tgt_smpls is not None else None

            tsf_inputs = self.transfer_params(tgt_path, tgt_smpl, cam_strategy, t=t)

            with torch.no_grad():
                preds = self.forward(tsf_inputs, self.T, visualizer=visualizer)
                outputs.append(preds)

            if visualizer is not None:
                gt = cv_utils.transform_img(self.tsf_info['image'], image_size=self._opt.image_size, transpose=True)
                visualizer.vis_named_img('tsf_img', tsf_inputs[:, 0:3])
                visualizer.vis_named_img('pred_' + cam_strategy, preds)
                visualizer.vis_named_img('gt', gt[None, ...], denormalize=False)

            if output_dir:
                preds = preds[0].permute(1, 2, 0)
                preds = preds.cpu().numpy()
                filename = os.path.split(tgt_path)[-1]

                cv_utils.save_cv2_img(preds, os.path.join(output_dir, 'pred_' + filename), normalize=True)
                cv_utils.save_cv2_img(self.tsf_info['image'], os.path.join(output_dir, 'gt_' + filename),
                                      image_size=self._opt.image_size)

            print('{} / {}'.format(t, length))

        return outputs

    def swap_smpl(self, src_cam, src_shape, tgt_smpl, cam_strategy='smooth'):
        tgt_cam = tgt_smpl[:, 0:3].contiguous()
        pose = tgt_smpl[:, 3:75].contiguous()

        # TODO, need more tricky ways
        if cam_strategy == 'smooth':

            cam = src_cam.clone()
            delta_xy = tgt_cam[:, 1:] - self.first_cam[:, 1:]
            cam[:, 1:] += delta_xy

        elif cam_strategy == 'source':
            cam = src_cam
        else:
            cam = tgt_cam

        tsf_smpl = torch.cat([cam, pose, src_shape], dim=1)

        return tsf_smpl

    def transfer_params(self, tgt_path, tgt_smpl=None, cam_strategy='smooth', t=0):
        with torch.no_grad():
            # get source info
            src_info = self.src_info

            ori_img = cv_utils.read_cv2_img(tgt_path)
            if tgt_smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.tensor(img_hmr, dtype=torch.float32).cuda()[None, ...]
                tgt_smpl = self.hmr(img_hmr)
            else:
                tgt_smpl = torch.tensor(tgt_smpl, dtype=torch.float32).cuda()[None, ...]

            if t == 0 and cam_strategy == 'smooth':
                self.first_cam = tgt_smpl[:, 0:3].clone()

            # get transfer smpl
            tsf_smpl = self.swap_smpl(src_info['cam'], src_info['shape'], tgt_smpl, cam_strategy=cam_strategy)
            # transfer process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
            tsf_info = self.hmr.get_details(tsf_smpl)

            tsf_f2verts, tsf_fim, tsf_wim = self.render.render_fim_wim(tsf_info['cam'], tsf_info['verts'])
            # src_f2pts = src_f2verts[:, :, :, 0:2]
            tsf_info['fim'] = tsf_fim
            tsf_info['wim'] = tsf_wim
            tsf_info['cond'], _ = self.render.encode_fim(tsf_info['cam'], tsf_info['verts'], fim=tsf_fim,
                                                         transpose=True)

            T = self.render.cal_bc_transform(src_info['p2verts'], tsf_fim, tsf_wim)
            tsf_img = F.grid_sample(src_info['img'], T)
            tsf_inputs = torch.cat([tsf_img, tsf_info['cond']], dim=1)

            # add target image to tsf info
            tsf_info['image'] = ori_img

            self.T = T
            self.tsf_info = tsf_info

            return tsf_inputs

    def forward(self, tsf_inputs, T, visualizer=None):
        bg_img = self.src_info['bg']
        src_encoder_outs, src_resnet_outs = self.src_info['feats']

        tsf_color, tsf_mask = self.generator.inference(src_encoder_outs, src_resnet_outs, tsf_inputs, T)

        if self._opt.morph_mask:
            tsf_mask = 1 - util.morph(tsf_inputs[:, -1:, :, :], ks=self._opt.ft_ks, mode='dilate')
        pred_imgs = tsf_mask * bg_img + (1 - tsf_mask) * tsf_color

        if visualizer is not None:
            visualizer.vis_named_img('tsf_mask', tsf_mask)
        return pred_imgs
