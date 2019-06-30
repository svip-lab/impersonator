import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
import utils.util as util
import utils.cv_utils as cv_utils
from .models import BaseModel
from networks.networks import NetworksFactory, HMRLoss, VGGLoss, SphereFaceLoss
from utils.nmr import SMPLRenderer
import ipdb


class ConcatBaseline(BaseModel):
    def __init__(self, opt):
        super(ConcatBaseline, self).__init__(opt)
        self._name = 'ConcatBaseline'

        # create networks
        self._init_create_networks()

        # init train variables and losses
        if self._is_train:
            self._init_train_vars()
            self._init_losses()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # prefetch variables
        self._init_prefetch_inputs()

    def _init_create_networks(self):
        # generator network
        self._G = self._create_generator()
        self._G.init_weights()
        if len(self._gpu_ids) > 1:
            self._G = torch.nn.DataParallel(self._G)
        self._G.cuda()

        # discriminator network
        self._D = self._create_discriminator()
        self._D.init_weights()
        if len(self._gpu_ids) > 1:
            self._D = torch.nn.DataParallel(self._D)
        self._D.cuda()

        self._criterion_hmr = HMRLoss(pretrain_model=self._opt.hmr_model, smpl_pkl_path=self._opt.smpl_model).cuda()

        self._render = SMPLRenderer(faces=self._criterion_hmr.hmr.smpl.faces,
                                    map_name=self._opt.map_name,
                                    uv_map_path=self._opt.uv_mapping,
                                    tex_size=self._opt.tex_size,
                                    image_size=self._opt.image_size, fill_back=True,
                                    anti_aliasing=True, background_color=(0, 0, 0), has_front_map=False)

    def _create_generator(self):
        return NetworksFactory.get_by_name('concat', bg_dim=4, src_dim=self._G_cond_nc, tsf_dim=self._G_cond_nc, repeat_num=6)

    def _create_discriminator(self):
        return NetworksFactory.get_by_name('discriminator_patch_gan', input_nc=3 + self._D_cond_nc,
                                           ndf=64, n_layers=4, use_sigmoid=False)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=[self._opt.D_adam_b1, self._opt.D_adam_b2])

    def _init_prefetch_inputs(self):
        self._input_src_img = None
        self._input_src_smpl = None
        self._input_src_cond = None

        self._input_desired_img = None
        self._input_desired_smpl = None
        self._input_desired_cond = None

        self._input_real_imgs = None

        self._bg_mask = None
        self._input_G_bg = None
        self._input_G_fg = None

        self._input_D = None
        self._src_info = None
        self._tgt_info = None

    def _init_losses(self):
        # define loss functions
        self._criterion_l1 = torch.nn.L1Loss().cuda()

        if self._opt.use_vgg:
            self._criterion_vgg = VGGLoss().cuda()

        if self._opt.use_face:
            self._criterion_face = SphereFaceLoss().cuda()

        # init losses G
        self._loss_g_l1 = self._Tensor([0])
        self._loss_g_vgg = self._Tensor([0])
        self._loss_g_face = self._Tensor([0])
        self._loss_g_adv = self._Tensor([0])
        self._loss_g_smooth = self._Tensor([0])
        self._loss_g_mask = self._Tensor([0])
        self._loss_g_mask_smooth = self._Tensor([0])

        # init losses D
        self._loss_d_real = self._Tensor([0])
        self._loss_d_fake = self._Tensor([0])

    def morph(self, src_bg_mask, ks, mode='erode'):
        n_ks = ks ** 2
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32).cuda()
        out = F.conv2d(src_bg_mask, kernel, padding=ks // 2)

        if mode == 'erode':
            out = (out == n_ks).float()
        else:
            out = (out >= 1).float()

        return out

    def set_input_cond(self, is_train=True):
        # source process
        # source process
        src_info = self._criterion_hmr.hmr.get_details(self._input_src_smpl)
        src_rd, src_info['tex'] = self._render.forward(src_info['cam'], src_info['verts'],
                                                       self._input_src_img, is_uv_sampler=False,
                                                       reverse_yz=True, get_fim=False)

        self._input_src_cond, src_info['fim'] = self._render.encode_fim(src_info['cam'], src_info['verts'], transpose=True)
        src_bg_mask = self.morph(self._input_src_cond[:, -1:, :, :], ks=15, mode='erode')

        # bg input
        self._input_G_bg = torch.cat([self._input_src_img * src_bg_mask, src_bg_mask], dim=1)

        # transfer
        tgt_info = self._criterion_hmr.hmr.get_details(self._input_desired_smpl)
        self._input_desired_cond, tgt_info['fim'] = self._render.encode_fim(tgt_info['cam'], tgt_info['verts'], transpose=True)

        self._input_G_fg = torch.cat([self._input_src_img, self._input_src_cond, self._input_desired_cond], dim=1)

        if is_train:
            tsf_crop_mask = self.morph(self._input_desired_cond[:, -1:, :, :], ks=3, mode='erode')
            self._bg_mask = tsf_crop_mask
            self._input_D = torch.cat([self._input_desired_img, self._input_src_cond, self._input_desired_cond], dim=1)
            self._input_real_imgs = self._input_desired_img

        self._src_info = src_info
        self._tgt_info = tgt_info

    def set_input(self, input):

        with torch.no_grad():
            images = input['images']
            smpls = input['smpls']
            self._input_src_img = images[:, 0, ...].contiguous().cuda()
            self._input_src_smpl = smpls[:, 0, ...].contiguous().cuda()
            self._input_desired_img = images[:, 1, ...].contiguous().cuda()
            self._input_desired_smpl = smpls[:, 1, ...].contiguous().cuda()

            self.set_input_cond()

    def set_test_input(self, input):
        with torch.no_grad():
            self._input_src_img = input['src_img']
            self._input_src_smpl = input['src_smpl']
            self._input_desired_smpl = input['desired_smpl']

            self.set_input_cond(is_train=False)

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        # generate fake images
        fake_bg, fake_tsf_color, fake_tsf_mask = self._G.forward(self._input_G_bg, self._input_G_fg)
        fake_tsf_mask = self._do_if_necessary_saturate_mask(fake_tsf_mask, saturate=self._opt.do_saturate_mask)
        fake_tsf_imgs = fake_tsf_mask * fake_bg + (1 - fake_tsf_mask) * fake_tsf_color

        fake_masks = fake_tsf_mask
        fake_imgs =  fake_tsf_imgs

        # keep data for visualization
        if keep_data_for_visuals:
            self.transfer_imgs(fake_bg, fake_imgs, fake_tsf_color, fake_masks)

        return fake_tsf_imgs, fake_imgs, fake_masks

    def optimize_parameters(self, train_generator=True, keep_data_for_visuals=False):
        if self._is_train:
            # convert tensor to variables
            self._B = self._input_src_img.size(0)

            # run
            fake_tsf_imgs, fake_imgs, fake_masks = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            loss_G = self._optimize_G(fake_tsf_imgs, fake_imgs, fake_masks)

            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

            # train D
            if train_generator:
                loss_D = self._optimize_D(fake_tsf_imgs)
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

    def _optimize_G(self, fake_tsf_imgs, fake_imgs, fake_masks):
        # D(G(Ic1, c2)*M, c2) masked
        fake_input_D = torch.cat([fake_tsf_imgs, self._input_src_cond, self._input_desired_cond], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        # l_cyc(G(Ic1,c2)*M, Ic2)
        self._loss_g_l1 = self._criterion_l1(fake_imgs, self._input_real_imgs) * self._opt.lambda_lp

        if self._opt.use_vgg:
            self._loss_g_vgg = self._criterion_vgg(fake_imgs, self._input_real_imgs) * self._opt.lambda_vgg

        if self._opt.use_face:
            self._loss_g_face = self._criterion_face(fake_tsf_imgs, self._input_desired_img,
                                                     self._tgt_info['j2d'], self._tgt_info['j2d']) * self._opt.lambda_face
        # loss mask
        self._loss_g_mask = torch.mean((fake_masks - self._bg_mask) ** 2) * self._opt.lambda_mask
        self._loss_g_mask_smooth = self._compute_loss_smooth(fake_masks) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_l1 + self._loss_g_vgg + self._loss_g_face + \
               self._loss_g_mask + self._loss_g_mask_smooth

    def _optimize_D(self, fake_tsf_imgs):
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), self._input_src_cond, self._input_desired_cond], dim=1)

        d_real_outs = self._D.forward(self._input_D)
        d_fake_outs = self._D.forward(fake_input_D)

        self._loss_d_real = self._compute_loss_D(d_real_outs, 1) * self._opt.lambda_D_prob
        self._loss_d_fake = self._compute_loss_D(d_fake_outs, -1) * self._opt.lambda_D_prob

        # combine losses
        return self._loss_d_real + self._loss_d_fake

    def _compute_loss_D(self, x, y):
        return torch.mean((x - y) ** 2)

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_l1', self._loss_g_l1.item()),
                                 ('g_vgg', self._loss_g_vgg.item()),
                                 ('g_face', self._loss_g_face.item()),
                                 ('g_adv', self._loss_g_adv.item()),
                                 ('g_mask', self._loss_g_mask.item()),
                                 ('g_mask_smooth', self._loss_g_mask_smooth.item()),
                                 ('d_real', self._loss_d_real.item()),
                                 ('d_fake', self._loss_d_fake.item())])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # inputs
        visuals['1_real_img'] = self._vis_real_img
        visuals['3_fake_bg'] = self._vis_fake_bg

        # outputs
        visuals['4_fake_img'] = self._vis_fake_img
        visuals['5_fake_color'] = self._vis_fake_color
        visuals['6_fake_mask'] = self._vis_fake_mask

        # batch outputs
        visuals['7_batch_real_img'] = self._vis_batch_real_img
        visuals['8_batch_fake_img'] = self._vis_batch_fake_img

        return visuals

    def transfer_imgs(self, fake_bg, fake_imgs, fake_color, fake_masks):
        self._vis_real_img = util.tensor2im(self._input_real_imgs)

        ids = fake_imgs.shape[0] // 2
        self._vis_fake_bg = util.tensor2im(fake_bg.data)
        self._vis_fake_color = util.tensor2im(fake_color.data)
        self._vis_fake_img = util.tensor2im(fake_imgs[ids].data)
        self._vis_fake_mask = util.tensor2maskim(fake_masks[ids].data)

        self._vis_batch_real_img = util.tensor2im(self._input_real_imgs, idx=-1)
        self._vis_batch_fake_img = util.tensor2im(fake_imgs.data, idx=-1)

    def save(self, label):
        # save networks
        self._save_network(self._G, 'G', label)
        self._save_network(self._D, 'D', label)

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_D, 'D', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch)

        if self._is_train:
            # load D
            self._load_network(self._D, 'D', load_epoch)

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

    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m

    def personalize(self, src_path, src_smpl=None, output_path=''):

        with torch.no_grad():
            ori_img = cv_utils.read_cv2_img(src_path)

            # resize image and convert the color space from [0, 255] to [-1, 1]
            img = cv_utils.transform_img(ori_img, self._opt.image_size, transpose=True) * 2 - 1.0
            img = torch.FloatTensor(img).cuda()[None, ...]

            if src_smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.FloatTensor(img_hmr).cuda()[None, ...]
                src_smpl = self._criterion_hmr.hmr(img_hmr)[-1]
            else:
                src_smpl = util.to_tensor(src_smpl).cuda()[None, ...]

            # source process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
            src_info = self._criterion_hmr.hmr.get_details(src_smpl)

            # add image to source info
            src_info['image'] = img

            # add texture into source info
            _, src_info['tex'] = self._render.forward(src_info['cam'], src_info['verts'],
                                                      img, is_uv_sampler=False, reverse_yz=True, get_fim=False)

            # add pose condition and face index map into source info
            src_info['cond'], src_info['fim'] = self._render.encode_fim(src_info['cam'],
                                                                        src_info['verts'], transpose=True)

            src_bg_mask = self.morph(src_info['cond'][:, -1:, :, :], ks=15, mode='erode')

            # bg input
            bg_inputs = torch.cat([img * src_bg_mask, src_bg_mask], dim=1)
            src_info['bg'] = self._G.bg_model(bg_inputs)

            self.src_info = src_info

            if output_path:
                cv_utils.save_cv2_img(ori_img, output_path, image_size=self._opt.image_size)

    def transfer(self, tgt_path, tgt_smpl=None):
        with torch.no_grad():
            # get source info
            src_info = self.src_info

            ori_img = cv_utils.read_cv2_img(tgt_path)
            if tgt_smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.FloatTensor(img_hmr).cuda()[None, ...]
                tgt_smpl = self._criterion_hmr.hmr(img_hmr)[-1]
            else:
                tgt_smpl = util.to_tensor(tgt_smpl).cuda()[None, ...]

            tgt_info = self._criterion_hmr.hmr.get_details(tgt_smpl)
            tgt_info['cond'], tgt_info['fim'] = self._render.encode_fim(tgt_info['cam'], tgt_info['verts'],
                                                                        transpose=True)

            tsf_inputs = torch.cat([src_info['image'], src_info['cond'], tgt_info['cond']], dim=1)
            tsf_color, tsf_mask = self._G.inference(tsf_inputs)
            pred_imgs = tsf_mask * src_info['bg'] + (1 - tsf_mask) * tsf_color

            return pred_imgs, ori_img

    def imitate(self, tgt_paths, tgt_smpls=None, output_dir='', visualizer=None, cam_strategy=None):
        length = len(tgt_paths)

        outputs = []
        for t in range(length):

            tgt_path = tgt_paths[t]
            tgt_smpl = tgt_smpls[t] if tgt_smpls is not None else None

            preds, ori_img = self.transfer(tgt_path, tgt_smpl)

            if visualizer is not None:
                visualizer.vis_named_img('pred_2', preds)

            if output_dir:
                preds = preds[0].permute(1, 2, 0)
                preds = preds.cpu().numpy()
                filename = os.path.split(tgt_path)[-1]

                cv_utils.save_cv2_img(preds, os.path.join(output_dir, 'pred_' + filename), normalize=True)
                cv_utils.save_cv2_img(ori_img, os.path.join(output_dir, 'gt_' + filename),
                                      image_size=self._opt.image_size)
            outputs.append(preds)
            print('{} / {}'.format(t, length))

        return outputs


class TextureWarpingBaseline(BaseModel):
    def __init__(self, opt):
        super(TextureWarpingBaseline, self).__init__(opt)
        self._name = 'TextureWarpingBaseline'

        # create networks
        self._init_create_networks()

        # init train variables and losses
        if self._is_train:
            self._init_train_vars()
            self._init_losses()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # prefetch variables
        self._init_prefetch_inputs()

    def _init_create_networks(self):
        # generator network
        self._G = self._create_generator()
        self._G.init_weights()
        if len(self._gpu_ids) > 1:
            self._G = torch.nn.DataParallel(self._G)
        self._G.cuda()

        # discriminator network
        self._D = self._create_discriminator()
        self._D.init_weights()
        if len(self._gpu_ids) > 1:
            self._D = torch.nn.DataParallel(self._D)
        self._D.cuda()

        self._criterion_hmr = HMRLoss(pretrain_model=self._opt.hmr_model, smpl_pkl_path=self._opt.smpl_model).cuda()

        self._render = SMPLRenderer(faces=self._criterion_hmr.hmr.smpl.faces,
                                    map_name=self._opt.map_name,
                                    uv_map_path=self._opt.uv_mapping,
                                    tex_size=self._opt.tex_size,
                                    image_size=self._opt.image_size, fill_back=True,
                                    anti_aliasing=True, background_color=(0, 0, 0), has_front_map=False)

    def _create_generator(self):
        return NetworksFactory.get_by_name('concat', bg_dim=4, src_dim=0, tsf_dim=self._G_cond_nc,
                                           repeat_num=6)

    def _create_discriminator(self):
        return NetworksFactory.get_by_name('discriminator_patch_gan', input_nc=3 + self._D_cond_nc,
                                           ndf=64, n_layers=4, use_sigmoid=False)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=[self._opt.D_adam_b1, self._opt.D_adam_b2])

    def _init_prefetch_inputs(self):
        self._input_src_img = None
        self._input_src_smpl = None
        self._input_src_cond = None

        self._input_desired_img = None
        self._input_desired_smpl = None
        self._input_desired_cond = None

        self._input_real_imgs = None

        self._bg_mask = None
        self._input_src = None
        self._input_tsf = None
        self._input_G_bg = None
        self._input_G_src = None
        self._input_G_tsf = None
        self._input_D = None
        self._src_info = None
        self._tgt_info = None

    def _init_losses(self):
        # define loss functions
        self._criterion_l1 = torch.nn.L1Loss().cuda()

        if self._opt.use_vgg:
            self._criterion_vgg = VGGLoss().cuda()

        if self._opt.use_face:
            self._criterion_face = SphereFaceLoss().cuda()

        # init losses G
        self._loss_g_l1 = self._Tensor([0])
        self._loss_g_vgg = self._Tensor([0])
        self._loss_g_face = self._Tensor([0])
        self._loss_g_adv = self._Tensor([0])
        self._loss_g_smooth = self._Tensor([0])
        self._loss_g_mask = self._Tensor([0])
        self._loss_g_mask_smooth = self._Tensor([0])

        # init losses D
        self._loss_d_real = self._Tensor([0])
        self._loss_d_fake = self._Tensor([0])

    def morph(self, src_bg_mask, ks, mode='erode'):
        n_ks = ks ** 2
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32).cuda()
        out = F.conv2d(src_bg_mask, kernel, padding=ks // 2)

        if mode == 'erode':
            out = (out == n_ks).float()
        else:
            out = (out >= 1).float()

        return out

    def set_input_cond(self, is_train=True):
        # source process
        # source process
        src_info = self._criterion_hmr.hmr.get_details(self._input_src_smpl)
        src_rd, src_info['tex'] = self._render.forward(src_info['cam'], src_info['verts'],
                                                       self._input_src_img, is_uv_sampler=False,
                                                       reverse_yz=True, get_fim=False)

        self._input_src_cond, src_info['fim'] = self._render.encode_fim(src_info['cam'], src_info['verts'], transpose=True)
        src_bg_mask = self.morph(self._input_src_cond[:, -1:, :, :], ks=15, mode='erode')
        src_crop_mask = self.morph(self._input_src_cond[:, -1:, :, :], ks=3, mode='erode')

        # bg input
        self._input_G_bg = torch.cat([self._input_src_img * src_bg_mask, src_bg_mask], dim=1)

        # transfer
        tgt_info = self._criterion_hmr.hmr.get_details(self._input_desired_smpl)
        self._input_tsf, _ = self._render.render(tgt_info['cam'], tgt_info['verts'], src_info['tex'], reverse_yz=True, get_fim=False)
        self._input_desired_cond, tgt_info['fim'] = self._render.encode_fim(tgt_info['cam'], tgt_info['verts'], transpose=True)
        self._input_G_tsf = torch.cat([self._input_tsf, self._input_desired_cond], dim=1)

        if is_train:
            tsf_crop_mask = self.morph(self._input_desired_cond[:, -1:, :, :], ks=3, mode='erode')
            self._bg_mask = tsf_crop_mask
            self._input_D = torch.cat([self._input_desired_img, self._input_src_cond, self._input_desired_cond], dim=1)
            self._input_real_imgs = self._input_desired_img

        self._src_info = src_info
        self._tgt_info = tgt_info

    def set_input(self, input):

        with torch.no_grad():
            images = input['images']
            smpls = input['smpls']
            self._input_src_img = images[:, 0, ...].contiguous().cuda()
            self._input_src_smpl = smpls[:, 0, ...].contiguous().cuda()
            self._input_desired_img = images[:, 1, ...].contiguous().cuda()
            self._input_desired_smpl = smpls[:, 1, ...].contiguous().cuda()

            self.set_input_cond()

    def set_test_input(self, input):
        with torch.no_grad():
            self._input_src_img = input['src_img']
            self._input_src_smpl = input['src_smpl']
            self._input_desired_smpl = input['desired_smpl']

            self.set_input_cond(is_train=False)

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        # generate fake images
        fake_bg, fake_tsf_color, fake_tsf_mask = self._G.forward(self._input_G_bg, self._input_G_tsf)
        fake_tsf_mask = self._do_if_necessary_saturate_mask(fake_tsf_mask, saturate=self._opt.do_saturate_mask)
        fake_tsf_imgs = fake_tsf_mask * fake_bg + (1 - fake_tsf_mask) * fake_tsf_color

        fake_masks = fake_tsf_mask
        fake_imgs = fake_tsf_imgs

        # keep data for visualization
        if keep_data_for_visuals:
            self.transfer_imgs(fake_bg, fake_imgs, fake_tsf_color, fake_masks)

        return fake_tsf_imgs, fake_imgs, fake_masks

    def optimize_parameters(self, train_generator=True, keep_data_for_visuals=False):
        if self._is_train:
            # convert tensor to variables
            self._B = self._input_src_img.size(0)

            # run
            fake_tsf_imgs, fake_imgs, fake_masks = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            loss_G = self._optimize_G(fake_tsf_imgs, fake_imgs, fake_masks)

            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

            # train D
            if train_generator:
                loss_D = self._optimize_D(fake_tsf_imgs)
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

    def _optimize_G(self, fake_tsf_imgs, fake_imgs, fake_masks):
        # D(G(Ic1, c2)*M, c2) masked
        fake_input_D = torch.cat([fake_tsf_imgs, self._input_src_cond, self._input_desired_cond], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        # l_cyc(G(Ic1,c2)*M, Ic2)
        self._loss_g_l1 = self._criterion_l1(fake_imgs, self._input_real_imgs) * self._opt.lambda_lp

        if self._opt.use_vgg:
            self._loss_g_vgg = self._criterion_vgg(fake_imgs, self._input_real_imgs) * self._opt.lambda_vgg

        if self._opt.use_face:
            self._loss_g_face = self._criterion_face(fake_tsf_imgs, self._input_desired_img,
                                                     self._tgt_info['j2d'], self._tgt_info['j2d']) * self._opt.lambda_face
        # loss mask
        self._loss_g_mask = torch.mean((fake_masks - self._bg_mask) ** 2) * self._opt.lambda_mask
        self._loss_g_mask_smooth = self._compute_loss_smooth(fake_masks) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_l1 + self._loss_g_vgg + self._loss_g_face + \
               self._loss_g_mask + self._loss_g_mask_smooth

    def _optimize_D(self, fake_tsf_imgs):
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), self._input_src_cond, self._input_desired_cond], dim=1)

        d_real_outs = self._D.forward(self._input_D)
        d_fake_outs = self._D.forward(fake_input_D)

        self._loss_d_real = self._compute_loss_D(d_real_outs, 1) * self._opt.lambda_D_prob
        self._loss_d_fake = self._compute_loss_D(d_fake_outs, -1) * self._opt.lambda_D_prob

        # combine losses
        return self._loss_d_real + self._loss_d_fake

    def _compute_loss_D(self, x, y):
        return torch.mean((x - y) ** 2)

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_l1', self._loss_g_l1.item()),
                                 ('g_vgg', self._loss_g_vgg.item()),
                                 ('g_face', self._loss_g_face.item()),
                                 ('g_adv', self._loss_g_adv.item()),
                                 ('g_mask', self._loss_g_mask.item()),
                                 ('g_mask_smooth', self._loss_g_mask_smooth.item()),
                                 ('d_real', self._loss_d_real.item()),
                                 ('d_fake', self._loss_d_fake.item())])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # inputs
        visuals['1_real_img'] = self._vis_real_img
        # visuals['2_input_tsf'] = self._vis_tsf
        visuals['3_fake_bg'] = self._vis_fake_bg

        # outputs
        visuals['4_fake_img'] = self._vis_fake_img
        visuals['5_fake_color'] = self._vis_fake_color
        visuals['6_fake_mask'] = self._vis_fake_mask

        # batch outputs
        visuals['7_batch_real_img'] = self._vis_batch_real_img
        visuals['8_batch_fake_img'] = self._vis_batch_fake_img

        return visuals

    def transfer_imgs(self, fake_bg, fake_imgs, fake_color, fake_masks):
        self._vis_real_img = util.tensor2im(self._input_real_imgs)

        ids = fake_imgs.shape[0] // 2
        # self._vis_tsf = util.tensor2im(self._input_tsf.data)
        self._vis_fake_bg = util.tensor2im(fake_bg.data)
        self._vis_fake_color = util.tensor2im(fake_color.data)
        self._vis_fake_img = util.tensor2im(fake_imgs[ids].data)
        self._vis_fake_mask = util.tensor2maskim(fake_masks[ids].data)

        self._vis_batch_real_img = util.tensor2im(self._input_real_imgs, idx=-1)
        self._vis_batch_fake_img = util.tensor2im(fake_imgs.data, idx=-1)

    def save(self, label):
        # save networks
        self._save_network(self._G, 'G', label)
        self._save_network(self._D, 'D', label)

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_D, 'D', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch)

        if self._is_train:
            # load D
            self._load_network(self._D, 'D', load_epoch)

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

    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m

    def swap_smpl(self, src_cam, src_shape, tgt_smpl, preserve_scale=True):
        cam = tgt_smpl[:, 0:3].contiguous()
        pose = tgt_smpl[:, 3:75].contiguous()

        if preserve_scale:
            cam[:, 0] = src_cam[:, 0]
            # cam[:, 1:] = (src_cam[:, 0] / cam[:, 0]) * cam[:, 1:] + src_cam[:, 1:]
            # cam[:, 0] = src_cam[:, 0]
        else:
            cam = src_cam

        tsf_smpl = torch.cat([cam, pose, src_shape], dim=1)

        return tsf_smpl

    def personalize(self, src_path, src_smpl=None, output_path=''):

        with torch.no_grad():
            ori_img = cv_utils.read_cv2_img(src_path)

            # resize image and convert the color space from [0, 255] to [-1, 1]
            img = cv_utils.transform_img(ori_img, self._opt.image_size, transpose=True) * 2 - 1.0
            img = torch.FloatTensor(img).cuda()[None, ...]

            if src_smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.FloatTensor(img_hmr).cuda()[None, ...]
                src_smpl = self._criterion_hmr.hmr(img_hmr)[-1]
            else:
                src_smpl = util.to_tensor(src_smpl).cuda()[None, ...]

            # source process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
            src_info = self._criterion_hmr.hmr.get_details(src_smpl)

            # add image to source info
            src_info['image'] = img

            # add texture into source info
            _, src_info['tex'] = self._render.forward(src_info['cam'], src_info['verts'],
                                                      img, is_uv_sampler=False, reverse_yz=True, get_fim=False)

            # add pose condition and face index map into source info
            src_info['cond'], src_info['fim'] = self._render.encode_fim(src_info['cam'],
                                                                        src_info['verts'], transpose=True)

            src_bg_mask = self.morph(src_info['cond'][:, -1:, :, :], ks=15, mode='erode')

            # bg input
            bg_inputs = torch.cat([img * src_bg_mask, src_bg_mask], dim=1)
            src_info['bg'] = self._G.bg_model(bg_inputs)

            self.src_info = src_info

            if output_path:
                cv_utils.save_cv2_img(ori_img, output_path, image_size=self._opt.image_size)

    def transfer(self, tgt_path, tgt_smpl=None):
        with torch.no_grad():
            # get source info
            src_info = self.src_info

            ori_img = cv_utils.read_cv2_img(tgt_path)
            if tgt_smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.FloatTensor(img_hmr).cuda()[None, ...]
                tgt_smpl = self._criterion_hmr.hmr(img_hmr)[-1]
            else:
                tgt_smpl = util.to_tensor(tgt_smpl).cuda()[None, ...]

            tsf_smpl = self.swap_smpl(src_info['cam'], src_info['shape'], tgt_smpl, preserve_scale=True)

            tsf_info = self._criterion_hmr.hmr.get_details(tsf_smpl)
            tsf_img, _ = self._render.render(tsf_info['cam'], tsf_info['verts'], src_info['tex'],
                                             reverse_yz=True, get_fim=False)
            tsf_info['cond'], tsf_info['fim'] = self._render.encode_fim(tsf_info['cam'], tsf_info['verts'],
                                                                        transpose=True)
            tsf_inputs = torch.cat([tsf_img, tsf_info['cond']], dim=1)
            tsf_color, tsf_mask = self._G.inference(tsf_inputs)
            pred_imgs = tsf_mask * src_info['bg'] + (1 - tsf_mask) * tsf_color

            return pred_imgs, ori_img

    def imitate(self, tgt_paths, tgt_smpls=None, output_dir='', visualizer=None, cam_strategy=None):
        length = len(tgt_paths)

        for t in range(length):

            tgt_path = tgt_paths[t]
            tgt_smpl = tgt_smpls[t] if tgt_smpls is not None else None

            preds, ori_img = self.transfer(tgt_path, tgt_smpl)

            if visualizer is not None:
                visualizer.vis_named_img('pred_2', preds)

            if output_dir:
                preds = preds[0].permute(1, 2, 0)
                preds = preds.cpu().numpy()
                filename = os.path.split(tgt_path)[-1]

                cv_utils.save_cv2_img(preds, os.path.join(output_dir, 'pred_' + filename), normalize=True)
                cv_utils.save_cv2_img(ori_img, os.path.join(output_dir, 'gt_' + filename),
                                      image_size=self._opt.image_size)

            print('{} / {}'.format(t, length))


class FeatureWarpingBaseline(BaseModel):
    def __init__(self, opt):
        super(FeatureWarpingBaseline, self).__init__(opt)
        self._name = 'FeatureWarpingBaseline'

        # create networks
        self._init_create_networks()

        # init train variables and losses
        if self._is_train:
            self._init_train_vars()
            self._init_losses()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # prefetch variables
        self._init_prefetch_inputs()

    def _init_create_networks(self):
        # generator network
        self._G = self._create_generator()
        self._G.init_weights()
        if len(self._gpu_ids) > 1:
            self._G = torch.nn.DataParallel(self._G)
        self._G.cuda()

        # discriminator network
        self._D = self._create_discriminator()
        self._D.init_weights()
        if len(self._gpu_ids) > 1:
            self._D = torch.nn.DataParallel(self._D)
        self._D.cuda()

        self._criterion_hmr = HMRLoss(pretrain_model=self._opt.hmr_model, smpl_pkl_path=self._opt.smpl_model).cuda()

        self._render = SMPLRenderer(faces=self._criterion_hmr.hmr.smpl.faces,
                                    map_name=self._opt.map_name,
                                    uv_map_path=self._opt.uv_mapping,
                                    tex_size=self._opt.tex_size,
                                    image_size=self._opt.image_size, fill_back=True,
                                    anti_aliasing=True, background_color=(0, 0, 0), has_front_map=False)

    def _create_generator(self):
        return NetworksFactory.get_by_name(self._opt.gen_name, bg_dim=4, src_dim=3+self._G_cond_nc,
                                           tsf_dim=self._G_cond_nc, repeat_num=6)

    def _create_discriminator(self):
        return NetworksFactory.get_by_name('discriminator_patch_gan', input_nc=3 + self._D_cond_nc,
                                           ndf=64, n_layers=4, use_sigmoid=False)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=[self._opt.D_adam_b1, self._opt.D_adam_b2])

    def _init_prefetch_inputs(self):
        self._input_src_img = None
        self._input_src_smpl = None
        self._input_src_cond = None

        self._input_desired_img = None
        self._input_desired_smpl = None
        self._input_desired_cond = None

        self._input_real_imgs = None

        self._bg_mask = None
        self._input_src = None
        self._input_tsf = None
        self._input_G_bg = None
        self._input_G_src = None
        self._input_G_tsf = None
        self._input_D = None
        self._src_info = None
        self._tgt_info = None

        self._initial_T = None
        self._T = None

        self._initialize_T()

    def _init_losses(self):
        # define loss functions
        self._criterion_l1 = torch.nn.L1Loss().cuda()

        if self._opt.use_vgg:
            self._criterion_vgg = VGGLoss().cuda()

        if self._opt.use_face:
            self._criterion_face = SphereFaceLoss().cuda()

        # init losses G
        self._loss_g_l1 = self._Tensor([0])
        self._loss_g_vgg = self._Tensor([0])
        self._loss_g_face = self._Tensor([0])
        self._loss_g_adv = self._Tensor([0])
        self._loss_g_smooth = self._Tensor([0])
        self._loss_g_mask = self._Tensor([0])
        self._loss_g_mask_smooth = self._Tensor([0])

        # init losses D
        self._loss_d_real = self._Tensor([0])
        self._loss_d_fake = self._Tensor([0])

    def _initialize_T(self):
        # initialize T
        image_size = self._opt.image_size
        T = torch.zeros(image_size, image_size, 2, dtype=torch.float32).cuda() - 1.0
        self._initial_T = T

    def _transformer(self, src_cams, src_verts, src_fim, tgt_fim):
        bs = src_fim.shape[0]

        T = self._initial_T.repeat(bs, 1, 1, 1)   # (bs, image_size, image_size, 2)

        # 2. calculate occlusion flows, (bs, no, 2)
        tgt_ids = tgt_fim != -1

        # 3. calculate tgt flows, (bs, nt, 2)
        points = self._render.batch_orth_proj_idrot(src_cams, src_verts)
        f2pts = self._render.points_to_faces(points)
        bc_f2pts = self._render.compute_barycenter(f2pts)  # (bs, nf, 2)

        for i in range(bs):
            Ti = T[i]

            tgt_i = tgt_ids[i]

            # (nf, 2)
            tgt_flows = bc_f2pts[i, tgt_fim[i, tgt_i].long()]      # (nt, 2)
            Ti[tgt_i] = tgt_flows

        return T

    def morph(self, src_bg_mask, ks, mode='erode'):
        n_ks = ks ** 2
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32).cuda()
        out = F.conv2d(src_bg_mask, kernel, padding=ks // 2)

        if mode == 'erode':
            out = (out == n_ks).float()
        else:
            out = (out >= 1).float()

        return out

    def set_input_cond(self, is_train=True):
        # source process
        # source process
        src_info = self._criterion_hmr.hmr.get_details(self._input_src_smpl)
        src_rd, src_info['tex'] = self._render.forward(src_info['cam'], src_info['verts'],
                                                       self._input_src_img, is_uv_sampler=False,
                                                       reverse_yz=True, get_fim=False)

        self._input_src_cond, src_info['fim'] = self._render.encode_fim(src_info['cam'], src_info['verts'], transpose=True)
        src_bg_mask = self.morph(self._input_src_cond[:, -1:, :, :], ks=15, mode='erode')
        src_crop_mask = self.morph(self._input_src_cond[:, -1:, :, :], ks=3, mode='erode')

        # bg input
        self._input_G_bg = torch.cat([self._input_src_img * src_bg_mask, src_bg_mask], dim=1)

        # src input
        self._input_src = self._input_src_img * (1 - src_crop_mask)
        self._input_G_src = torch.cat([self._input_src, self._input_src_cond], dim=1)

        # transfer
        tgt_info = self._criterion_hmr.hmr.get_details(self._input_desired_smpl)
        self._input_tsf, _ = self._render.render(tgt_info['cam'], tgt_info['verts'], src_info['tex'], reverse_yz=True, get_fim=False)
        self._input_desired_cond, tgt_info['fim'] = self._render.encode_fim(tgt_info['cam'], tgt_info['verts'], transpose=True)
        self._input_G_tsf = self._input_desired_cond

        if is_train:
            tsf_crop_mask = self.morph(self._input_desired_cond[:, -1:, :, :], ks=3, mode='erode')
            self._bg_mask = tsf_crop_mask
            self._input_D = torch.cat([self._input_desired_img, self._input_src_cond, self._input_desired_cond], dim=1)
            self._input_real_imgs = self._input_desired_img

        self._src_info = src_info
        self._tgt_info = tgt_info

        # set transformation matrix
        self._T = self._transformer(src_info['cam'], src_info['verts'], src_fim=src_info['fim'], tgt_fim=tgt_info['fim'])

    def set_input(self, input):

        with torch.no_grad():
            images = input['images']
            smpls = input['smpls']
            self._input_src_img = images[:, 0, ...].contiguous().cuda()
            self._input_src_smpl = smpls[:, 0, ...].contiguous().cuda()
            self._input_desired_img = images[:, 1, ...].contiguous().cuda()
            self._input_desired_smpl = smpls[:, 1, ...].contiguous().cuda()

            self.set_input_cond()

    def set_test_input(self, input):
        with torch.no_grad():
            self._input_src_img = input['src_img']
            self._input_src_smpl = input['src_smpl']
            self._input_desired_smpl = input['desired_smpl']

            self.set_input_cond(is_train=False)

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
            self._G.forward(self._input_G_bg, self._input_G_src, self._input_G_tsf, T=self._T)
        fake_src_mask = self._do_if_necessary_saturate_mask(fake_src_mask, saturate=self._opt.do_saturate_mask)
        fake_tsf_mask = self._do_if_necessary_saturate_mask(fake_tsf_mask, saturate=self._opt.do_saturate_mask)

        fake_src_imgs = fake_src_mask * fake_bg + (1 - fake_src_mask) * fake_src_color
        fake_tsf_imgs = fake_tsf_mask * fake_bg + (1 - fake_tsf_mask) * fake_tsf_color

        fake_masks = torch.cat([fake_src_mask, fake_tsf_mask], dim=0)
        fake_imgs = torch.cat([fake_src_imgs, fake_tsf_imgs], dim=0)

        # keep data for visualization
        if keep_data_for_visuals:
            self.transfer_imgs(fake_bg, fake_imgs, fake_tsf_color, fake_masks)

        return fake_tsf_imgs, fake_tsf_imgs, fake_tsf_mask

    def optimize_parameters(self, train_generator=True, keep_data_for_visuals=False):
        if self._is_train:
            # convert tensor to variables
            self._B = self._input_src_img.size(0)

            # run
            fake_tsf_imgs, fake_imgs, fake_masks = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            loss_G = self._optimize_G(fake_tsf_imgs, fake_imgs, fake_masks)

            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

            # train D
            if train_generator:
                loss_D = self._optimize_D(fake_tsf_imgs)
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

    def _optimize_G(self, fake_tsf_imgs, fake_imgs, fake_masks):
        # D(G(Ic1, c2)*M, c2) masked
        fake_input_D = torch.cat([fake_tsf_imgs, self._input_src_cond, self._input_desired_cond], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        # l_cyc(G(Ic1,c2)*M, Ic2)
        self._loss_g_l1 = self._criterion_l1(fake_imgs, self._input_real_imgs) * self._opt.lambda_lp

        if self._opt.use_vgg:
            self._loss_g_vgg = self._criterion_vgg(fake_imgs, self._input_real_imgs) * self._opt.lambda_vgg

        if self._opt.use_face:
            self._loss_g_face = self._criterion_face(fake_tsf_imgs, self._input_desired_img,
                                                     self._tgt_info['j2d'], self._tgt_info['j2d']) * self._opt.lambda_face
        # loss mask
        self._loss_g_mask = torch.mean((fake_masks - self._bg_mask) ** 2) * self._opt.lambda_mask
        self._loss_g_mask_smooth = self._compute_loss_smooth(fake_masks) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_l1 + self._loss_g_vgg + self._loss_g_face + \
               self._loss_g_mask + self._loss_g_mask_smooth

    def _optimize_D(self, fake_tsf_imgs):
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), self._input_src_cond, self._input_desired_cond], dim=1)

        d_real_outs = self._D.forward(self._input_D)
        d_fake_outs = self._D.forward(fake_input_D)

        self._loss_d_real = self._compute_loss_D(d_real_outs, 1) * self._opt.lambda_D_prob
        self._loss_d_fake = self._compute_loss_D(d_fake_outs, -1) * self._opt.lambda_D_prob

        # combine losses
        return self._loss_d_real + self._loss_d_fake

    def _compute_loss_D(self, x, y):
        return torch.mean((x - y) ** 2)

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_l1', self._loss_g_l1.item()),
                                 ('g_vgg', self._loss_g_vgg.item()),
                                 ('g_face', self._loss_g_face.item()),
                                 ('g_adv', self._loss_g_adv.item()),
                                 ('g_mask', self._loss_g_mask.item()),
                                 ('g_mask_smooth', self._loss_g_mask_smooth.item()),
                                 ('d_real', self._loss_d_real.item()),
                                 ('d_fake', self._loss_d_fake.item())])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # inputs
        visuals['1_real_img'] = self._vis_real_img
        # visuals['2_input_tsf'] = self._vis_tsf
        visuals['3_fake_bg'] = self._vis_fake_bg

        # outputs
        visuals['4_fake_img'] = self._vis_fake_img
        visuals['5_fake_color'] = self._vis_fake_color
        visuals['6_fake_mask'] = self._vis_fake_mask

        # batch outputs
        visuals['7_batch_real_img'] = self._vis_batch_real_img
        visuals['8_batch_fake_img'] = self._vis_batch_fake_img

        return visuals

    def transfer_imgs(self, fake_bg, fake_imgs, fake_color, fake_masks):
        self._vis_real_img = util.tensor2im(self._input_real_imgs)

        ids = fake_imgs.shape[0] // 2
        # self._vis_tsf = util.tensor2im(self._input_tsf.data)
        self._vis_fake_bg = util.tensor2im(fake_bg.data)
        self._vis_fake_color = util.tensor2im(fake_color.data)
        self._vis_fake_img = util.tensor2im(fake_imgs[ids].data)
        self._vis_fake_mask = util.tensor2maskim(fake_masks[ids].data)

        self._vis_batch_real_img = util.tensor2im(self._input_real_imgs, idx=-1)
        self._vis_batch_fake_img = util.tensor2im(fake_imgs.data, idx=-1)

    def save(self, label):
        # save networks
        self._save_network(self._G, 'G', label)
        self._save_network(self._D, 'D', label)

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_D, 'D', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch)

        if self._is_train:
            # load D
            self._load_network(self._D, 'D', load_epoch)

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

    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m

    def swap_smpl(self, src_cam, src_shape, tgt_smpl, preserve_scale=True):
        cam = tgt_smpl[:, 0:3].contiguous()
        pose = tgt_smpl[:, 3:75].contiguous()

        if preserve_scale:
            cam[:, 0] = src_cam[:, 0]
            # cam[:, 1:] = (src_cam[:, 0] / cam[:, 0]) * cam[:, 1:] + src_cam[:, 1:]
            # cam[:, 0] = src_cam[:, 0]
        else:
            cam = src_cam

        tsf_smpl = torch.cat([cam, pose, src_shape], dim=1)

        return tsf_smpl

    def personalize(self, src_path, src_smpl=None, output_path=''):

        with torch.no_grad():
            ori_img = cv_utils.read_cv2_img(src_path)

            # resize image and convert the color space from [0, 255] to [-1, 1]
            img = cv_utils.transform_img(ori_img, self._opt.image_size, transpose=True) * 2 - 1.0
            img = torch.FloatTensor(img).cuda()[None, ...]

            if src_smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.FloatTensor(img_hmr).cuda()[None, ...]
                src_smpl = self._criterion_hmr.hmr(img_hmr)[-1]
            else:
                src_smpl = util.to_tensor(src_smpl).cuda()[None, ...]

            # source process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
            src_info = self._criterion_hmr.hmr.get_details(src_smpl)

            # add image to source info
            src_info['image'] = img

            # add texture into source info
            _, src_info['tex'] = self._render.forward(src_info['cam'], src_info['verts'],
                                                      img, is_uv_sampler=False, reverse_yz=True, get_fim=False)

            # add pose condition and face index map into source info
            src_info['cond'], src_info['fim'] = self._render.encode_fim(src_info['cam'],
                                                                        src_info['verts'], transpose=True)

            src_bg_mask = self.morph(src_info['cond'][:, -1:, :, :], ks=15, mode='erode')

            # bg input
            bg_inputs = torch.cat([img * src_bg_mask, src_bg_mask], dim=1)
            src_info['bg'] = self._G.bg_model(bg_inputs)

            # src input
            src_crop_mask = self.morph(src_info['cond'][:, -1:, :, :], ks=3, mode='erode')
            input_src = img * (1 - src_crop_mask)
            input_G_src = torch.cat([input_src, src_info['cond']], dim=1)

            src_info['feats'] = self._G.src_model.inference(input_G_src)

            self.src_info = src_info

            if output_path:
                cv_utils.save_cv2_img(ori_img, output_path, image_size=self._opt.image_size)

    def transfer(self, tgt_path, tgt_smpl=None):
        with torch.no_grad():
            # get source info
            src_info = self.src_info

            ori_img = cv_utils.read_cv2_img(tgt_path)
            if tgt_smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.FloatTensor(img_hmr).cuda()[None, ...]
                tgt_smpl = self._criterion_hmr.hmr(img_hmr)[-1]
            else:
                tgt_smpl = util.to_tensor(tgt_smpl).cuda()[None, ...]

            tsf_smpl = self.swap_smpl(src_info['cam'], src_info['shape'], tgt_smpl, preserve_scale=True)

            tsf_info = self._criterion_hmr.hmr.get_details(tsf_smpl)
            tsf_img, _ = self._render.render(tsf_info['cam'], tsf_info['verts'], src_info['tex'],
                                             reverse_yz=True, get_fim=False)
            tsf_info['cond'], tsf_info['fim'] = self._render.encode_fim(tsf_info['cam'], tsf_info['verts'],
                                                                        transpose=True)

            # transfer
            tsf_img, _ = self._render.render(tsf_info['cam'], tsf_info['verts'], src_info['tex'],
                                             reverse_yz=True, get_fim=False)
            tsf_inputs, tsf_info['fim'] = self._render.encode_fim(tsf_info['cam'], tsf_info['verts'],
                                                                  transpose=True)
            T = self._transformer(src_info['cam'], src_info['verts'], src_info['fim'], tsf_info['fim'])
            src_encoder_outs, src_resnet_outs = src_info['feats']
            tsf_color, tsf_mask = self._G.inference(src_encoder_outs, src_resnet_outs, tsf_inputs, T)
            pred_imgs = tsf_mask * src_info['bg'] + (1 - tsf_mask) * tsf_color

            return pred_imgs, ori_img

    def imitate(self, tgt_paths, tgt_smpls=None, output_dir='', visualizer=None, cam_strategy=None):
        length = len(tgt_paths)

        for t in range(length):

            tgt_path = tgt_paths[t]
            tgt_smpl = tgt_smpls[t] if tgt_smpls is not None else None

            preds, ori_img = self.transfer(tgt_path, tgt_smpl)

            if visualizer is not None:
                visualizer.vis_named_img('pred_2', preds)

            if output_dir:
                preds = preds[0].permute(1, 2, 0)
                preds = preds.cpu().numpy()
                filename = os.path.split(tgt_path)[-1]

                cv_utils.save_cv2_img(preds, os.path.join(output_dir, 'pred_' + filename), normalize=True)
                cv_utils.save_cv2_img(ori_img, os.path.join(output_dir, 'gt_' + filename),
                                      image_size=self._opt.image_size)

            print('{} / {}'.format(t, length))
