import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
import utils.util as util
from .models import BaseModel
from networks.networks import NetworksFactory, HumanModelRecovery, Vgg19, VGGLoss, FaceLoss, StyleLoss
# from utils.nmr import SMPLRendererTrainer
from utils.nmr import SMPLRenderer
import ipdb


class BodyRecoveryFlow(torch.nn.Module):

    def __init__(self, opt):
        super(BodyRecoveryFlow, self).__init__()
        self._name = 'BodyRecoveryFlow'
        self._opt = opt

        # create networks
        self._init_create_networks()

    def _create_hmr(self):
        hmr = HumanModelRecovery(smpl_pkl_path=self._opt.smpl_model)
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def _create_render(self, faces):
        render = SMPLRenderer(map_name=self._opt.map_name,
                              uv_map_path=self._opt.uv_mapping,
                              tex_size=self._opt.tex_size,
                              image_size=self._opt.image_size, fill_back=False,
                              anti_aliasing=True, background_color=(0, 0, 0), has_front=False)

        return render

    def _init_create_networks(self):
        # hmr and render
        self._hmr = self._create_hmr()
        self._render = self._create_render(self._hmr.smpl.faces)

    def forward(self, src_img, src_smpl, ref_smpl):
        # get smpl information
        src_info = self._hmr.get_details(src_smpl)
        ref_info = self._hmr.get_details(ref_smpl)

        # process source inputs
        src_f2verts, src_fim, _ = self._render.render_fim_wim(src_info['cam'], src_info['verts'])
        src_f2verts = src_f2verts[:, :, :, 0:2]
        src_f2verts[:, :, :, 1] *= -1
        src_cond, _ = self._render.encode_fim(src_info['cam'], src_info['verts'], fim=src_fim, transpose=True)
        src_bg_mask = util.morph(src_cond[:, -1:, :, :], ks=15, mode='erode')
        src_crop_mask = util.morph(src_cond[:, -1:, :, :], ks=3, mode='erode')

        # bg input
        input_G_bg = torch.cat([src_img * src_bg_mask, src_bg_mask], dim=1)

        # src input
        input_G_src = torch.cat([src_img * (1 - src_crop_mask), src_cond], dim=1)

        # process reference inputs
        _, ref_fim, ref_wim = self._render.render_fim_wim(ref_info['cam'], ref_info['verts'])
        ref_cond, _ = self._render.encode_fim(ref_info['cam'], ref_info['verts'], fim=ref_fim, transpose=True)
        T = self._render.cal_bc_transform(src_f2verts, ref_fim, ref_wim)
        syn_img = F.grid_sample(src_img, T)
        input_G_tsf = torch.cat([syn_img, ref_cond], dim=1)

        # masks
        tsf_crop_mask = util.morph(ref_cond[:, -1:, :, :], ks=3, mode='erode')
        bg_mask = torch.cat([src_crop_mask, tsf_crop_mask], dim=0)
        conds = torch.cat([src_cond, ref_cond], dim=1)

        return syn_img, input_G_bg, input_G_src, input_G_tsf, T, bg_mask, conds, ref_info['j2d']


class Impersonator(BaseModel):
    def __init__(self, opt):
        super(Impersonator, self).__init__(opt)
        self._name = 'Impersonator'

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
        multi_gpus = len(self._gpu_ids) > 1

        # body recovery Flow
        self._bdr = BodyRecoveryFlow(opt=self._opt)
        if multi_gpus:
            self._bdr = torch.nn.DataParallel(self._bdr)

        self._bdr.eval()
        self._bdr.cuda()

        # generator network
        self._G = self._create_generator()
        self._G.init_weights()
        self._G = torch.nn.DataParallel(self._G)
        self._G.cuda()

        # discriminator network
        self._D = self._create_discriminator()
        self._D.init_weights()
        self._D = torch.nn.DataParallel(self._D)
        self._D.cuda()

    def _create_generator(self):
        return NetworksFactory.get_by_name(self._opt.gen_name, bg_dim=4, src_dim=3+self._G_cond_nc,
                                           tsf_dim=3+self._G_cond_nc, repeat_num=self._opt.repeat_num)

    def _create_discriminator(self):
        return NetworksFactory.get_by_name('discriminator_patch_gan', input_nc=3 + self._D_cond_nc,
                                           norm_type=self._opt.norm_type, ndf=64, n_layers=4, use_sigmoid=False)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=(self._opt.G_adam_b1, self._opt.G_adam_b2))
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=(self._opt.D_adam_b1, self._opt.D_adam_b2))

    def _init_prefetch_inputs(self):
        self._input_src_img = None
        self._input_src_smpl = None

        self._input_desired_img = None
        self._input_desired_smpl = None

        self._input_real_imgs = None

        self._bg_mask = None
        self._input_src = None
        self._input_tsf = None
        self._input_G_bg = None
        self._input_G_src = None
        self._input_G_tsf = None
        self._input_cond = None
        self._j2d = None

        self._T = None

    def _init_losses(self):
        # define loss functions
        multi_gpus = len(self._gpu_ids) > 1
        self._crt_l1 = torch.nn.L1Loss()

        if self._opt.mask_bce:
            self._crt_mask = torch.nn.BCELoss()
        else:
            self._crt_mask = torch.nn.MSELoss()

        vgg_net = Vgg19()
        if self._opt.use_vgg:
            self._criterion_vgg = VGGLoss(vgg=vgg_net)
            if multi_gpus:
                self._criterion_vgg = torch.nn.DataParallel(self._criterion_vgg)
            self._criterion_vgg.cuda()

        if self._opt.use_style:
            self._criterion_style = StyleLoss(feat_extractors=vgg_net)
            if multi_gpus:
                self._criterion_style = torch.nn.DataParallel(self._criterion_style)
            self._criterion_style.cuda()

        if self._opt.use_face:
            self._criterion_face = FaceLoss(pretrained_path=self._opt.face_model)
            if multi_gpus:
                self._criterion_face = torch.nn.DataParallel(self._criterion_face)
            self._criterion_face.cuda()

        # init losses G
        self._loss_g_l1 = self._Tensor([0])
        self._loss_g_vgg = self._Tensor([0])
        self._loss_g_style = self._Tensor([0])
        self._loss_g_face = self._Tensor([0])
        self._loss_g_adv = self._Tensor([0])
        self._loss_g_smooth = self._Tensor([0])
        self._loss_g_mask = self._Tensor([0])
        self._loss_g_mask_smooth = self._Tensor([0])

        # init losses D
        self._d_real = self._Tensor([0])
        self._d_fake = self._Tensor([0])

    def set_input(self, input):

        with torch.no_grad():
            images = input['images']
            smpls = input['smpls']
            self._input_src_img = images[:, 0, ...].contiguous().cuda()
            self._input_src_smpl = smpls[:, 0, ...].contiguous().cuda()
            self._input_desired_img = images[:, 1, ...].contiguous().cuda()
            self._input_desired_smpl = smpls[:, 1, ...].contiguous().cuda()

            syn_img, input_G_bg, input_G_src, input_G_tsf, T, bg_mask, conds, j2ds = self._bdr(
                self._input_src_img, self._input_src_smpl, self._input_desired_smpl)

            self._input_tsf = syn_img
            self._input_G_bg = input_G_bg
            self._input_G_src = input_G_src
            self._input_G_tsf = input_G_tsf
            self._input_cond = conds
            self._bg_mask = bg_mask
            self._T = T
            self._j2d = j2ds
            self._input_real_imgs = torch.cat([self._input_src_img, self._input_desired_img], dim=0)

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

        fake_src_imgs = fake_src_mask * fake_bg + (1 - fake_src_mask) * fake_src_color
        fake_tsf_imgs = fake_tsf_mask * fake_bg + (1 - fake_tsf_mask) * fake_tsf_color

        fake_masks = torch.cat([fake_src_mask, fake_tsf_mask], dim=0)
        fake_imgs = torch.cat([fake_src_imgs, fake_tsf_imgs], dim=0)

        # keep data for visualization
        if keep_data_for_visuals:
            self.visual_imgs(fake_bg, fake_imgs, fake_tsf_color, fake_masks)

        return fake_tsf_imgs, fake_imgs, fake_masks

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
        if self._is_train:
            # convert tensor to variables
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
        fake_input_D = torch.cat([fake_tsf_imgs, self._input_cond], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        self._loss_g_l1 = self._crt_l1(fake_imgs, self._input_real_imgs) * self._opt.lambda_lp

        if self._opt.use_vgg:
            self._loss_g_vgg = torch.mean(self._criterion_vgg(fake_imgs, self._input_real_imgs)) * self._opt.lambda_vgg

        if self._opt.use_style:
            self._loss_g_style = torch.mean(self._criterion_style(fake_imgs,
                                                                  self._input_real_imgs)) * self._opt.lambda_style

        if self._opt.use_face:
            self._loss_g_face = torch.mean(self._criterion_face(fake_tsf_imgs, self._input_desired_img,
                                                                self._j2d, self._j2d)) * self._opt.lambda_face
        # loss mask
        self._loss_g_mask = self._crt_mask(fake_masks, self._bg_mask) * self._opt.lambda_mask

        if self._opt.lambda_mask_smooth != 0:
            self._loss_g_mask_smooth = self._compute_loss_smooth(fake_masks) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_l1 + self._loss_g_vgg + self._loss_g_style + self._loss_g_face + \
               self._loss_g_mask + self._loss_g_mask_smooth

    def _optimize_D(self, fake_tsf_imgs):
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), self._input_cond], dim=1)
        real_input_D = torch.cat([self._input_desired_img, self._input_cond], dim=1)

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
        return torch.mean(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.mean(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_l1', self._loss_g_l1.item()),
                                 ('g_vgg', self._loss_g_vgg.item()),
                                 ('g_style', self._loss_g_style.item()),
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
        visuals['1_real_img'] = self._vis_real_img
        visuals['2_input_tsf'] = self._vis_tsf
        visuals['3_fake_bg'] = self._vis_fake_bg

        # outputs
        visuals['4_fake_img'] = self._vis_fake_img
        visuals['5_fake_color'] = self._vis_fake_color
        visuals['6_fake_mask'] = self._vis_fake_mask

        # batch outputs
        visuals['7_batch_real_img'] = self._vis_batch_real_img
        visuals['8_batch_fake_img'] = self._vis_batch_fake_img

        return visuals

    def visual_imgs(self, fake_bg, fake_imgs, fake_color, fake_masks):
        self._vis_real_img = util.tensor2im(self._input_real_imgs)

        ids = fake_imgs.shape[0] // 2
        self._vis_tsf = util.tensor2im(self._input_tsf)
        self._vis_fake_bg = util.tensor2im(fake_bg.detach())
        self._vis_fake_color = util.tensor2im(fake_color.detach())
        self._vis_fake_img = util.tensor2im(fake_imgs[ids].detach())
        self._vis_fake_mask = util.tensor2maskim(fake_masks[ids].detach())

        self._vis_batch_real_img = util.tensor2im(self._input_real_imgs, idx=-1)
        self._vis_batch_fake_img = util.tensor2im(fake_imgs.detach(), idx=-1)

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
        self._load_network(self._G.module, 'G', load_epoch, need_module=False)

        if self._is_train:
            # load D
            self._load_network(self._D.module, 'D', load_epoch, need_module=False)

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

    def debug_fim_transfer(self, visualizer):
        import torch.nn.functional as F
        import utils.util as util

        src_fims = util.plot_fim_enc(fim_enc=self._input_G_src[:, 3:], map_name=self._opt.map_name)
        tsf_fims = util.plot_fim_enc(fim_enc=self._input_G_tsf[:, 3:], map_name=self._opt.map_name)

        visualizer.vis_named_img('src_fims', src_fims)
        visualizer.vis_named_img('tsf_fims', tsf_fims)

        visualizer.vis_named_img('src_imgs', self._input_src_img)
        visualizer.vis_named_img('tsf_imgs', self._input_desired_img)
        ipdb.set_trace()
