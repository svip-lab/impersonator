import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict

import utils.util as util
import utils.cv_utils as cv_utils
from models.models import BaseRunnerModel, BaseModel, BaseTrainerModel
import networks
import networks.losses as losses

import ipdb


class BodyRecoveryFlow(torch.nn.Module):

    def __init__(self, opt):
        super(BodyRecoveryFlow, self).__init__()
        self._name = 'BodyRecoveryFlow'
        self._opt = opt

        # create networks
        self._init_create_networks()

        # buffers
        initial = torch.zeros(self._opt.image_size, self._opt.image_size, 2, dtype=torch.float32) - 1.0
        self.register_buffer('initial_T', initial)

    def _create_hmr(self):
        hmr = networks.HumanModelRecovery(smpl_data=util.load_pickle_file(self._opt.smpl_model))
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def _create_render(self, faces):
        render = networks.SMPLRendererTrainer(faces=faces, map_name=self._opt.map_name,
                                              uv_map_path=self._opt.uv_mapping, tex_size=self._opt.tex_size,
                                              image_size=self._opt.image_size, fill_back=True,
                                              anti_aliasing=True, background_color=(0, 0, 0),
                                              has_front_map=self._opt.wrap_face)

        return render

    def _init_create_networks(self):
        # hmr and render
        self.hmr = self._create_hmr()
        self.render = self._create_render(self.hmr.smpl.faces)

    def _transformer(self, src_cams, src_verts, src_fim, tgt_fim):
        bs = src_fim.shape[0]

        T = self.initial_T.repeat(bs, 1, 1, 1)   # (bs, image_size, image_size, 2)

        # 2. calculate occlusion flows, (bs, no, 2)
        tgt_ids = tgt_fim != -1

        # 3. calculate tgt flows, (bs, nt, 2)
        points = self.render.batch_orth_proj_idrot(src_cams, src_verts)
        f2pts = self.render.points_to_faces(points)
        ambiguity_bcf2pts = self.render.compute_barycenter(f2pts)  # (bs, nf, 2)

        if self._opt.only_visible:
            src_ids = src_fim != -1
            initial_bcf2pts = torch.zeros_like(ambiguity_bcf2pts) - 1.0
            # initial_bcf2pts = torch.zeros_like(ambiguity_bcf2pts) - 2.0
            # src_tex = torch.zeros_like(ambiguity_tex) - 1.0  # (bs, nf, t, t, t, 3)

            for i in range(bs):
                Ti = T[i]
                bcf2pts = initial_bcf2pts[i]

                src_i = src_ids[i]
                tgt_i = tgt_ids[i]

                # (nf, 2)
                src_vis_fi = src_fim[i, src_i].unique().long()
                bcf2pts[src_vis_fi] = ambiguity_bcf2pts[i, src_vis_fi]

                tgt_vis_fi = tgt_fim[i, tgt_i].long()
                tgt_flows = bcf2pts[tgt_vis_fi]      # (nt, 2)
                Ti[tgt_i] = tgt_flows
        else:

            for i in range(bs):
                Ti = T[i]

                tgt_i = tgt_ids[i]

                # (nf, 2)
                tgt_flows = ambiguity_bcf2pts[i, tgt_fim[i, tgt_i].long()]      # (nt, 2)
                Ti[tgt_i] = tgt_flows

        return T

    def forward(self, src_img, src_smpl, ref_smpl):
        # source process
        src_info = self.hmr.get_details(src_smpl)
        src_rd, src_info['tex'] = self.render.forward(src_info['cam'], src_info['verts'],
                                                      src_img, is_uv_sampler=False,
                                                      reverse_yz=True, get_fim=False)

        src_cond, src_info['fim'] = self.render.encode_fim(src_info['cam'], src_info['verts'], transpose=True)

        if self._opt.only_visible:
            src_info['tex'] = self.render.get_visible_tex(src_info['tex'], src_info['fim'])

        src_bg_mask = self.morph(src_cond[:, -1:, :, :], ks=15, mode='erode')
        src_crop_mask = self.morph(src_cond[:, -1:, :, :], ks=3, mode='erode')

        # bg input
        input_G_bg = torch.cat([src_img * src_bg_mask, src_bg_mask], dim=1)

        # src input
        input_src = src_img * (1 - src_crop_mask)
        input_G_src = torch.cat([input_src, src_cond], dim=1)

        # transfer
        ref_info = self.hmr.get_details(ref_smpl)
        syn_img, _ = self.render.render(ref_info['cam'], ref_info['verts'], src_info['tex'], reverse_yz=True, get_fim=False)
        ref_cond, ref_info['fim'] = self.render.encode_fim(ref_info['cam'], ref_info['verts'], transpose=True)
        input_G_tsf = torch.cat([syn_img, ref_cond], dim=1)

        tsf_crop_mask = self.morph(ref_cond[:, -1:, :, :], ks=3, mode='erode')
        bg_mask = torch.cat([src_crop_mask, tsf_crop_mask], dim=0)

        conds = torch.cat([src_cond, ref_cond], dim=1)
        # set transformation matrix
        T = self._transformer(src_info['cam'], src_info['verts'], src_fim=src_info['fim'], tgt_fim=ref_info['fim'])

        if self._opt.wrap_face:
            front_mask = self.render.encode_front_fim(ref_info['fim'], transpose=True)
            return syn_img, input_G_bg, input_G_src, input_G_tsf, T, bg_mask, conds, ref_info['j2d'], front_mask
        else:
            return syn_img, input_G_bg, input_G_src, input_G_tsf, T, bg_mask, conds, ref_info['j2d']

    # @staticmethod
    # def morph(src_bg_mask, ks, mode='erode'):
    #     device = src_bg_mask.device
    #
    #     n_ks = ks ** 2
    #     kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32).to(device)
    #
    #     pad_s = ks // 2
    #     src_bg_mask_pad = F.pad(src_bg_mask, (pad_s, pad_s, pad_s, pad_s), value=1.0)
    #     # print(src_bg_mask.shape, src_bg_mask_pad.shape)
    #     out = F.conv2d(src_bg_mask_pad, kernel)
    #     # print(out.shape)
    #
    #     if mode == 'erode':
    #         out = (out == n_ks).float()
    #     else:
    #         out = (out >= 1).float()
    #
    #     return out

    @staticmethod
    def morph(src_bg_mask, ks, mode='erode'):
        n_ks = ks ** 2
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32).cuda()
        out = F.conv2d(src_bg_mask, kernel, padding=ks // 2)

        if mode == 'erode':
            out = (out == n_ks).float()
        else:
            out = (out >= 1).float()

        return out


class ImpersonatorTrainer(BaseTrainerModel):
    def __init__(self, opt):
        super(ImpersonatorTrainer, self).__init__(opt)
        self._name = 'ImpersonatorTrainer'

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
        return networks.create_by_name('res_unet', bg_dim=4, src_dim=3+self._G_cond_nc, tsf_dim=3+self._G_cond_nc)

    def _create_discriminator(self):
        return networks.create_by_name('patch', input_nc=3 + self._D_cond_nc, ndf=64, n_layers=4, use_sigmoid=False)

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
        self._front_mask = None

        self._T = None

    def _init_losses(self):
        # define loss functions
        multi_gpus = len(self._gpu_ids) > 1
        self._criterion_l1 = torch.nn.L1Loss()

        if multi_gpus:
            self._criterion_l1 = torch.nn.DataParallel(self._criterion_l1)
        self._criterion_l1.cuda()

        if self._opt.use_vgg:

            self._criterion_vgg = losses.VGGLoss()
            if multi_gpus:
                self._criterion_vgg = torch.nn.DataParallel(self._criterion_vgg)
            self._criterion_vgg.cuda()

        if self._opt.use_face:
            self._criterion_face = losses.SphereFaceLoss()
            if multi_gpus:
                self._criterion_face = torch.nn.DataParallel(self._criterion_face)
            self._criterion_face.cuda()

        # init losses G
        self._loss_g_l1 = self._Tensor([0])
        self._loss_g_vgg = self._Tensor([0])
        self._loss_g_face = self._Tensor([0])
        self._loss_g_adv = self._Tensor([0])
        self._loss_g_smooth = self._Tensor([0])
        self._loss_g_mask = self._Tensor([0])
        self._loss_g_mask_smooth = self._Tensor([0])

        # init losses D
        self._d_real = self._Tensor([0])
        self._d_fake = self._Tensor([0])

    @property
    def hmr(self):
        if 'module' in self._bdr.__dict__:
            return self._bdr.module.hmr
        else:
            return self._bdr.hmr

    def set_input(self, input):

        with torch.no_grad():
            images = input['images']
            smpls = input['smpls']
            self._input_src_img = images[:, 0, ...].contiguous().cuda()
            self._input_src_smpl = smpls[:, 0, ...].contiguous().cuda()
            self._input_desired_img = images[:, 1, ...].contiguous().cuda()
            self._input_desired_smpl = smpls[:, 1, ...].contiguous().cuda()

            if self._opt.wrap_face:
                syn_img, input_G_bg, input_G_src, input_G_tsf, T, bg_mask, conds, j2ds, front_mask = self._bdr(
                    self._input_src_img, self._input_src_smpl, self._input_desired_smpl)
                self._front_mask = front_mask
            else:
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

    def set_test_input(self, input):
        with torch.no_grad():
            self._input_src_img = input['src_img']
            self._input_src_smpl = input['src_smpl']
            self._input_desired_smpl = input['desired_smpl']

            if self._opt.wrap_face:
                syn_img, input_G_bg, input_G_src, input_G_tsf, T, bg_mask, conds, j2ds, front_mask = self._bdr(
                    self._input_src_img, self._input_src_smpl, self._input_desired_smpl)
                self._front_mask = front_mask
            else:
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
            self.transfer_imgs(fake_bg, fake_imgs, fake_tsf_color, fake_masks)

        return fake_tsf_imgs, fake_imgs, fake_masks

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
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
            if trainable:
                loss_D = self._optimize_D(fake_tsf_imgs)
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

    def _optimize_G(self, fake_tsf_imgs, fake_imgs, fake_masks):
        fake_input_D = torch.cat([fake_tsf_imgs, self._input_cond], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        self._loss_g_l1 = torch.mean(self._criterion_l1(fake_imgs, self._input_real_imgs)) * self._opt.lambda_lp

        if self._opt.use_vgg:
            self._loss_g_vgg = torch.mean(self._criterion_vgg(fake_imgs, self._input_real_imgs)) * self._opt.lambda_vgg

        if self._opt.use_face:
            self._loss_g_face = torch.mean(self._criterion_face(fake_tsf_imgs, self._input_desired_img,
                                                                self._j2d, self._j2d)) * self._opt.lambda_face
        # loss mask
        self._loss_g_mask = torch.mean((fake_masks - self._bg_mask) ** 2) * self._opt.lambda_mask
        # self._loss_g_mask = torch.mean(torch.abs(fake_masks - self._bg_mask)) * self._opt.lambda_mask
        self._loss_g_mask_smooth = self._compute_loss_smooth(fake_masks) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_l1 + self._loss_g_vgg + self._loss_g_face + \
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
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_l1', self._loss_g_l1.item()),
                                 ('g_vgg', self._loss_g_vgg.item()),
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

    def transfer_imgs(self, fake_bg, fake_imgs, fake_color, fake_masks):
        self._vis_real_img = util.tensor2im(self._input_real_imgs)

        ids = fake_imgs.shape[0] // 2
        self._vis_tsf = util.tensor2im(self._input_tsf.data)
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

    @staticmethod
    def morph(src_bg_mask, ks, mode='erode'):
        device = src_bg_mask.device

        n_ks = ks ** 2
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32).to(device)

        pad_s = ks // 2
        src_bg_mask_pad = F.pad(src_bg_mask, (pad_s, pad_s, pad_s, pad_s), value=1.0)
        # print(src_bg_mask.shape, src_bg_mask_pad.shape)
        out = F.conv2d(src_bg_mask_pad, kernel)
        # print(out.shape)

        if mode == 'erode':
            out = (out == n_ks).float()
        else:
            out = (out >= 1).float()

        return out

    def post_process(self, fake_tsf_imgs):
        fake_tsf_imgs = self._input_tsf * self._front_mask + (1 - self._front_mask) * fake_tsf_imgs
        return fake_tsf_imgs

    def debug(self, visualizer):
        import torch.nn.functional as F
        import utils.util as util

        with torch.no_grad():
            visualizer.vis_named_img('input_G_tsf', self._input_G_tsf[:, 0:3, ...])

        src_fims = util.plot_fim_enc(fim_enc=self._input_cond[:, 0:3, ...], map_name=self._opt.map_name)
        tsf_fims = util.plot_fim_enc(fim_enc=self._input_cond[:, 3:6, ...], map_name=self._opt.map_name)
        src_mask = 1 - self._input_cond[:, 2:3, ...]

        visualizer.vis_named_img('src_fims', src_fims)
        visualizer.vis_named_img('tsf_fims', tsf_fims)
        visualizer.vis_named_img('src_mask', src_mask)

        visualizer.vis_named_img('src', self._input_src_img)
        visualizer.vis_named_img('tgt', self._input_desired_img)

        for scale in [256]:
            # for scale in [256, 128, 64, 32]:

            # tsf_resized = self._G.transform((self._input_src_img + 1) / 2.0, self._T) * 2 - 1.0
            tsf_resized = self._G.transform(self._input_src_img, self._T)
            tsf_mask = self._G.stn(src_mask, self._T)
            tsf_mask_dilate = self.morph(tsf_mask, ks=5, mode='dilate')

            visualizer.vis_named_img('tsf_%d' % scale, tsf_resized)
            visualizer.vis_named_img('tsf_mask_%d' % scale, tsf_mask)
            visualizer.vis_named_img('tsf_mask_dilate%d' % scale, tsf_mask_dilate)

            print(scale)

        ipdb.set_trace()

    def debug_wrap(self, visualizer):
        with torch.no_grad():
            fake_tsf_imgs, fake_imgs, _ = self.forward()
            fake_tsf_paste = self.post_process(fake_tsf_imgs)

            masks = self._front_mask.cpu().numpy()
            tsf_imgs = self._input_tsf.cpu().numpy()
            fake_tsf_imgs = fake_tsf_imgs.cpu().numpy()
            fake_tsf_paste = fake_tsf_paste.cpu().numpy()

            bs = fake_tsf_imgs.shape[0]
            fake_tsf_seamless = []
            for i in range(bs):
                src = np.transpose(tsf_imgs[i], axes=(1, 2, 0))
                dst = np.transpose(fake_tsf_imgs[i], axes=(1, 2, 0))
                # dst = np.transpose(fake_tsf_paste[i], axes=(1, 2, 0))
                mask = masks[i, 0][:, :, np.newaxis]
                if np.sum(mask) > 50:
                    out = cv_utils.seamless_paste(src, dst, mask)
                else:
                    out = (dst + 1.0) / 2.0
                out = np.transpose(out, axes=(2, 0, 1))
                fake_tsf_seamless.append(out)
            fake_tsf_seamless = np.stack(fake_tsf_seamless, axis=0)
            visualizer.vis_named_img('src', self._input_src_img, normalize=False)
            visualizer.vis_named_img('ref', self._input_desired_img, normalize=False)
            visualizer.vis_named_img('tsf', tsf_imgs, normalize=False)
            visualizer.vis_named_img('mask', masks, normalize=True)
            visualizer.vis_named_img('fake', fake_tsf_imgs, normalize=False)
            visualizer.vis_named_img('out', fake_tsf_seamless, normalize=True)
            visualizer.vis_named_img('out_paste', fake_tsf_paste, normalize=False)
            visualizer.vis_named_img('out_copy', self._input_tsf * self._front_mask, normalize=False)

        ipdb.set_trace()

    def debug_get_data(self, visualizer):
        with torch.no_grad():
            fake_tsf_imgs, _, _ = self.forward()
            # fake_tsf_paste = self.post_process(fake_tsf_imgs)

            src_imgs = self._input_src_img.cpu().numpy()
            ref_imgs = self._input_desired_img.cpu().numpy()

            dst_imgs = fake_tsf_imgs.cpu().numpy()
            obj_imgs = self._input_tsf.cpu().numpy()
            obj_masks = self._front_mask.cpu().numpy()

            visualizer.vis_named_img('src', src_imgs, normalize=False)
            visualizer.vis_named_img('ref', ref_imgs, normalize=False)
            visualizer.vis_named_img('dst', dst_imgs, normalize=False)
            visualizer.vis_named_img('obj', obj_imgs, normalize=False)
            visualizer.vis_named_img('mask', obj_masks, normalize=False)

            return src_imgs, ref_imgs, dst_imgs, obj_imgs, obj_masks

        #     masks = self._front_mask.cpu().numpy()
        #     tsf_imgs = self._input_tsf.cpu().numpy()
        #     fake_tsf_imgs = fake_tsf_imgs.cpu().numpy()
        #     fake_tsf_paste = fake_tsf_paste.cpu().numpy()
        #
        #     bs = fake_tsf_imgs.shape[0]
        #     fake_tsf_seamless = []
        #     for i in range(bs):
        #         src = np.transpose(tsf_imgs[i], axes=(1, 2, 0))
        #         dst = np.transpose(fake_tsf_imgs[i], axes=(1, 2, 0))
        #         # dst = np.transpose(fake_tsf_paste[i], axes=(1, 2, 0))
        #         mask = masks[i, 0][:, :, np.newaxis]
        #         if np.sum(mask) > 50:
        #             out = cv_utils.seamless_paste(src, dst, mask)
        #         else:
        #             out = (dst + 1.0) / 2.0
        #         out = np.transpose(out, axes=(2, 0, 1))
        #         fake_tsf_seamless.append(out)
        #     fake_tsf_seamless = np.stack(fake_tsf_seamless, axis=0)
        #     visualizer.vis_named_img('src', self._input_src_img, normalize=False)
        #     visualizer.vis_named_img('ref', self._input_desired_img, normalize=False)
        #     visualizer.vis_named_img('tsf', tsf_imgs, normalize=False)
        #     visualizer.vis_named_img('mask', masks, normalize=True)
        #     visualizer.vis_named_img('fake', fake_tsf_imgs, normalize=False)
        #     visualizer.vis_named_img('out', fake_tsf_seamless, normalize=True)
        #     visualizer.vis_named_img('out_paste', fake_tsf_paste, normalize=False)
        #     visualizer.vis_named_img('out_copy', self._input_tsf * self._front_mask, normalize=False)
        #
        # ipdb.set_trace()


class Impersonator(BaseTrainerModel):
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

        self._criterion_hmr = losses.HMRLoss(pretrain_model=self._opt.hmr_model,
                                             smpl_data=util.load_pickle_file(self._opt.smpl_model)).cuda()

        self._render = networks.SMPLRenderer(faces=self._criterion_hmr.hmr.smpl.faces,
                                             map_name=self._opt.map_name,
                                             uv_map_path=self._opt.uv_mapping,
                                             tex_size=self._opt.tex_size,
                                             image_size=self._opt.image_size, fill_back=True,
                                             anti_aliasing=True, background_color=(0, 0, 0), has_front_map=False)

    def _create_generator(self):
        return networks.create_by_name('res_unet',
                                       bg_dim=4, src_dim=3+self._G_cond_nc, tsf_dim=3+self._G_cond_nc, repeat_num=6)

    def _create_discriminator(self):
        return networks.create_by_name('patch', input_nc=3 + self._D_cond_nc,
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
            self._criterion_vgg = losses.VGGLoss().cuda()
            # self._criterion_vgg = VGGLoss().cuda()

        if self._opt.use_face:
            self._criterion_face = losses.SphereFaceLoss().cuda()
            # self._criterion_face = SphereFaceLoss().cuda()

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

    @property
    def hmr(self):
        return self._criterion_hmr.hmr

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
        self._input_G_tsf = torch.cat([self._input_tsf, self._input_desired_cond], dim=1)

        if is_train:
            tsf_crop_mask = self.morph(self._input_desired_cond[:, -1:, :, :], ks=3, mode='erode')
            self._bg_mask = torch.cat([src_crop_mask, tsf_crop_mask], dim=0)
            self._input_D = torch.cat([self._input_desired_img, self._input_src_cond, self._input_desired_cond], dim=1)
            self._input_real_imgs = torch.cat([self._input_src_img, self._input_desired_img], dim=0)

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

        return fake_tsf_imgs, fake_imgs, fake_masks

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
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
            if trainable:
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

    def transfer_imgs(self, fake_bg, fake_imgs, fake_color, fake_masks):
        self._vis_real_img = util.tensor2im(self._input_real_imgs)

        ids = fake_imgs.shape[0] // 2
        self._vis_tsf = util.tensor2im(self._input_tsf.data)
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

    def debug_fim_transfer(self, visualizer):
        import torch.nn.functional as F
        import utils.util as util

        src_fims = util.plot_fim_enc(fim_enc=self._input_src_cond, map_name=self._opt.map_name)
        tsf_fims = util.plot_fim_enc(fim_enc=self._input_desired_cond, map_name=self._opt.map_name)

        visualizer.vis_named_img('src_fims', src_fims)
        visualizer.vis_named_img('tsf_fims', tsf_fims)

        visualizer.vis_named_img('src', self._input_src_img)
        visualizer.vis_named_img('tgt', self._input_desired_img)
        visualizer.vis_named_img('tsf', self._input_tsf)

        for scale in [512, 256]:
            # for scale in [256, 128, 64, 32]:
            src_resized = F.interpolate(self._input_src_img, size=(scale, scale), mode='bilinear', align_corners=True)

            tsf_resized = self._G.transform(src_resized, self._T)

            visualizer.vis_named_img('tsf_%d' % scale, tsf_resized)

            print(scale)

        visualizer.vis_named_img('bg_mask', self._bg_mask)
        visualizer.vis_named_img('src_head', self._input_src_head)
        visualizer.vis_named_img('tsf_head', self._input_tsf_head)

        ipdb.set_trace()
