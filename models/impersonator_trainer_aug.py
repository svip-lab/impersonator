import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
import utils.util as util
from .models import BaseModel
from networks.networks import NetworksFactory, HumanModelRecovery, Vgg19, VGGLoss, FaceLoss, StyleLoss
# from utils.nmr import SMPLRenderer
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

    def forward(self, aug_img, src_img, src_smpl, ref_smpl):
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
        input_G_aug_bg = torch.cat([aug_img * src_bg_mask, src_bg_mask], dim=1)
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

        head_bbox = self.cal_head_bbox(ref_info['j2d'])
        body_bbox = self.cal_body_bbox(ref_info['j2d'])

        return input_G_aug_bg, input_G_bg, input_G_src, input_G_tsf, T, bg_mask, head_bbox, body_bbox

    def cal_head_bbox(self, kps):
        """
        Args:
            kps: (N, 19, 2)

        Returns:
            bbox: (N, 4)
        """
        NECK_IDS = 12

        image_size = self._opt.image_size

        kps = (kps + 1) / 2.0

        necks = kps[:, NECK_IDS, 0]
        zeros = torch.zeros_like(necks)
        ones = torch.ones_like(necks)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_x, _ = torch.min(kps[:, NECK_IDS:, 0] - 0.05, dim=1)
        min_x = torch.max(min_x, zeros)

        max_x, _ = torch.max(kps[:, NECK_IDS:, 0] + 0.05, dim=1)
        max_x = torch.min(max_x, ones)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_y, _ = torch.min(kps[:, NECK_IDS:, 1] - 0.05, dim=1)
        min_y = torch.max(min_y, zeros)

        max_y, _ = torch.max(kps[:, NECK_IDS:, 1], dim=1)
        max_y = torch.min(max_y, ones)

        min_x = (min_x * image_size).long()  # (T, 1)
        max_x = (max_x * image_size).long()  # (T, 1)
        min_y = (min_y * image_size).long()  # (T, 1)
        max_y = (max_y * image_size).long()  # (T, 1)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        rects = torch.stack((min_x, max_x, min_y, max_y), dim=1)
        # import ipdb
        # ipdb.set_trace()
        return rects

    def cal_body_bbox(self, kps, factor=1.2):
        """
        Args:
            kps (torch.cuda.FloatTensor): (N, 19, 2)
            factor (float):

        Returns:
            bbox: (N, 4)
        """
        image_size = self._opt.image_size
        bs = kps.shape[0]
        kps = (kps + 1) / 2.0
        zeros = torch.zeros((bs,), device=kps.device)
        ones = torch.ones((bs,), device=kps.device)

        min_x, _ = torch.min(kps[:, :, 0], dim=1)
        max_x, _ = torch.max(kps[:, :, 0], dim=1)
        middle_x = (min_x + max_x) / 2
        width = (max_x - min_x) * factor
        min_x = torch.max(zeros, middle_x - width / 2)
        max_x = torch.min(ones, middle_x + width / 2)

        min_y, _ = torch.min(kps[:, :, 1], dim=1)
        max_y, _ = torch.max(kps[:, :, 1], dim=1)
        middle_y = (min_y + max_y) / 2
        height = (max_y - min_y) * factor
        min_y = torch.max(zeros, middle_y - height / 2)
        max_y = torch.min(ones, middle_y + height / 2)

        min_x = (min_x * image_size).long()  # (T,)
        max_x = (max_x * image_size).long()  # (T,)
        min_y = (min_y * image_size).long()  # (T,)
        max_y = (max_y * image_size).long()  # (T,)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        bboxs = torch.stack((min_x, max_x, min_y, max_y), dim=1)

        return bboxs


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
        return NetworksFactory.get_by_name(self._opt.gen_name, bg_dim=4, src_dim=3 + self._G_cond_nc,
                                           tsf_dim=3+self._G_cond_nc, repeat_num=self._opt.repeat_num)

    def _create_discriminator(self):
        return NetworksFactory.get_by_name('global_local', input_nc=3 + self._D_cond_nc // 2,
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

        self._real_bg = None
        self._real_src = None
        self._real_tsf = None

        self._bg_mask = None
        self._input_G_aug_bg = None
        self._input_G_src = None
        self._input_G_tsf = None
        self._head_bbox = None
        self._body_bbox = None

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
            self._crt_vgg = VGGLoss(vgg=vgg_net)
            if multi_gpus:
                self._crt_vgg = torch.nn.DataParallel(self._crt_vgg)
            self._crt_vgg.cuda()

        if self._opt.use_style:
            self._crt_sty = StyleLoss(feat_extractors=vgg_net)
            if multi_gpus:
                self._crt_sty = torch.nn.DataParallel(self._crt_sty)
            self._crt_sty.cuda()

        if self._opt.use_face:
            self._crt_face = FaceLoss(pretrained_path=self._opt.face_model)
            if multi_gpus:
                self._criterion_face = torch.nn.DataParallel(self._crt_face)
            self._crt_face.cuda()

        # init losses G
        self._g_l1 = self._Tensor([0])
        self._g_vgg = self._Tensor([0])
        self._g_style = self._Tensor([0])
        self._g_face = self._Tensor([0])
        self._g_adv = self._Tensor([0])
        self._g_smooth = self._Tensor([0])
        self._g_mask = self._Tensor([0])
        self._g_mask_smooth = self._Tensor([0])

        # init losses D
        self._d_real = self._Tensor([0])
        self._d_fake = self._Tensor([0])

    @torch.no_grad()
    def set_input(self, input):

        images = input['images']
        smpls = input['smpls']
        aug_bg = input['bg'].cuda()
        src_img = images[:, 0, ...].contiguous().cuda()
        src_smpl = smpls[:, 0, ...].contiguous().cuda()
        tsf_img = images[:, 1, ...].contiguous().cuda()
        tsf_smpl = smpls[:, 1, ...].contiguous().cuda()

        input_G_aug_bg, input_G_bg, input_G_src, input_G_tsf, T, bg_mask, head_bbox, body_bbox = \
            self._bdr(aug_bg, src_img, src_smpl, tsf_smpl)

        self._input_G_aug_bg = torch.cat([input_G_bg, input_G_aug_bg], dim=0)
        self._input_G_src = input_G_src
        self._input_G_tsf = input_G_tsf
        self._bg_mask = bg_mask
        self._T = T
        self._head_bbox = head_bbox
        self._body_bbox = body_bbox
        self._real_src = src_img
        self._real_tsf = tsf_img
        self._real_bg = aug_bg

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        # generate fake images
        fake_aug_bg, fake_src_color, fake_src_mask, fake_tsf_color, fake_tsf_mask = \
            self._G.forward(self._input_G_aug_bg, self._input_G_src, self._input_G_tsf, T=self._T)

        bs = fake_src_color.shape[0]
        fake_bg = fake_aug_bg[0:bs]
        fake_src_imgs = fake_src_mask * fake_bg + (1 - fake_src_mask) * fake_src_color
        fake_tsf_imgs = fake_tsf_mask * fake_bg + (1 - fake_tsf_mask) * fake_tsf_color

        fake_masks = torch.cat([fake_src_mask, fake_tsf_mask], dim=0)

        # keep data for visualization
        if keep_data_for_visuals:
            self.visual_imgs(fake_bg, fake_aug_bg, fake_src_imgs, fake_tsf_imgs, fake_masks)
            # self.visualizer.vis_named_img('fake_aug_bg', fake_aug_bg)
            # self.visualizer.vis_named_img('fake_aug_bg_input', self._input_G_aug_bg[:, 0:3])
            # self.visualizer.vis_named_img('real_bg', self._real_bg)

        return fake_aug_bg[bs:], fake_src_imgs, fake_tsf_imgs, fake_masks

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
        if self._is_train:
            # convert tensor to variables
            fake_aug_bg, fake_src_imgs, fake_tsf_imgs, fake_masks = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            loss_G = self._optimize_G(fake_aug_bg, fake_src_imgs, fake_tsf_imgs, fake_masks)

            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

            # train D
            if trainable:
                loss_D = self._optimize_D(fake_aug_bg, fake_tsf_imgs)
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

    def _optimize_G(self, fake_aug_bg, fake_src_imgs, fake_tsf_imgs, fake_masks):
        bs = fake_tsf_imgs.shape[0]

        fake_global = torch.cat([fake_aug_bg, self._input_G_aug_bg[bs:, -1:]], dim=1)
        fake_local = torch.cat([fake_tsf_imgs, self._input_G_tsf[:, 3:]], dim=1)
        d_fake_outs = self._D.forward(fake_global, fake_local, self._body_bbox)
        self._g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        self._g_l1 = self._crt_l1(fake_src_imgs, self._real_src) * self._opt.lambda_lp

        if self._opt.use_vgg:
            self._g_vgg = torch.mean(self._crt_vgg(fake_tsf_imgs, self._real_tsf)
                                     + self._crt_vgg(fake_aug_bg, self._real_bg)) * self._opt.lambda_vgg

        if self._opt.use_style:
            self._g_style = torch.mean(self._crt_sty(fake_tsf_imgs, self._real_tsf)
                                       + self._crt_sty(fake_aug_bg, self._real_bg)) * self._opt.lambda_style

        if self._opt.use_face:
            self._g_face = torch.mean(self._crt_face(fake_tsf_imgs, self._real_tsf, bbox1=self._head_bbox,
                                                     bbox2=self._head_bbox)) * self._opt.lambda_face
        # loss mask
        self._g_mask = self._crt_mask(fake_masks, self._bg_mask) * self._opt.lambda_mask

        if self._opt.lambda_mask_smooth != 0:
            self._g_mask_smooth = self._compute_loss_smooth(fake_masks) * self._opt.lambda_mask_smooth

        # combine losses
        return self._g_adv + self._g_l1 + self._g_vgg + self._g_style + self._g_face + self._g_mask + self._g_mask_smooth

    def _optimize_D(self, fake_aug_bg, fake_tsf_imgs):
        bs = fake_tsf_imgs.shape[0]
        fake_global = torch.cat([fake_aug_bg.detach(), self._input_G_aug_bg[bs:, -1:]], dim=1)
        fake_local = torch.cat([fake_tsf_imgs.detach(), self._input_G_tsf[:, 3:]], dim=1)
        real_global = torch.cat([self._real_bg, self._input_G_aug_bg[bs:, -1:]], dim=1)
        real_local = torch.cat([self._real_tsf, self._input_G_tsf[:, 3:]], dim=1)

        d_real_outs = self._D.forward(real_global, real_local, self._body_bbox)
        d_fake_outs = self._D.forward(fake_global, fake_local, self._body_bbox)

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
        loss_dict = OrderedDict([('g_l1', self._g_l1.item()),
                                 ('g_vgg', self._g_vgg.item()),
                                 ('g_face', self._g_face.item()),
                                 ('g_adv', self._g_adv.item()),
                                 ('g_mask', self._g_mask.item()),
                                 ('g_mask_smooth', self._g_mask_smooth.item()),
                                 ('d_real', self._d_real.item()),
                                 ('d_fake', self._d_fake.item())])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # inputs
        visuals['1_real_img'] = self._vis_input
        visuals['2_input_tsf'] = self._vis_tsf
        visuals['3_fake_bg'] = self._vis_fake_bg

        # outputs
        visuals['4_fake_tsf'] = self._vis_fake_tsf
        visuals['5_fake_src'] = self._vis_fake_src
        visuals['6_fake_mask'] = self._vis_mask

        # batch outputs
        visuals['7_batch_real_img'] = self._vis_batch_real
        visuals['8_batch_fake_img'] = self._vis_batch_fake

        return visuals

    @torch.no_grad()
    def visual_imgs(self, fake_bg, fake_aug_bg, fake_src_imgs, fake_tsf_imgs, fake_masks):
        ids = fake_masks.shape[0] // 2
        self._vis_input = util.tensor2im(self._real_src)
        self._vis_tsf = util.tensor2im(self._input_G_tsf[0, 0:3])
        self._vis_fake_bg = util.tensor2im(fake_bg)
        self._vis_fake_src = util.tensor2im(fake_src_imgs)
        self._vis_fake_tsf = util.tensor2im(fake_tsf_imgs)
        self._vis_mask = util.tensor2maskim(fake_masks[ids])

        self._vis_batch_real = util.tensor2im(torch.cat([self._real_tsf, self._real_bg], dim=0), idx=-1)
        self._vis_batch_fake = util.tensor2im(torch.cat([fake_tsf_imgs, fake_aug_bg], dim=0), idx=-1)

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
        self._load_network(self._G, 'G', load_epoch, need_module=True)

        if self._is_train:
            # load D
            self._load_network(self._D, 'D', load_epoch, need_module=True)

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
        visualizer.vis_named_img('bg_inputs', self._input_G_aug_bg[:, 0:3])
        ipdb.set_trace()


class ImpersonatorAllSetTrain(Impersonator):
    def __init__(self, opt):
        super(ImpersonatorAllSetTrain, self).__init__(opt)
        self._name = 'ImpersonatorAllSetTrain'

    @torch.no_grad()
    def set_input(self, input):

        images = input['images']
        smpls = input['smpls']
        aug_bg = input['bg'].cuda()
        src_img = images[:, 0, ...].contiguous().cuda()
        src_smpl = smpls[:, 0, ...].contiguous().cuda()
        tsf_img = images[:, 1, ...].contiguous().cuda()
        tsf_smpl = smpls[:, 1, ...].contiguous().cuda()

        input_G_aug_bg, input_G_bg, input_G_src, input_G_tsf, T, bg_mask, head_bbox, body_bbox = \
            self._bdr(aug_bg, src_img, src_smpl, tsf_smpl)
        bs = input_G_bg.shape[0]

        fashion_images = input['fashion_images'].cuda()
        fashion_G_bg = input['fashion_bg_inputs'].cuda()
        fashion_G_src = input['fashion_src_inputs'].cuda()
        fashion_G_tsf = input['fashion_tsf_inputs'].cuda()
        fashion_T = input['fashion_T'].cuda()
        fashion_head_bbox = input['fashion_head_bbox'].cuda()
        fashion_body_bbox = input['fashion_body_bbox'].cuda()
        fashion_bg_mask = input['fashion_masks'].cuda()

        self._input_G_aug_bg = torch.cat([input_G_bg, fashion_G_bg, input_G_aug_bg], dim=0)
        self._input_G_src = torch.cat([input_G_src, fashion_G_src], dim=0)
        self._input_G_tsf = torch.cat([input_G_tsf, fashion_G_tsf], dim=0)
        self._bg_mask = torch.cat([bg_mask[0:bs], fashion_bg_mask[:, 0],
                                   bg_mask[bs:], fashion_bg_mask[:, 1]], dim=0)
        self._T = torch.cat([T, fashion_T], dim=0)

        self._head_bbox = torch.cat([head_bbox, fashion_head_bbox], dim=0)
        self._body_bbox = torch.cat([body_bbox, fashion_body_bbox], dim=0)
        self._real_src = torch.cat([src_img, fashion_images[:, 0]], dim=0)
        self._real_tsf = torch.cat([tsf_img, fashion_images[:, 1]], dim=0)
        self._real_bg = aug_bg

    def debug(self, visualizer):
        T = self._T
        tsf_imgs = F.grid_sample(self._real_src, T)

        visualizer.vis_named_img('bg_inputs', self._input_G_aug_bg[:, 0:3])
        visualizer.vis_named_img('src_inputs', self._input_G_src[:, 0:3])
        visualizer.vis_named_img('tsf_inputs', self._input_G_tsf[:, 0:3])
        visualizer.vis_named_img('tsf_imgs', tsf_imgs)
        visualizer.vis_named_img('real_src', self._real_src)
        visualizer.vis_named_img('real_tsf', self._real_tsf)
        visualizer.vis_named_img('real_bg', self._real_bg)
        visualizer.vis_named_img('bg_masks', self._bg_mask)

        print(self._head_bbox)
        print()
        print(self._body_bbox)
        ipdb.set_trace()





