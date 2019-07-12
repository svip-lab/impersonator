import torch
import torch.nn.functional as F
from collections import OrderedDict


import utils.util as util
import utils.cv_utils as cv_utils
from models.models import BaseRunnerModel, BaseTrainerModel
import networks
import networks.losses as losses

import ipdb


class BodyRecoveryFlow(torch.nn.Module):

    def __init__(self, opt, ks=15):
        super(BodyRecoveryFlow, self).__init__()
        self._name = 'BodyRecoveryFlow'
        self._opt = opt
        self.ks = ks

        # create networks
        self._init_create_networks()

        # buffers
        im_size = self._opt.image_size
        self.register_buffer('initial_T', torch.zeros(im_size, im_size, 2, dtype=torch.float32))

    def _create_hmr(self):
        hmr = networks.HumanModelRecovery(smpl_data=util.load_pickle_file(self._opt.smpl_model))
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def _create_render(self, faces):
        render = networks.SMPLRendererTrainer(faces=faces,
                                              map_name=self._opt.map_name,
                                              uv_map_path=self._opt.uv_mapping,
                                              tex_size=self._opt.tex_size,
                                              image_size=self._opt.image_size, fill_back=True,
                                              anti_aliasing=True, background_color=(0, 0, 0), has_front_map=False)

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

    def forward(self, src_img, src_smpl):
        # source process
        src_info = self.hmr.get_details(src_smpl)
        src_rd, src_info['tex'] = self.render.forward(src_info['cam'], src_info['verts'],
                                                      src_img, is_uv_sampler=False,
                                                      reverse_yz=True, get_fim=False)

        src_cond, src_info['fim'] = self.render.encode_fim(src_info['cam'], src_info['verts'], transpose=True)
        # src_bg_mask = self.morph(src_cond[:, -1:, :, :], ks=self.ks, mode='erode')
        src_bg_mask = self.morph(1 - src_cond[:, -1:, :, :], ks=self.ks, mode='erode')

        # bg input
        mask = 1 - src_bg_mask

        return src_img * src_bg_mask, mask

    @staticmethod
    def morph(src_bg_mask, ks, mode='erode'):
        device = src_bg_mask.device

        n_ks = ks ** 2
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32).to(device)

        pad_s = ks // 2
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=1.0)
        # print(src_bg_mask.shape, src_bg_mask_pad.shape)
        out = F.conv2d(src_bg_mask_pad, kernel)
        # print(out.shape)

        if mode == 'erode':
            out = (out == n_ks).float()
        else:
            out = (out >= 1).float()

        return out


class InpaintorTrainer(BaseTrainerModel):
    def __init__(self, opt):
        super(InpaintorTrainer, self).__init__(opt)
        self._name = 'InpaintorTrainer'

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
        self._G = networks.create_by_name('isag', c_dim=4)
        # self._G.init_weights()
        if multi_gpus:
            self._G = torch.nn.DataParallel(self._G)
        self._G.cuda()

        # discriminator network
        self._D = networks.create_by_name('isad', c_dim=4)
        # self._D.init_weights()
        if multi_gpus:
            self._D = torch.nn.DataParallel(self._D)
        self._D.cuda()

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=[self._opt.D_adam_b1, self._opt.D_adam_b2])

    def _init_prefetch_inputs(self):
        self._images = None
        self._masks = None
        self._incomp_imgs = None

    def _init_losses(self):
        # define loss functions
        multi_gpus = len(self._gpu_ids) > 1

        # criterion of reconstruction
        self._ctr_recons = losses.ReconLoss(*self._opt.L1_LOSS_ALPHA)

        # criterion of gan
        self._ctr_gan = losses.SNGenLoss(self._opt.GAN_LOSS_ALPHA)

        # criterion of dis
        self._ctr_dis = losses.SNDisLoss()

        # if multi_gpus:
        #     self._ctr_recons = torch.nn.DataParallel(self._ctr_recons)
        #     self._ctr_gan = torch.nn.DataParallel(self._ctr_gan)
        #     self._ctr_dis = torch.nn.DataParallel(self._ctr_dis)

        # init losses G
        self._loss_g_recons = self._Tensor([0])
        self._loss_g_adv = self._Tensor([0])

        # init losses D
        self._loss_d = self._Tensor([0])

    @property
    def bdr(self):
        return self._bdr

    def set_input(self, input):

        with torch.no_grad():
            images = input['images'].cuda()
            smpls = input['smpls'].cuda()
            self._incomp_imgs, self._masks = self._bdr(images, smpls)
            self._images = images

    def set_G_train(self):
        self._G.train()

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        # generate fake images
        coarse_imgs, refine_imgs, comp_imgs = self._G.forward(imgs=self._images, masks=self._masks)

        # keep data for visualization
        if keep_data_for_visuals:
            self.visual_setup(coarse_imgs, refine_imgs, comp_imgs)

        return coarse_imgs, refine_imgs, comp_imgs

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
        if self._is_train:
            # convert tensor to variables
            self._B = min(self._images.size(0), 8)

            # run
            coarse_imgs, refine_imgs, comp_imgs = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            loss_G = self._optimize_G(coarse_imgs, refine_imgs, comp_imgs)

            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

            # train D
            if trainable:
                loss_D = self._optimize_D(comp_imgs)
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

    def _optimize_G(self, coarse_imgs, refine_imgs, comp_imgs):
        # adversarial loss
        fake_inputs = torch.cat([comp_imgs, self._masks], dim=1)
        d_fake_outs = self._D.forward(fake_inputs)
        self._loss_g_adv = self._ctr_gan(d_fake_outs)

        # reconstruction loss
        self._loss_g_recons = self._ctr_recons(
            imgs=self._images, coarse_imgs=coarse_imgs, recon_imgs=refine_imgs, masks=self._masks)

        # combine losses
        return self._loss_g_adv + self._loss_g_recons

    def _optimize_D(self, comp_imgs):
        real_inputs = torch.cat([self._images, self._masks], dim=1)
        fake_inputs = torch.cat([comp_imgs.detach(), self._masks], dim=1)

        real_outs = self._D.forward(real_inputs)
        fake_outs = self._D.forward(fake_inputs)

        self._loss_d = self._ctr_dis(real_outs, fake_outs)
        return self._loss_d

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_recons', self._loss_g_recons.item()),
                                 ('g_adv', self._loss_g_adv.item()),
                                 ('d_loss', self._loss_d.item())])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # inputs
        visuals['1_real_image'] = self._vis_batch_real_img
        visuals['2_coarse_image'] = self._vis_batch_coarse_img
        visuals['3_refine_image'] = self._vis_batch_refine_img
        visuals['4_comp_image'] = self._vis_batch_comp_img

        return visuals

    def visual_setup(self, coarse_imgs, refine_imgs, comp_imgs):
        self._vis_batch_real_img = util.tensor2im(self._images[0: self._B], idx=-1)
        self._vis_batch_coarse_img = util.tensor2im(coarse_imgs[0: self._B].data, idx=-1)
        self._vis_batch_refine_img = util.tensor2im(refine_imgs[0: self._B].data, idx=-1)
        self._vis_batch_comp_img = util.tensor2im(comp_imgs[0: self._B].data, idx=-1)

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
        with torch.no_grad():
            input_G_src = self._input_G_bg[:, 0:3, ...]
            input_G_mask = self._input_G_bg[:, 3:, ...]

            visualizer.vis_named_img('input_G_src', input_G_src)
            visualizer.vis_named_img('input_G_mask', input_G_mask)

            ipdb.set_trace()


class Inpaintor(BaseRunnerModel):
    def __init__(self, opt):
        super(Inpaintor, self).__init__(opt)
        self._name = 'Inpaintor'

        # create networks
        self._init_create_networks()

        # prefetch variables
        self._init_prefetch_inputs()

    def _init_create_networks(self):
        # body recovery Flow
        self._bdr = BodyRecoveryFlow(opt=self._opt, ks=15)
        self._bdr.eval()
        self._bdr.cuda()

        # generator network
        self._G = networks.create_by_name('isag', c_dim=4)
        self._G.cuda()

        # load parameters
        if self._opt.load_path:
            self._load_params(self._G, self._opt.load_path)
        elif self._opt.load_epoch > 0:
            self._load_network(self._G, 'G', self._opt.load_epoch)
        else:
            raise ValueError('load_path {} is empty and load_epoch {} is 0'.format(
                self._opt.load_path, self._opt.load_epoch))

        # set eval
        self.set_eval()

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def _init_prefetch_inputs(self):
        self._images = None
        self._masks = None
        self._incomp_imgs = None

    def set_input(self, input):

        with torch.no_grad():
            images = input['images'].cuda()
            smpls = input['smpls'].cuda()
            self._incomp_imgs, self._masks = self._bdr(images, smpls)
            self._images = images

    def forward(self, keep_data_for_visuals=False):
        # generate fake images
        coarse_imgs, refine_imgs, comp_imgs = self._G.forward(imgs=self._images, masks=self._masks)

        # keep data for visualization
        if keep_data_for_visuals:
            self.visual_setup(coarse_imgs, refine_imgs, comp_imgs)

        return coarse_imgs, refine_imgs, comp_imgs

    def inference(self, image_path, smpl=None, visualizer=None):
        with torch.no_grad():
            ori_img = cv_utils.read_cv2_img(image_path)
            # resize image and convert the color space from [0, 255] to [-1, 1]
            img = cv_utils.transform_img(ori_img, self._opt.image_size, transpose=True) * 2 - 1.0
            img = torch.FloatTensor(img).cuda()[None, ...]

            if smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.FloatTensor(img_hmr).cuda()[None, ...]
                smpl = self._bdr.hmr(img_hmr)[-1]
            else:
                smpl = util.to_tensor(smpl).cuda()[None, ...]

            print(img.shape)
            inputs = {'images': img, 'smpls': smpl}
            self.set_input(inputs)

            coarse_imgs, refine_imgs, comp_imgs = self.forward(keep_data_for_visuals=visualizer is not None)

            if visualizer:
                visuals = self.get_current_visuals()
                self.visualize(visualizer, visuals)

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # inputs
        visuals['1_real_image'] = self._vis_batch_real_img
        visuals['2_coarse_image'] = self._vis_batch_coarse_img
        visuals['3_refine_image'] = self._vis_batch_refine_img
        visuals['4_comp_image'] = self._vis_batch_comp_img

        return visuals

    def visual_setup(self, coarse_imgs, refine_imgs, comp_imgs):
        num_vis = min(4, coarse_imgs.shape[0])
        self._vis_batch_real_img = util.tensor2im(self._images[0: num_vis], idx=-1)
        self._vis_batch_coarse_img = util.tensor2im(coarse_imgs[0: num_vis].data, idx=-1)
        self._vis_batch_refine_img = util.tensor2im(refine_imgs[0: num_vis].data, idx=-1)
        self._vis_batch_comp_img = util.tensor2im(comp_imgs[0: num_vis].data, idx=-1)

    def visualize(self, visualizer, visuals_data):
        visualizer.vis_named_img('1_real_image', visuals_data['1_real_image'][None], normalize=True)
        visualizer.vis_named_img('2_coarse_image', visuals_data['2_coarse_image'][None], normalize=True)
        visualizer.vis_named_img('3_refine_image', visuals_data['3_refine_image'][None], normalize=True)
        visualizer.vis_named_img('4_comp_image', visuals_data['4_comp_image'][None], normalize=True)
        visualizer.vis_named_img('5_masks', self._masks, normalize=True)

        ipdb.set_trace()
