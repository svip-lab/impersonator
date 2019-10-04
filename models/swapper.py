import torch
import torch.nn.functional as F
from tqdm import tqdm
from .models import BaseModel
from networks.networks import NetworksFactory, HumanModelRecovery
from utils.detectors import PersonMaskRCNNDetector
from utils.nmr import SMPLRenderer
import utils.cv_utils as cv_utils
import utils.util as util
import utils.mesh as mesh

import ipdb


class Swapper(BaseModel):

    PART_IDS = {
        'body': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    def __init__(self, opt):
        super(Swapper, self).__init__(opt)
        self._name = 'Swapper'

        self._create_networks()

        # prefetch variables
        self.src_info = None
        self.tsf_info = None
        self.T = None
        self.T12 = None
        self.T21 = None
        self.grid = self.render.create_meshgrid(self._opt.image_size).cuda()
        self.part_fn = torch.tensor(mesh.create_mapping('par', self._opt.uv_mapping,
                                                        contain_bg=True, fill_back=False)).float().cuda()
        self.part_faces_dict = mesh.get_part_face_ids(part_type='par', fill_back=False)
        self.part_faces = list(self.part_faces_dict.values())

    def _create_networks(self):
        # 0. create generator
        self.generator = self._create_generator().cuda()

        # 0. create bgnet
        if self._opt.bg_model != 'ORIGINAL':
            self.bgnet = self._create_bgnet().cuda()
        else:
            self.bgnet = self.generator.bg_model

        # 2. create hmr
        self.hmr = self._create_hmr().cuda()

        # 3. create render
        self.render = SMPLRenderer(image_size=self._opt.image_size, tex_size=self._opt.tex_size,
                                   has_front=self._opt.front_warp, fill_back=False).cuda()
        # 4. pre-processor
        if self._opt.has_detector:
            self.detector = PersonMaskRCNNDetector(ks=self._opt.bg_ks, threshold=0.5, to_gpu=True)
        else:
            self.detector = None

    def _create_bgnet(self):
        net = NetworksFactory.get_by_name('deepfillv2', c_dim=4)
        self._load_params(net, self._opt.bg_model, need_module=False)
        net.eval()
        return net

    def _create_generator(self):
        net = NetworksFactory.get_by_name(self._opt.gen_name, bg_dim=4, src_dim=3+self._G_cond_nc,
                                          tsf_dim=3+self._G_cond_nc, repeat_num=self._opt.repeat_num)

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
        hmr = HumanModelRecovery(self._opt.smpl_model)
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    @staticmethod
    def visualize(*args, **kwargs):
        visualizer = args[0]
        if visualizer is not None:
            for key, value in kwargs.items():
                visualizer.vis_named_img(key, value)

    # TODO it dose not support mini-batch inputs currently.
    @torch.no_grad()
    def personalize(self, src_path, src_smpl=None, output_path='', visualizer=None):

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
        src_info['cond'], _ = self.render.encode_fim(src_info['cam'], src_info['verts'], fim=src_fim, transpose=True)
        src_info['f2verts'] = src_f2verts
        src_info['p2verts'] = src_f2verts[:, :, :, 0:2]
        src_info['p2verts'][:, :, :, 1] *= -1

        if self._opt.only_vis:
            src_info['p2verts'] = self.render.get_vis_f2pts(src_info['p2verts'], src_fim)

        src_info['part'], _ = self.render.encode_fim(src_info['cam'], src_info['verts'],
                                                     fim=src_fim, transpose=True, map_fn=self.part_fn)
        # add image to source info
        src_info['img'] = img
        src_info['image'] = ori_img

        # 2. process the src inputs
        if self.detector is not None:
            bbox, body_mask = self.detector.inference(img[0])
            bg_mask = 1 - body_mask
        else:
            bg_mask = util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.bg_ks, mode='erode')
            body_mask = 1 - bg_mask

        if self._opt.bg_model != 'ORIGINAL':
            src_info['bg'] = self.bgnet(img, masks=body_mask, only_x=True)
        else:
            incomp_img = img * bg_mask
            bg_inputs = torch.cat([incomp_img, bg_mask], dim=1)
            img_bg = self.bgnet(bg_inputs)
            # src_info['bg_inputs'] = bg_inputs
            src_info['bg'] = img_bg
            # src_info['bg'] = incomp_img + img_bg * body_mask

        ft_mask = 1 - util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.ft_ks, mode='erode')
        src_inputs = torch.cat([img * ft_mask, src_info['cond']], dim=1)

        src_info['feats'] = self.generator.encode_src(src_inputs)
        src_info['src_inputs'] = src_inputs

        src_info = src_info

        # if visualizer is not None:
        #     self.visualize(visualizer, src=img, bg=src_info['bg'])

        if output_path:
            cv_utils.save_cv2_img(src_info['image'], output_path, image_size=self._opt.image_size)

        return src_info

    def _extract_smpls(self, input_file):
        img = cv_utils.read_cv2_img(input_file)
        img = cv_utils.transform_img(img, image_size=224) * 2 - 1.0  # hmr receive [-1, 1]
        img = img.transpose((2, 0, 1))
        img = torch.FloatTensor(img).cuda()[None, ...]
        theta = self.hmr(img)[-1]

        return theta

    @torch.no_grad()
    def swap_smpl(self, src_cam, src_shape, tgt_smpl, preserve_scale=True):
        cam = tgt_smpl[:, 0:3].contiguous()
        pose = tgt_smpl[:, 3:75].contiguous()

        if preserve_scale:
            cam[:, 0] = src_cam[:, 0]
            cam[:, 1:] = (src_cam[:, 0] / cam[:, 0]) * cam[:, 1:] + src_cam[:, 1:]
            cam[:, 0] = src_cam[:, 0]
        else:
            cam[: 0] = src_cam[:, 0]

        tsf_smpl = torch.cat([cam, pose, src_shape], dim=1)

        return tsf_smpl

    @torch.no_grad()
    def swap_setup(self, src_path, tgt_path, src_smpl=None, tgt_smpl=None, output_dir=''):
        self.src_info = self.personalize(src_path, src_smpl)
        self.tsf_info = self.personalize(tgt_path, tgt_smpl)

    @torch.no_grad()
    def swap(self, src_info, tgt_info, target_part='body', visualizer=None):
        assert target_part in self.PART_IDS.keys()

        def merge_list(part_ids):
            faces = set()
            for i in part_ids:
                fs = set(self.part_faces[i])
                faces |= fs
            return list(faces)

        # get target selected face index map
        selected_ids = self.PART_IDS[target_part]
        left_ids = [i for i in self.PART_IDS['all'] if i not in selected_ids]

        src_part_mask = (torch.sum(src_info['part'][:, selected_ids, ...], dim=1) != 0).bool()
        src_left_mask = torch.sum(src_info['part'][:, left_ids, ...], dim=1).bool()

        # selected_faces = merge_list(selected_ids)
        left_faces = merge_list(left_ids)

        T11, T21 = self.calculate_trans(src_left_mask, left_faces)

        tsf21 = self.generator.transform(tgt_info['img'], T21)
        tsf11 = self.generator.transform(src_info['img'], T11)

        src_part_mask = src_part_mask[:, None, :, :].float()
        src_left_mask = src_left_mask[:, None, :, :].float()
        tsf_img = tsf21 * src_part_mask + tsf11 * src_left_mask

        tsf_inputs = torch.cat([tsf_img, src_info['cond']], dim=1)

        preds, tsf_mask = self.forward(tsf_inputs, tgt_info['feats'], T21, src_info['feats'], T11, src_info['bg'])

        if self._opt.front_warp:
            # preds = tsf11 * src_left_mask + (1 - src_left_mask) * preds
            preds = self.warp(preds, src_info['img'], src_info['fim'], tsf_mask)

        if visualizer is not None:
            self.visualize(visualizer, src_img=src_info['img'], tgt_img=tgt_info['img'], preds=preds)

        return preds

    # TODO it dose not support mini-batch inputs currently.
    def calculate_trans(self, src_left_mask, left_faces):
        # calculate T11
        T11 = self.grid.clone()
        T11[~src_left_mask[0]] = -2
        T11.unsqueeze_(0)

        # calculate T21
        tsf_f2p = self.tsf_info['p2verts'].clone()
        tsf_f2p[0, left_faces] = -2
        T21 = self.render.cal_bc_transform(tsf_f2p, self.src_info['fim'], self.src_info['wim'])
        T21.clamp_(-2, 2)
        return T11, T21

    def warp(self, preds, tsf, fim, fake_tsf_mask):
        front_mask = self.render.encode_front_fim(fim, transpose=True)
        preds = (1 - front_mask) * preds + tsf * front_mask * (1 - fake_tsf_mask)
        # preds = torch.clamp(preds + tsf * front_mask, -1, 1)
        return preds

    def forward(self, tsf_inputs, feats21, T21, feats11, T11, bg):
        with torch.no_grad():
            # generate fake images
            src_encoder_outs21, src_resnet_outs21 = feats21
            src_encoder_outs11, src_resnet_outs11 = feats11

            tsf_color, tsf_mask = self.generator.swap(tsf_inputs, src_encoder_outs21, src_encoder_outs11,
                                                      src_resnet_outs21, src_resnet_outs11, T21, T11)
            pred_imgs = tsf_mask * bg + (1 - tsf_mask) * tsf_color

        return pred_imgs, tsf_mask

    def post_personalize(self, out_dir, visualizer, verbose=True):
        from networks.networks import FaceLoss
        bs = 2 if self._opt.batch_size > 1 else 1

        init_bg = torch.cat([self.src_info['bg'], self.tsf_info['bg']], dim=0)

        @torch.no_grad()
        def initialize(src_info, tsf_info):
            src_encoder_outs, src_resnet_outs = src_info['feats']
            src_f2p = src_info['p2verts']

            tsf_fim = tsf_info['fim']
            tsf_wim = tsf_info['wim']
            tsf_cond = tsf_info['cond']

            T = self.render.cal_bc_transform(src_f2p, tsf_fim, tsf_wim)
            tsf_img = F.grid_sample(src_info['img'], T)
            tsf_inputs = torch.cat([tsf_img, tsf_cond], dim=1)

            tsf_color, tsf_mask = self.generator.inference(
                src_encoder_outs, src_resnet_outs, tsf_inputs, T)

            preds = src_info['bg'] * tsf_mask + tsf_color * (1 - tsf_mask)

            if self._opt.front_warp:
                preds = self.warp(preds, tsf_img, tsf_fim, tsf_mask)

            return preds, T, tsf_inputs

        @torch.no_grad()
        def set_inputs(src_info, tsf_info):
            s2t_init_preds, s2t_T, s2t_tsf_inputs = initialize(src_info, tsf_info)
            t2s_init_preds, t2s_T, t2s_tsf_inputs = initialize(tsf_info, src_info)

            s2t_j2d = torch.cat([src_info['j2d'], tsf_info['j2d']], dim=0)
            t2s_j2d = torch.cat([tsf_info['j2d'], src_info['j2d']], dim=0)
            j2ds = torch.stack([s2t_j2d, t2s_j2d], dim=0)

            init_preds = torch.cat([s2t_init_preds, t2s_init_preds], dim=0)
            images = torch.cat([src_info['img'], tsf_info['img']], dim=0)
            T = torch.cat([s2t_T, t2s_T], dim=0)
            T_cycle = torch.cat([t2s_T, s2t_T], dim=0)
            tsf_inputs = torch.cat([s2t_tsf_inputs, t2s_tsf_inputs], dim=0)
            src_fim = torch.cat([src_info['fim'], tsf_info['fim']], dim=0)
            tsf_fim = torch.cat([tsf_info['fim'], src_info['fim']], dim=0)

            s2t_inputs = src_info['src_inputs']
            t2s_inputs = tsf_info['src_inputs']

            src_inputs = torch.cat([s2t_inputs, t2s_inputs], dim=0)

            src_mask = util.morph(src_inputs[:, -1:, ], ks=self._opt.ft_ks, mode='erode')
            tsf_mask = util.morph(tsf_inputs[:, -1:, ], ks=self._opt.ft_ks, mode='erode')

            pseudo_masks = torch.cat([src_mask, tsf_mask], dim=0)

            return src_fim, tsf_fim, j2ds, T, T_cycle, src_inputs, tsf_inputs, images, init_preds, pseudo_masks

        def set_cycle_inputs(fake_tsf_imgs, src_inputs, tsf_inputs, T_cycle):
            # set cycle bg inputs
            tsf_bg_mask = tsf_inputs[:, -1:, ...]

            # set cycle src inputs
            cycle_src_inputs = torch.cat([fake_tsf_imgs * tsf_bg_mask, tsf_inputs[:, 3:]], dim=1)

            # set cycle tsf inputs
            cycle_tsf_img = F.grid_sample(fake_tsf_imgs, T_cycle)
            cycle_tsf_inputs = torch.cat([cycle_tsf_img, src_inputs[:, 3:]], dim=1)

            return cycle_src_inputs, cycle_tsf_inputs

        def inference(bg, src_inputs, tsf_inputs, T, T_cycle, src_fim, tsf_fim):
            fake_src_color, fake_src_mask, fake_tsf_color, fake_tsf_mask = \
                self.generator.infer_front(src_inputs, tsf_inputs, T=T)
            fake_src_imgs = fake_src_mask * bg + (1 - fake_src_mask) * fake_src_color
            fake_tsf_imgs = fake_tsf_mask * bg + (1 - fake_tsf_mask) * fake_tsf_color

            if self._opt.front_warp:
                fake_tsf_imgs = self.warp(fake_tsf_imgs, tsf_inputs[:, 0:3], tsf_fim, fake_tsf_mask)

            cycle_src_inputs, cycle_tsf_inputs = set_cycle_inputs(
                fake_tsf_imgs, src_inputs, tsf_inputs, T_cycle)

            cycle_src_color, cycle_src_mask, cycle_tsf_color, cycle_tsf_mask = \
                self.generator.infer_front(cycle_src_inputs, cycle_tsf_inputs, T=T_cycle)

            cycle_src_imgs = cycle_src_mask * bg + (1 - cycle_src_mask) * cycle_src_color
            cycle_tsf_imgs = cycle_tsf_mask * bg + (1 - cycle_tsf_mask) * cycle_tsf_color

            if self._opt.front_warp:
                cycle_tsf_imgs = self.warp(cycle_tsf_imgs, src_inputs[:, 0:3], src_fim, fake_src_mask)

            return fake_src_imgs, fake_tsf_imgs, cycle_src_imgs, cycle_tsf_imgs, fake_src_mask, fake_tsf_mask, cycle_tsf_inputs

        def create_criterion():
            face_criterion = FaceLoss(pretrained_path=self._opt.face_model).cuda()
            idt_criterion = torch.nn.L1Loss()
            mask_criterion = torch.nn.BCELoss()

            return face_criterion, idt_criterion, mask_criterion

        def print_losses(*args, **kwargs):

            print('step = {}'.format(kwargs['step']))
            for key, value in kwargs.items():
                if key == 'step':
                    continue
                print('\t{}, {:.6f}'.format(key, value.item()))

        def update_learning_rate(optimizer, current_lr, init_lr, final_lr, nepochs_decay):
            # updated learning rate G
            lr_decay = (init_lr - final_lr) / nepochs_decay
            current_lr -= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            # print('update G learning rate: %f -> %f' % (current_lr + lr_decay, current_lr))
            return current_lr

        init_lr = 0.0002
        cur_lr = init_lr
        final_lr = 0.00001
        fix_iters = 25
        total_iters = 50
        # fix_iters = int(50 / bs)
        # total_iters = int(100 / bs)
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=init_lr, betas=(0.5, 0.999))
        face_cri, idt_cri, msk_cri = create_criterion()

        # set up inputs
        src_fim, tsf_fim, j2ds, T, T_cycle, src_inputs, tsf_inputs, \
        src_imgs, init_preds, pseudo_masks = set_inputs(
            src_info=self.src_info, tsf_info=self.tsf_info
        )

        logger = tqdm(range(total_iters))
        for step in logger:
            if bs == 1:
                i = step % 2
                _bg = init_bg[i][None]
                _init_preds = init_preds[i][None]
                _src_imgs = src_imgs[i][None]
                _src_inputs = src_inputs[i][None]
                _tsf_inputs = tsf_inputs[i][None]
                _T = T[i][None]
                _T_cycle = T_cycle[i][None]
                _src_fim = src_fim[i][None]
                _tsf_fim = tsf_fim[i][None]

                _pseudo_masks = pseudo_masks[i:4:2]
            else:
                _bg = init_bg
                _src_imgs = src_imgs
                _init_preds = init_preds
                _pseudo_masks = pseudo_masks
                _src_inputs, _tsf_inputs, _T, _T_cycle, _src_fim, _tsf_fim = \
                    src_inputs, tsf_inputs, T, T_cycle, src_fim, tsf_fim
            fake_src_imgs, fake_tsf_imgs, cycle_src_imgs, cycle_tsf_imgs, fake_src_mask, fake_tsf_mask, \
            cycle_tsf_inputs = inference(_bg, _src_inputs, _tsf_inputs, _T, _T_cycle, _src_fim, _tsf_fim)

            # cycle reconstruction loss
            cycle_loss = idt_cri(_src_imgs, fake_src_imgs) + idt_cri(_src_imgs, cycle_tsf_imgs)

            # structure loss
            bg_mask = _src_inputs[:, -1:]
            body_mask = 1.0 - bg_mask
            str_src_imgs = _src_imgs * body_mask
            cycle_warp_imgs = cycle_tsf_inputs[:, 0:3]

            struct_loss = idt_cri(_init_preds, fake_tsf_imgs) + \
                          2 * idt_cri(str_src_imgs, cycle_warp_imgs)

            fid_loss = face_cri(_src_imgs, cycle_tsf_imgs, kps1=j2ds[:, 0], kps2=j2ds[:, 0]) + \
                       face_cri(_tsf_inputs[:, 0:3], fake_tsf_imgs, kps1=j2ds[:, 1], kps2=j2ds[:, 1])

            # mask loss
            # mask_loss = msk_cri(fake_tsf_mask, tsf_inputs[:, -1:]) + msk_cri(fake_src_mask, src_inputs[:, -1:])
            mask_loss = msk_cri(torch.cat([fake_src_mask, fake_tsf_mask], dim=0), _pseudo_masks)

            loss = 10 * cycle_loss + 10 * struct_loss + fid_loss + 5 * mask_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                logger.set_description(
                    (
                        f'step: {step}; '
                        f'total: {loss.item():.6f}; cyc: {cycle_loss.item():.6f}; '
                        f'str: {struct_loss.item():.6f}; fid: {fid_loss.item():.6f}; '
                        f'msk: {mask_loss.item():.6f}'
                    )
                )

            if step % 10 == 0:
                self.visualize(visualizer, input_imgs=src_imgs, tsf_imgs=fake_tsf_imgs,
                               cyc_imgs=cycle_tsf_imgs, fake_tsf_mask=fake_tsf_mask,
                               init_preds=init_preds,
                               str_src_imgs=str_src_imgs,
                               cycle_warp_imgs=cycle_warp_imgs)

            if step > fix_iters:
                cur_lr = update_learning_rate(optimizer, cur_lr, init_lr, final_lr, fix_iters)

        self.generator.eval()

    # def post_personalize_previous(self, out_dir, visualizer, verbose=True):
    #     from networks.networks import FaceLoss
    #     bs = 2 if self._opt.batch_size > 1 else 1
    #
    #     init_bg = torch.cat([self.src_info['bg'], self.tsf_info['bg']], dim=0)
    #
    #     @torch.no_grad()
    #     def initialize(src_info, tsf_info):
    #         src_encoder_outs, src_resnet_outs = src_info['feats']
    #         src_f2p = src_info['p2verts']
    #
    #         tsf_fim = tsf_info['fim']
    #         tsf_wim = tsf_info['wim']
    #         tsf_cond = tsf_info['cond']
    #
    #         T = self.render.cal_bc_transform(src_f2p, tsf_fim, tsf_wim)
    #         tsf_img = F.grid_sample(src_info['img'], T)
    #         tsf_inputs = torch.cat([tsf_img, tsf_cond], dim=1)
    #
    #         tsf_color, tsf_mask = self.generator.inference(
    #             src_encoder_outs, src_resnet_outs, tsf_inputs, T)
    #
    #         preds = src_info['bg'] * tsf_mask + tsf_color * (1 - tsf_mask)
    #
    #         if self._opt.front_warp:
    #             preds = self.warp(preds, tsf_img, tsf_fim, tsf_mask)
    #
    #         return preds, T, tsf_inputs
    #
    #     @torch.no_grad()
    #     def set_inputs(src_info, tsf_info):
    #         s2t_init_preds, s2t_T, s2t_tsf_inputs = initialize(src_info, tsf_info)
    #         t2s_init_preds, t2s_T, t2s_tsf_inputs = initialize(tsf_info, src_info)
    #
    #         s2t_j2d = torch.cat([src_info['j2d'], tsf_info['j2d']], dim=0)
    #         t2s_j2d = torch.cat([tsf_info['j2d'], src_info['j2d']], dim=0)
    #         j2ds = torch.stack([s2t_j2d, t2s_j2d], dim=0)
    #
    #         init_preds = torch.cat([s2t_init_preds, t2s_init_preds], dim=0)
    #         images = torch.cat([src_info['img'], tsf_info['img']], dim=0)
    #         T = torch.cat([s2t_T, t2s_T], dim=0)
    #         T_cycle = torch.cat([t2s_T, s2t_T], dim=0)
    #         tsf_inputs = torch.cat([s2t_tsf_inputs, t2s_tsf_inputs], dim=0)
    #         src_fim = torch.cat([src_info['fim'], tsf_info['fim']], dim=0)
    #         tsf_fim = torch.cat([tsf_info['fim'], src_info['fim']], dim=0)
    #
    #         s2t_inputs = src_info['src_inputs']
    #         t2s_inputs = tsf_info['src_inputs']
    #
    #         src_inputs = torch.cat([s2t_inputs, t2s_inputs], dim=0)
    #
    #         src_mask = util.morph(src_inputs[:, -1:, ], ks=self._opt.ft_ks, mode='erode')
    #         tsf_mask = util.morph(tsf_inputs[:, -1:, ], ks=self._opt.ft_ks, mode='erode')
    #
    #         pseudo_masks = torch.cat([src_mask, tsf_mask], dim=0)
    #
    #         return src_fim, tsf_fim, j2ds, T, T_cycle, src_inputs, tsf_inputs, images, init_preds, pseudo_masks
    #
    #     def set_cycle_inputs(fake_tsf_imgs, src_inputs, tsf_inputs, T_cycle):
    #         # set cycle bg inputs
    #         tsf_bg_mask = tsf_inputs[:, -1:, ...]
    #
    #         # set cycle src inputs
    #         cycle_src_inputs = torch.cat([fake_tsf_imgs * tsf_bg_mask, tsf_inputs[:, 3:]], dim=1)
    #
    #         # set cycle tsf inputs
    #         cycle_tsf_img = F.grid_sample(fake_tsf_imgs, T_cycle)
    #         cycle_tsf_inputs = torch.cat([cycle_tsf_img, src_inputs[:, 3:]], dim=1)
    #
    #         return cycle_src_inputs, cycle_tsf_inputs
    #
    #     def inference(src_inputs, tsf_inputs, T, T_cycle, src_fim, tsf_fim):
    #         fake_src_color, fake_src_mask, fake_tsf_color, fake_tsf_mask = \
    #             self.generator.infer_front(src_inputs, tsf_inputs, T=T)
    #
    #         fake_src_imgs = fake_src_mask * init_bg + (1 - fake_src_mask) * fake_src_color
    #         fake_tsf_imgs = fake_tsf_mask * init_bg + (1 - fake_tsf_mask) * fake_tsf_color
    #
    #         if self._opt.front_warp:
    #             fake_tsf_imgs = self.warp(fake_tsf_imgs, tsf_inputs[:, 0:3], tsf_fim, fake_tsf_mask)
    #
    #         cycle_src_inputs, cycle_tsf_inputs = set_cycle_inputs(
    #             fake_tsf_imgs, src_inputs, tsf_inputs, T_cycle)
    #
    #         cycle_src_color, cycle_src_mask, cycle_tsf_color, cycle_tsf_mask = \
    #             self.generator.infer_front(cycle_src_inputs, cycle_tsf_inputs, T=T_cycle)
    #
    #         cycle_src_imgs = cycle_src_mask * init_bg + (1 - cycle_src_mask) * cycle_src_color
    #         cycle_tsf_imgs = cycle_tsf_mask * init_bg + (1 - cycle_tsf_mask) * cycle_tsf_color
    #
    #         if self._opt.front_warp:
    #             cycle_tsf_imgs = self.warp(cycle_tsf_imgs, src_inputs[:, 0:3], src_fim, fake_src_mask)
    #
    #         return fake_src_imgs, fake_tsf_imgs, cycle_src_imgs, cycle_tsf_imgs, fake_src_mask, fake_tsf_mask, cycle_tsf_inputs
    #
    #     def create_criterion():
    #         face_criterion = FaceLoss(pretrained_path=self._opt.face_model).cuda()
    #         idt_criterion = torch.nn.L1Loss()
    #         mask_criterion = torch.nn.BCELoss()
    #
    #         return face_criterion, idt_criterion, mask_criterion
    #
    #     def print_losses(*args, **kwargs):
    #
    #         print('step = {}'.format(kwargs['step']))
    #         for key, value in kwargs.items():
    #             if key == 'step':
    #                 continue
    #             print('\t{}, {:.6f}'.format(key, value.item()))
    #
    #     def update_learning_rate(optimizer, current_lr, init_lr, final_lr, nepochs_decay):
    #         # updated learning rate G
    #         lr_decay = (init_lr - final_lr) / nepochs_decay
    #         current_lr -= lr_decay
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = current_lr
    #         # print('update G learning rate: %f -> %f' % (current_lr + lr_decay, current_lr))
    #         return current_lr
    #
    #     init_lr = 0.0002
    #     cur_lr = init_lr
    #     final_lr = 0.00001
    #     fix_iters = 25
    #     total_iters = 50
    #     optimizer = torch.optim.Adam(self.generator.parameters(), lr=init_lr, betas=(0.5, 0.999))
    #     face_cri, idt_cri, msk_cri = create_criterion()
    #
    #     # set up inputs
    #     src_fim, tsf_fim, j2ds, T, T_cycle, src_inputs, tsf_inputs, \
    #     src_imgs, init_preds, pseudo_masks = set_inputs(
    #         src_info=self.src_info, tsf_info=self.tsf_info
    #     )
    #
    #     logger = tqdm(range(total_iters))
    #     for step in logger:
    #
    #         fake_src_imgs, fake_tsf_imgs, cycle_src_imgs, cycle_tsf_imgs, \
    #         fake_src_mask, fake_tsf_mask, cycle_tsf_inputs = inference(src_inputs, tsf_inputs,
    #                                                                    T, T_cycle, src_fim, tsf_fim)
    #
    #         # cycle reconstruction loss
    #         cycle_loss = idt_cri(src_imgs, fake_src_imgs) + idt_cri(src_imgs, cycle_tsf_imgs)
    #
    #         # structure loss
    #         bg_mask = src_inputs[:, -1:]
    #         body_mask = 1.0 - bg_mask
    #         str_src_imgs = src_imgs * body_mask
    #         cycle_warp_imgs = cycle_tsf_inputs[:, 0:3]
    #         # back_head_mask = 1 - self.render.encode_front_fim(tsf_fim, transpose=True, front_fn=False)
    #         # struct_loss = idt_cri(init_preds, fake_tsf_imgs) + \
    #         #               2 * idt_cri(str_src_imgs * back_head_mask, cycle_warp_imgs * back_head_mask)
    #
    #         struct_loss = idt_cri(init_preds, fake_tsf_imgs) + \
    #                       2 * idt_cri(str_src_imgs, cycle_warp_imgs)
    #
    #         # fid_loss = face_cri(src_imgs, cycle_tsf_imgs, kps1=j2ds[:, 0], kps2=j2ds[:, 0]) + \
    #         #            face_cri(init_preds, fake_tsf_imgs, kps1=j2ds[:, 1], kps2=j2ds[:, 1])
    #
    #         fid_loss = face_cri(src_imgs, cycle_tsf_imgs, kps1=j2ds[:, 0], kps2=j2ds[:, 0]) + \
    #                    face_cri(tsf_inputs[:, 0:3], fake_tsf_imgs, kps1=j2ds[:, 1], kps2=j2ds[:, 1])
    #
    #         # mask loss
    #         # mask_loss = msk_cri(fake_tsf_mask, tsf_inputs[:, -1:]) + msk_cri(fake_src_mask, src_inputs[:, -1:])
    #         mask_loss = msk_cri(torch.cat([fake_src_mask, fake_tsf_mask], dim=0), pseudo_masks)
    #
    #         loss = 10 * cycle_loss + 10 * struct_loss + fid_loss + 5 * mask_loss
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print_losses(step=step, total=loss, cyc=cycle_loss,
    #         #              str=struct_loss, fid=fid_loss, msk=mask_loss)
    #
    #         if verbose:
    #             logger.set_description(
    #                 (
    #                     f'step: {step}; '
    #                     f'total: {loss.item():.6f}; cyc: {cycle_loss.item():.6f}; '
    #                     f'str: {struct_loss.item():.6f}; fid: {fid_loss.item():.6f}; '
    #                     f'msk: {mask_loss.item():.6f}'
    #                 )
    #             )
    #
    #         if step % 10 == 0:
    #             self.visualize(visualizer, input_imgs=src_imgs, tsf_imgs=fake_tsf_imgs,
    #                            cyc_imgs=cycle_tsf_imgs, fake_tsf_mask=fake_tsf_mask,
    #                            init_preds=init_preds,
    #                            str_src_imgs=str_src_imgs,
    #                            cycle_warp_imgs=cycle_warp_imgs)
    #
    #         if step > fix_iters:
    #             cur_lr = update_learning_rate(optimizer, cur_lr, init_lr, final_lr, fix_iters)
    #
    #     self.generator.eval()
