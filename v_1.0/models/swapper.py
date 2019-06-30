import os
import torch
import torch.nn.functional as F
from .models import BaseModel
from networks.networks import NetworksFactory, HumanModelRecovery
from utils.nmr import SMPLRenderer
from utils.util import to_tensor
import utils.cv_utils as cv_utils

import ipdb


class Swapper(BaseModel):

    PART_IDS = {
        'body': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'upper_body': [1, 2, 3, 4, 9],
        'lower_body': [4, 5],
        'torso': [9],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    def __init__(self, opt):
        super(Swapper, self).__init__(opt)
        self._name = 'Swapper'

        # create networks
        self._init_create_networks()

        # prefetch variables
        self.src_info = None
        self.tgt_info = None

        # initialize T
        self.initial_T = torch.zeros(opt.image_size, opt.image_size, 2, dtype=torch.float32).cuda() - 1.0
        self.initial_T_grid = self._make_grid()

    def _make_grid(self):
        # initialize T
        image_size = self._opt.image_size
        xy = torch.arange(0, image_size, dtype=torch.float32) / (image_size - 1) * 2 - 1.0
        grid_y, grid_x = torch.meshgrid(xy, xy)

        # 1. make mesh grid
        T = torch.stack([grid_x, grid_y], dim=-1).cuda()  # (image_size, image_size, 2)

        return T

    def _create_generator(self):
        net = NetworksFactory.get_by_name(self._opt.gen_name, bg_dim=4, src_dim=3+self._G_cond_nc,
                                          tsf_dim=3+self._G_cond_nc, repeat_num=6).cuda()

        if self._opt.load_path:
            self._load_params(net, self._opt.load_path)
        elif self._opt.load_epoch > 0:
            self._load_network(net, 'G', self._opt.load_epoch)
        else:
            raise ValueError('load_path {} is empty and load_epoch {} is 0'.format(
                self._opt.load_path, self._opt.load_epoch))

        net.eval()
        return net

    def _create_mesh_model(self):
        hmr = HumanModelRecovery(smpl_pkl_path=self._opt.smpl_model).cuda()
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)

        hmr.eval()
        print('load hmr model from {}'.format(self._opt.hmr_model))
        return hmr

    def _create_render(self, faces):
        render = SMPLRenderer(faces=faces, map_name=self._opt.map_name, uv_map_path=self._opt.uv_mapping,
                              tex_size=self._opt.tex_size, image_size=self._opt.image_size, fill_back=True,
                              anti_aliasing=True, background_color=(0, 0, 0), has_front_map=True).cuda()
        return render

    def _init_create_networks(self):
        # generator network
        self.model = self._create_generator()
        self.hmr = self._create_mesh_model()
        self.render = self._create_render(self.hmr.smpl.faces)

    def _extract_smpls(self, input_file):
        img = cv_utils.read_cv2_img(input_file)
        img = cv_utils.transform_img(img, image_size=224) * 2 - 1.0  # hmr receive [-1, 1]
        img = img.transpose((2, 0, 1))
        img = torch.FloatTensor(img).cuda()[None, ...]
        theta = self.hmr(img)[-1]

        return theta

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

    def swap_setup(self, src_path, tgt_path, src_smpl=None, tgt_smpl=None, output_dir=''):

        with torch.no_grad():
            self.src_info = self.personalize(src_path, src_smpl)
            self.tgt_info = self.personalize(tgt_path, tgt_smpl)

    def swap_preview(self, src_info, tgt_info, target_part='lower_body', visualizer=None):
        with torch.no_grad():
            assert target_part in self.PART_IDS.keys()

            # get target selected face index map
            part_id = self.PART_IDS[target_part]
            part_mask = torch.sum(tgt_info['part'][:, part_id, ...], dim=1).byte()

            selected_fids = tgt_info['fim'][part_mask].unique().long()
            front_selected_fids = selected_fids[selected_fids < self.render.base_nf]
            back_selected_fids = selected_fids[selected_fids >= self.render.base_nf]

            # left id
            left_ids = [i for i in self.PART_IDS['all'] if i not in part_id]
            left_part_mask = torch.sum(src_info['part'][:, left_ids, ...], dim=1).byte()

            # fill back
            all_selected_fids = torch.cat([selected_fids, front_selected_fids + self.render.base_nf,
                                           back_selected_fids - self.render.base_nf], dim=0)

            # copy target selected part face into source
            tsf_tex = src_info['tex'].clone()
            tsf_tex[0, all_selected_fids, ...] = tgt_info['tex'][0, all_selected_fids, ...]

            # render tsf image
            tsf_img, _ = self.render.render(src_info['cam'], src_info['verts'], tsf_tex, reverse_yz=True, get_fim=False)
            tsf_inputs = torch.cat([tsf_img, src_info['cond']], dim=1)

            ipdb.set_trace()

            # part_mask = left_part_mask
            self.T12 = self.calculate_trans(tgt_info['bc_f2pts'], src_info['fim'], tgt_info['fim'], part_mask)
            self.T21 = self.calculate_trans(src_info['bc_f2pts'], tgt_info['fim'], src_info['fim'], left_part_mask)

            tsf12 = self.model.transform(tgt_info['image'], self.T21)
            tsf21 = self.model.transform(src_info['image'], self.T12)

            visualizer.vis_named_img('tsf12', tsf12)
            visualizer.vis_named_img('tsf21', tsf21)

            # front net
            pred_imgs = self.forward(tsf_inputs, tgt_info['feats'], src_info['feats'],
                                     self.T12, self.T21, src_info['bg'])
            # pred_imgs = self.auto_encoder(tsf_inputs, src_info['bg'])

            if visualizer is not None:
                visualizer.vis_named_img('tsf_img', tsf_img)
                visualizer.vis_named_img('pred_img', pred_imgs)
                visualizer.vis_named_img('part_mask', part_mask)

            return pred_imgs

    def swap(self, src_info, tgt_info, target_part='lower_body', visualizer=None):
        with torch.no_grad():
            assert target_part in self.PART_IDS.keys()

            # get target selected face index map
            selected_part_id = self.PART_IDS[target_part]
            left_id = [i for i in self.PART_IDS['all'] if i not in selected_part_id]

            src_part_mask = (torch.sum(src_info['part'][:, selected_part_id, ...], dim=1) != 0).byte()
            tgt_part_mask = (torch.sum(tgt_info['part'][:, selected_part_id, ...], dim=1) != 0).byte()
            src_left_mask = torch.sum(src_info['part'][:, left_id, ...], dim=1).byte()

            T11, T21, T = self.calculate_trans(tgt_info['bc_f2pts'], src_info['fim'], tgt_info['fim'], src_part_mask, src_left_mask)

            tsf21 = self.model.transform(tgt_info['image'], T21)
            tsf11 = self.model.transform(src_info['image'], T11)

            tsf_img = tsf21 * (src_part_mask[:, None, :, :].float()) + tsf11 * (src_left_mask[:, None, :, :].float())

            # tsf_img = torch.zeros_like(tsf21)
            # tsf_img[0, :, src_part_mask[0]] = tsf21[0, :, src_part_mask[0]]
            # tsf_img[0, :, src_left_mask[0]] = tsf11[0, :, src_left_mask[0]]

            tsf_inputs = torch.cat([tsf_img, src_info['cond']], dim=1)
            # feats = self.model.src_model.inference(tsf_inputs)
            # preds = self.forward(tsf_inputs, feats, T, src_info['bg'])

            preds = self.forward2(tsf_inputs, tgt_info['feats'], T21, src_info['feats'], T11, src_info['bg'])

            if visualizer is not None:
                visualizer.vis_named_img('src_part_mask', src_part_mask)
                visualizer.vis_named_img('src_left_mask', src_left_mask)
                visualizer.vis_named_img('tgt_part_mask', tgt_part_mask)
                visualizer.vis_named_img('tsf21', tsf21)
                visualizer.vis_named_img('tsf11', tsf11)
                visualizer.vis_named_img('tsf_img', tsf_img)
                visualizer.vis_named_img('pred_img', preds)
                visualizer.vis_named_img('src_img', src_info['image'])
                visualizer.vis_named_img('tgt_img', tgt_info['image'])
                visualizer.vis_named_img('src_cond', src_info['cond'])
                visualizer.vis_named_img('tsf_cond', tgt_info['cond'])

                src_cond_left = torch.zeros_like(src_info['cond'])
                src_cond_left[:, 2, :, :] = 1.0
                src_cond_mask = src_cond_left.clone()
                tgt_cond_mask = src_cond_left.clone()

                src_cond_left[0, :, src_left_mask[0]] = src_info['cond'][0, :, src_left_mask[0]]
                src_cond_mask[0, :, src_part_mask[0]] = src_info['cond'][0, :, src_part_mask[0]]
                tgt_cond_mask[0, :, tgt_part_mask[0]] = tgt_info['cond'][0, :, tgt_part_mask[0]]

                visualizer.vis_named_img('src_cond_left', src_cond_left)
                visualizer.vis_named_img('src_cond_mask', src_cond_mask)
                visualizer.vis_named_img('tgt_cond_mask', tgt_cond_mask)

            return preds

    def calculate_trans(self, bc_f2pts, src_fim, tsf_dim, src_part_mask, src_left_mask):
        bs = tsf_dim.shape[0]
        grid_T = self.initial_T_grid.clone()
        T = self.initial_T_grid.repeat(bs, 1, 1, 1)
        T11 = self.initial_T.repeat(bs, 1, 1, 1)
        T21 = self.initial_T.repeat(bs, 1, 1, 1)   # (bs, image_size, image_size, 2)
        for i in range(bs):
            Ti21 = T21[i]
            src_i = src_fim[i, src_part_mask[i]].long()
            # (nf, 2)
            src_flows = bc_f2pts[i, src_i]      # (nt, 2)
            Ti21[src_part_mask[i]] = src_flows

            T11[i, src_left_mask[i]] = grid_T[src_left_mask[i]]

        return T11, T21, T

    def get_src_bc_f2pts(self, src_cams, src_verts):

        points = self.render.batch_orth_proj_idrot(src_cams, src_verts)
        f2pts = self.render.points_to_faces(points)
        bc_f2pts = self.render.compute_barycenter(f2pts)  # (bs, nf, 2)

        return bc_f2pts

    def personalize(self, src_path, src_smpl=None, ):

        with torch.no_grad():
            ori_img = cv_utils.read_cv2_img(src_path)

            # resize image and convert the color space from [0, 255] to [-1, 1]
            img = cv_utils.transform_img(ori_img, self._opt.image_size, transpose=True) * 2 - 1.0
            img = torch.FloatTensor(img).cuda()[None, ...]

            if src_smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.FloatTensor(img_hmr).cuda()[None, ...]
                src_smpl = self.hmr(img_hmr)[-1]
            else:
                src_smpl = to_tensor(src_smpl).cuda()[None, ...]

            # source process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
            src_info = self.hmr.get_details(src_smpl)

            # add source bary-center points
            src_info['bc_f2pts'] = self.get_src_bc_f2pts(src_info['cam'], src_info['verts'])

            # add image to source info
            src_info['image'] = img

            # add texture into source info
            _, src_info['tex'] = self.render.forward(src_info['cam'], src_info['verts'],
                                                     img, is_uv_sampler=False, reverse_yz=True, get_fim=False)

            # add pose condition and face index map into source info
            src_info['cond'], src_info['fim'] = self.render.encode_fim(src_info['cam'],
                                                                       src_info['verts'], transpose=True)

            # add part condition into source info
            src_info['part'] = self.render.encode_front_fim(src_info['fim'], transpose=True)

            # bg input and inpaiting background
            src_bg_mask = self.morph(src_info['cond'][:, -1:, :, :], ks=15, mode='erode')
            bg_inputs = torch.cat([img * src_bg_mask, src_bg_mask], dim=1)
            src_info['bg'] = self.model.bg_model(bg_inputs)
            #
            # source identity
            src_crop_mask = self.morph(src_info['cond'][:, -1:, :, :], ks=3, mode='erode')
            src_inputs = torch.cat([img * (1 - src_crop_mask), src_info['cond']], dim=1)
            src_info['feats'] = self.model.src_model.inference(src_inputs)
            #
            # self.src_info = src_info

            return src_info

    def morph(self, src_bg_mask, ks, mode='erode'):
        n_ks = ks ** 2
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32).cuda()
        out = F.conv2d(src_bg_mask, kernel, padding=ks // 2)

        if mode == 'erode':
            out = (out == n_ks).float()
        else:
            out = (out >= 1).float()

        return out

    def forward(self, tsf_inputs, feats, T, bg):
        with torch.no_grad():
            # generate fake images
            src_encoder_outs, src_resnet_outs = feats

            tsf_color, tsf_mask = self.model.inference(src_encoder_outs, src_resnet_outs, tsf_inputs, T)
            tsf_mask = self._do_if_necessary_saturate_mask(tsf_mask, saturate=self._opt.do_saturate_mask)
            pred_imgs = tsf_mask * bg + (1 - tsf_mask) * tsf_color

        return pred_imgs

    def forward2(self, tsf_inputs, feats21, T21, feats11, T11, bg):
        with torch.no_grad():
            # generate fake images
            src_encoder_outs21, src_resnet_outs21 = feats21
            src_encoder_outs11, src_resnet_outs11 = feats11

            tsf_color, tsf_mask = self.model.swap(tsf_inputs, src_encoder_outs21, src_encoder_outs11,
                                                  src_resnet_outs21, src_resnet_outs11, T21, T11)
            tsf_mask = self._do_if_necessary_saturate_mask(tsf_mask, saturate=self._opt.do_saturate_mask)
            pred_imgs = tsf_mask * bg + (1 - tsf_mask) * tsf_color

        return pred_imgs

    def auto_encoder(self, tsf_inputs, bg_img):
        with torch.no_grad():
            tsf_color, tsf_mask = self.model.src_model(tsf_inputs)
            tsf_mask = self._do_if_necessary_saturate_mask(tsf_mask, saturate=self._opt.do_saturate_mask)
            pred_imgs = tsf_mask * bg_img + (1 - tsf_mask) * tsf_color

        return pred_imgs

    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m
