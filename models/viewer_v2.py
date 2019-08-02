import os
import torch
import torch.nn.functional as F
from .models import BaseModel
from networks.networks import NetworksFactory, HumanModelRecovery
from utils.nmr import SMPLRenderer
from utils.util import to_tensor
import utils.cv_utils as cv_utils

import ipdb


class Viewer(BaseModel):

    def __init__(self, opt):
        super(Viewer, self).__init__(opt)
        self._name = 'Viewer'

        # create networks
        self._init_create_networks()

        # prefetch variables
        self.src_info = None
        self.tgt_info = None

        # initialize T
        self.initial_T = torch.zeros(opt.image_size, opt.image_size, 2, dtype=torch.float32).cuda() - 1.0

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

    def setup(self, src_path, src_smpl=None):
        with torch.no_grad():
            self.src_info = self.personalize(src_path, src_smpl=src_smpl)

    def rotate_trans(self, rt, t, X):
        R = cv_utils.euler2matrix(rt)    # (3 x 3)

        R = torch.FloatTensor(R)[None, :, :].cuda()
        t = torch.FloatTensor(t)[None, None, :].cuda()

        # (bs, Nv, 3) + (bs, 1, 3)
        return torch.bmm(X, R) + t

    def view(self, rt, t, visualizer=None, name='1'):

        src_info = self.src_info
        src_mesh = self.src_info['verts']
        tsf_mesh = self.rotate_trans(rt, t, src_mesh)

        tsf_img, _ = self.render.render(src_info['cam'], tsf_mesh, src_info['tex'], reverse_yz=True, get_fim=False)
        tsf_cond, tsf_fim = self.render.encode_fim(src_info['cam'], tsf_mesh, transpose=True)
        tsf_inputs = torch.cat([tsf_img, tsf_cond], dim=1)
        T = self.calculate_trans(src_info['bc_f2pts'], src_info['fim'], tsf_fim)

        bg = torch.zeros_like(src_info['bg'])
        preds = self.forward(tsf_inputs, src_info['feats'], T, bg)

        if visualizer is not None:
            # self.render.set_ambient_light()
            # textures = self.render.debug_textures()[None, ...].cuda()
            # src_mesh, _ = self.render.render(src_info['cam'], src_X, textures, reverse_yz=True, get_fim=False)
            # tsf_mesh, _ = self.render.render(src_info['cam'], tsf_X, textures, reverse_yz=True, get_fim=False)

            visualizer.vis_named_img('src_img', src_info['image'])
            visualizer.vis_named_img('pred_' + name, preds)
            visualizer.vis_named_img('cond_' + name, tsf_cond)

        return preds

    def calculate_trans(self, bc_f2pts, src_fim, tsf_dim):
        bs = src_fim.shape[0]
        T = self.initial_T.repeat(bs, 1, 1, 1)   # (bs, image_size, image_size, 2)

        tsf_ids = tsf_dim != -1

        for i in range(bs):
            Ti = T[i]

            tsf_i = tsf_ids[i]

            # (nf, 2)
            tsf_flows = bc_f2pts[i, tsf_dim[i, tsf_i].long()]      # (nt, 2)
            Ti[tsf_i] = tsf_flows

        return T

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

    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m
