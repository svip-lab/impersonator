import os
import torch
import torch.nn.functional as F
from .models import BaseModel
from networks.networks import NetworksFactory, HumanModelRecovery
from networks.detectors import PersonMaskRCNNDetector
from networks.sa_generator import InpaintSANet
from utils.nmr import SMPLRenderer
from utils.util import to_tensor
import utils.cv_utils as cv_utils

import ipdb


class Imitator(BaseModel):
    def __init__(self, opt):
        super(Imitator, self).__init__(opt)
        self._name = 'Imitator_v2'

        # create networks
        self._init_create_networks()

        # 4. pre-processor
        if self._opt.has_detector:
            print('loading detector')
            self.detector = PersonMaskRCNNDetector(ks=self._opt.bg_ks, threshold=0.5, to_gpu=True)
        else:
            self.detector = None

        # 5. bg-pretrain
        if self._opt.bg_pretrain:
            print('loading bg pretrain model')
            self.bg_net = self._create_bgnet().cuda()
        else:
            self.bg_net = None

        # prefetch variables
        self.src_info = None
        self.tsf_info = None
        self.first_cam = None
        self.initial_T = torch.zeros(opt.image_size, opt.image_size, 2, dtype=torch.float32).cuda() - 1.0
        self.T = None

    def _create_generator(self):
        net = NetworksFactory.get_by_name(self._opt.gen_name, bg_dim=4, src_dim=3+self._G_cond_nc,
                                          tsf_dim=3+self._G_cond_nc, repeat_num=self._opt.repeat_num).cuda()

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
                              anti_aliasing=True, background_color=(0, 0, 0)).cuda()
        return render

    def _create_bgnet(self):
        net = InpaintSANet(c_dim=4)
        self._load_params(net, self._opt.bg_pretrain, need_module=False)
        net.eval()
        return net

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

    def imitate(self, tgt_paths, tgt_smpls=None, cam_strategy='smooth', output_dir='', visualizer=None):
        length = len(tgt_paths)

        outputs = []
        for t in range(length):
            tgt_path = tgt_paths[t]
            tgt_smpl = tgt_smpls[t] if tgt_smpls is not None else None

            tsf_inputs = self.transfer(tgt_path, tgt_smpl, cam_strategy, t=t)

            preds = self.forward(tsf_inputs, self.T, visualizer=visualizer)
            outputs.append(preds)

            if visualizer is not None:
                gt = cv_utils.transform_img(self.tsf_info['image'], image_size=self._opt.image_size, transpose=True)
                visualizer.vis_named_img('pred_' + cam_strategy, preds)
                visualizer.vis_named_img('gt', gt[None, ...], normalize=True)

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

    def transfer(self, tgt_path, tgt_smpl=None, cam_strategy='smooth', t=0):
        with torch.no_grad():
            # get source info
            src_info = self.src_info

            ori_img = cv_utils.read_cv2_img(tgt_path)
            if tgt_smpl is None:
                img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
                img_hmr = torch.FloatTensor(img_hmr).cuda()[None, ...]
                tgt_smpl = self.hmr(img_hmr)[-1]
            else:
                tgt_smpl = to_tensor(tgt_smpl).cuda()[None, ...]

            if t == 0 and cam_strategy == 'smooth':
                self.first_cam = tgt_smpl[:, 0:3].clone()

            # get transfer smpl
            tsf_smpl = self.swap_smpl(src_info['cam'], src_info['shape'], tgt_smpl, cam_strategy=cam_strategy)
            # transfer process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
            tsf_info = self.hmr.get_details(tsf_smpl)

            tsf_img, _ = self.render.render(tsf_info['cam'], tsf_info['verts'], src_info['tex'],
                                            reverse_yz=True, get_fim=False)
            tsf_info['cond'], tsf_info['fim'] = self.render.encode_fim(tsf_info['cam'],
                                                                       tsf_info['verts'], transpose=True)
            tsf_inputs = torch.cat([tsf_img, tsf_info['cond']], dim=1)

            T = self.calculate_trans(src_info['bc_f2pts'], src_info['fim'], tsf_info['fim'])

            # add target image to tsf info
            tsf_info['image'] = ori_img

            self.T = T
            self.tsf_info = tsf_info

            return tsf_inputs

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

    def personalize(self, src_path, src_smpl=None, output_path='', visualizer=None):

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
            src_info['image'] = ori_img

            # add texture into source info
            _, src_info['tex'] = self.render.forward(src_info['cam'], src_info['verts'],
                                                     img, is_uv_sampler=False, reverse_yz=True, get_fim=False)

            # add pose condition and face index map into source info
            src_info['cond'], src_info['fim'] = self.render.encode_fim(src_info['cam'],
                                                                       src_info['verts'], transpose=True)

            # bg input and inpaiting background
            # TODO

            if self.detector is not None:
                bbox, src_bg_mask = self.detector.inference(img[0])
            else:
                if self._opt.bg_replace:
                    src_bg_mask = self.correct_morph(src_info['cond'][:, -1:, :, :], ks=15, mode='erode')
                else:
                    src_bg_mask = self.morph(src_info['cond'][:, -1:, :, :], ks=15, mode='erode')

            if self.bg_net is not None:
                src_info['bg'] = self.bg_net(img, masks=src_bg_mask, only_x=True)
            else:
                src_bg_mask = 1 - src_bg_mask
                bg_inputs = torch.cat([img * src_bg_mask, src_bg_mask], dim=1)
                src_info['bg'] = self.model.bg_model(bg_inputs)

            # source identity
            # src_crop_mask = self.morph(src_info['cond'][:, -1:, :, :], ks=3, mode='erode')
            src_crop_mask = self.correct_morph(src_info['cond'][:, -1:, :, :], ks=3, mode='erode')
            src_inputs = torch.cat([img * (1 - src_crop_mask), src_info['cond']], dim=1)
            src_info['feats'] = self.model.src_model.inference(src_inputs)

            self.src_info = src_info

            if visualizer is not None:
                visualizer.vis_named_img('src', img)
                visualizer.vis_named_img('bg', src_info['bg'])
                visualizer.vis_named_img('src_fim', src_info['fim'])
                visualizer.vis_named_img('src_bg_mask', src_bg_mask)

            if output_path:
                cv_utils.save_cv2_img(src_info['image'], output_path, image_size=self._opt.image_size)

    def morph(self, src_bg_mask, ks, mode='erode'):
        n_ks = ks ** 2
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32).cuda()
        out = F.conv2d(src_bg_mask, kernel, padding=ks // 2)

        if mode == 'erode':
            out = (out == n_ks).float()
        else:
            out = (out >= 1).float()

        return out

    def correct_morph(self, src_bg_mask, ks, mode='erode'):
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

    def forward(self, tsf_inputs, T, visualizer=None):
        with torch.no_grad():
            # generate fake images

            bg_img = self.src_info['bg']
            src_encoder_outs, src_resnet_outs = self.src_info['feats']

            tsf_color, tsf_mask = self.model.inference(src_encoder_outs, src_resnet_outs, tsf_inputs, T)
            tsf_mask = self._do_if_necessary_saturate_mask(tsf_mask, saturate=self._opt.do_saturate_mask)
            # # tsf_mask = (tsf_mask > 0.5).float()
            # tsf_mask = self.correct_morph(tsf_mask, ks=3, mode='dilate')
            pred_imgs = tsf_mask * bg_img + (1 - tsf_mask) * tsf_color

            if visualizer is not None:
                visualizer.vis_named_img('tsf_mask', tsf_mask)

        return pred_imgs

    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m
