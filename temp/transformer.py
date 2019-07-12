import os
from utils import cv_utils
import torch
import torch.nn.functional as F
import numpy as np
from networks.bodymesh.hmr import HumanModelRecovery
from data.dataset import DatasetFactory
from utils.visualizer.demo_visualizer import MotionImitationVisualizer
from options.test_options import TestOptions

from networks.bodymesh.nmr import SMPLRenderer
import time


class MotionImitator(object):
    def __init__(self, opt):
        self._opt = opt
        # self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        # self._model.set_eval()

        self._hmr = HumanModelRecovery(self._opt.smpl_model).cuda()
        self.load_model(self._opt.hmr_model)
        self._hmr.eval()

        self._render = SMPLRenderer(faces=self._hmr.smpl.faces,
                                    uv_map_path=self._opt.uv_mapping,
                                    tex_size=self._opt.tex_size,
                                    image_size=self._opt.image_size, fill_back=True,
                                    anti_aliasing=True, background_color=(0, 0, 0))

        if self._opt.visual:
            self._visualizer = MotionImitationVisualizer(env=self._opt.name, ip=self._opt.ip, port=self._opt.port)

        self._src_img = None
        self._tgt_img = None

    def _extract_smpls(self, input_file):
        img = cv_utils.read_cv2_img(input_file)
        img = cv_utils.transform_img(img, image_size=224) * 2 - 1.0  # hmr receive [-1, 1]
        img = img.transpose((2, 0, 1))
        img = torch.FloatTensor(img).cuda()[None, ...]
        theta = self._hmr(img)[-1]

        self._tgt_img = img

        return theta

    def _cal_smpl_outs(self, tgt_file, get_detail=False):
        if isinstance(tgt_file, str):
            tgt_cond = self._extract_smpls(tgt_file)

        elif isinstance(tgt_file, torch.FloatTensor):
            tgt_cond = tgt_file[None, ...].cuda()

        elif isinstance(tgt_file, np.ndarray):
            tgt_cond = torch.FloatTensor(tgt_file[None, ...]).cuda()

        else:
            raise ValueError('type error of {}'.format(type(tgt_file)))

        smpl_out = tgt_cond
        if get_detail:
            smpl_out = self._hmr.get_details(tgt_cond)

        return smpl_out

    def imitate(self, src_img_file, tgt_seqs_file):
        with torch.no_grad():
            # 1. load source image
            org_img = cv_utils.read_cv2_img(src_img_file)
            src_imgs = cv_utils.transform_img(org_img, image_size=self._opt.image_size) * 2 - 1.0  # hmr receive [-1, 1]
            src_imgs = torch.FloatTensor(src_imgs.transpose((2, 0, 1)))[None, ...].cuda()
            src_info = self._cal_smpl_outs(src_img_file, get_detail=True)

            src_rd, src_info['tex'] = self._render.forward(src_info['cam'], src_info['verts'],
                                                           src_imgs, is_uv_sampler=False,
                                                           reverse_yz=True, get_fim=False)

            src_fim = self._render.infer_face_index_map(src_info['cam'], src_info['verts'])

            # ipdb.set_trace()
            for t, tgt_file in enumerate(tgt_seqs_file):
                # torch.cuda.FloatTensor, (1, 85)
                tgt_info = self._cal_smpl_outs(tgt_file, get_detail=True)
                tgt_rd, _ = self._render.render(tgt_info['cam'], tgt_info['verts'], src_info['tex'],
                                                reverse_yz=True, get_fim=False)
                tgt_fim = self._render.infer_face_index_map(tgt_info['cam'], tgt_info['verts'])

                T = self.transformer(src_info['cam'], src_info['verts'], src_fim, tgt_fim)

                if self._opt.visual:
                    self._visualizer.vis_named_img('source', src_imgs)

                print(T.shape)
                for scale in [256, 128, 64, 32, 16, 8]:
                    T_scale = T.permute(0, 3, 1, 2)     # (bs, 2, h, w)
                    T_scale = F.interpolate(T_scale, size=(scale, scale), mode='bilinear', align_corners=True)
                    T_scale = T_scale.permute(0, 2, 3, 1)   # (bs, h, w, 2)

                    print(scale, T_scale.shape)
                    T_trans = F.grid_sample(src_imgs, T_scale)
                    print(src_imgs.shape, T_trans.shape)

                    if self._opt.visual:
                        self._visualizer.vis_named_img('transf_%d' % scale, T_trans)

                time.sleep(1.0)

    def transformer(self, src_cams, src_verts, src_fim, tgt_fim):
        bs = src_fim.shape[0]

        image_size = self._opt.image_size
        xy = torch.arange(0, image_size, dtype=torch.float32) / (image_size - 1) * 2 - 1.0
        grid_y, grid_x = torch.meshgrid(xy, xy)

        # 1. make mesh grid
        T = torch.stack([grid_x, grid_y], dim=-1).cuda()   # (image_size, image_size, 2)
        T = T.repeat(bs, 1, 1, 1)   # (bs, image_size, image_size, 2)

        # 2. calculate occlusion flows, (bs, no, 2)
        src_ids = src_fim != -1
        src_bgs = src_fim == -1
        tgt_ids = tgt_fim != -1

        # 3. calculate tgt flows, (bs, nt, 2)
        points = self._render.batch_orth_proj_idrot(src_cams, src_verts)
        f2pts = self._render.points_to_faces(points)
        bc_f2pts = self._render.compute_barycenter(f2pts)  # (bs, nf, 2)

        for i in range(bs):
            Ti = T[i]

            src_bi = src_bgs[i]
            src_fi = src_ids[i]
            tgt_i = tgt_ids[i]

            # (ns, 2) = -1.0
            Ti[src_fi] = -1.0

            # (nf, 2)
            tgt_flows = bc_f2pts[i, tgt_fim[i, tgt_i].long()]      # (nt, 2)
            Ti[tgt_i] = tgt_flows

        return T

    def transformer_1(self, src_cams, src_verts, src_fim, tgt_fim):
        bs = src_fim.shape[0]

        image_size = self._opt.image_size
        xy = torch.arange(0, image_size, dtype=torch.float32) / (image_size - 1) * 2 - 1.0
        grid_y, grid_x = torch.meshgrid(xy, xy)

        # 1. make mesh grid
        T = torch.stack([grid_x, grid_y], dim=-1).cuda()   # (image_size, image_size, 2)
        T = T.repeat(bs, 1, 1, 1)   # (bs, image_size, image_size, 2)

        # 2. calculate occlusion flows, (bs, no, 2)
        src_ids = src_fim != -1
        src_bgs = src_fim == -1
        tgt_ids = tgt_fim != -1

        # 3. calculate tgt flows, (bs, nt, 2)
        points = self._render.batch_orth_proj_idrot(src_cams, src_verts)
        f2pts = self._render.points_to_faces(points)
        bc_f2pts = self._render.compute_barycenter(f2pts)  # (bs, nf, 2)

        for i in range(bs):
            Ti = T[i]

            src_bi = src_bgs[i]
            src_fi = src_ids[i]
            tgt_i = tgt_ids[i]
            src_coords = Ti[src_bi]  # (ns, 2)
            occ_coords = Ti[src_fi]  # (no, 2)

            norm_A = torch.sum(occ_coords ** 2, dim=1, keepdim=True)
            norm_B = torch.sum(src_coords ** 2, dim=1)[None, :]
            prod_AB = torch.matmul(occ_coords, src_coords.permute(1, 0))
            # print(norm_A.shape, norm_B.shape, prod_AB.shape)

            # ipdb.set_trace()
            occ2src_dist = norm_A + norm_B - 2 * prod_AB

            _, occ_flows = torch.median(occ2src_dist, dim=-1)   # (bs, no)

            # (ns, 2) = -1.0
            Ti[src_fi] = src_coords[occ_flows]

            # (nf, 2)
            tgt_flows = bc_f2pts[i, tgt_fim[i, tgt_i].long()]      # (nt, 2)
            Ti[tgt_i] = tgt_flows

        return T

    def load_model(self, pretrain_model):
        saved_data = torch.load(pretrain_model)
        self._hmr.load_state_dict(saved_data)
        print('load hmr model from {}'.format(pretrain_model))


def main():
    opt = TestOptions().parse()

    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    imitator = MotionImitator(opt)
    dataset = DatasetFactory.get_by_name(opt.dataset_mode, opt, is_for_train=False)
    videos_info = dataset.video_info
    num_videos = len(videos_info)

    for v_i in range(0, num_videos):
        src_info = videos_info[v_i]
        src_image_path = src_info['images'][0]
        for v_j in range(v_i + 3, num_videos):
            tgt_info = videos_info[v_j]
            tgt_images = tgt_info['images']
            # cams = tgt_info['cams']
            # thetas = tgt_info['thetas']
            # betas = tgt_info['betas']
            #
            # tgt_conds = np.concatenate([cams, thetas, betas], axis=1)

            imitator.imitate(src_image_path, tgt_images)


if __name__ == '__main__':
    main()
