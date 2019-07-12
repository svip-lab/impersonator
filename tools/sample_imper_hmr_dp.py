import numpy as np
import cv2
import argparse
import glob
import os
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from multiprocessing import Process

import ipdb

from networks.bodymesh.nmr import SMPLRendererV2
from utils.util import load_pickle_file, write_pickle_file
from utils.visualizer import MotionImitationVisualizer
import utils.cv_utils as cv_utils

import time


IMG_SIZE = 256
BG_KS = 9


def parse_args():
    parser = argparse.ArgumentParser(description='Options to run the inference scripts.')
    parser.add_argument('-g', '--gpu', type=int, nargs='*', required=True, help='the device id of gpu.')
    parser.add_argument('--root_dir', type=str, required=True, help='the path of video directory.')
    parser.add_argument('--out_dir', type=str, required=True, help='the path of video directory.')
    parser.add_argument('--smpl_root', type=str, help='the path of smpl data directory.')
    parser.add_argument('--num_process', type=int, default=1, help='number process')
    parser.add_argument('--split_by_gpu', action='store_true', help='split by gpu or not')

    return parser.parse_args()


def morph(src_bg_mask, ks, mode='erode'):
    src_bg_mask = torch.FloatTensor(src_bg_mask[None, None])
    device = src_bg_mask.device

    n_ks = ks ** 2
    kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32).to(device)

    pad_s = ks // 2

    if mode == 'erode':
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=1.0)
        # print(src_bg_mask.shape, src_bg_mask_pad.shape)
        out = F.conv2d(src_bg_mask_pad, kernel)
        # print(out.shape)
        out = (out == n_ks).float()
    else:
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=0.0)
        # print(src_bg_mask.shape, src_bg_mask_pad.shape)
        out = F.conv2d(src_bg_mask_pad, kernel)
        # print(out.shape)
        out = (out >= 1).float()

    out = out[0].numpy()
    return out


def convert_pair_im_paths(root_dir, pair_pkl_path):

    def cvt_fun(im_path):
        """
        Args:
            im_path (str): /p300/MI_dataset_new_order/006_1_1/425.jpg

        Returns:
            new_im_path (str): root_dir/006/1/1/425.jpg
        """
        im_path_splits = im_path.split('/')
        vid = str(im_path_splits[-2])
        cvt_vid = '/'.join(vid.split('_'))
        new_im_path = os.path.join(root_dir, cvt_vid, im_path_splits[-1])

        return new_im_path

    with open(pair_pkl_path, 'rb') as f:
        pair_list = pickle.load(f)
        new_pair_list = []

        for i, pairs in enumerate(pair_list):
            # ('/p300/MI_dataset_new_order/006_1_1/425.jpg',
            #  '/p300/MI_dataset_new_order/006_1_1/062.jpg')
            src_path = cvt_fun(pairs[0])
            dst_path = cvt_fun(pairs[1])

            new_pair_list.append((src_path, dst_path))

        return new_pair_list


def load_images(im_path):
    img = cv_utils.read_cv2_img(im_path)
    img = cv_utils.transform_img(img, IMG_SIZE, transpose=True)
    img = img * 2 - 1
    return img


def sample_pair_data(pair_im_paths, pair_data_path, smpl_data, render=None, visualizer=None):
    """
    Args:
        pair_im_paths (tuple or list): (src_path, dst_path)

        pair_data_path (str): pickle path and it contains following items:
            --from_pixel_xy_on_uv (np.ndarray.float64): (img_size, img_size, 2)
            --to_pixel_xy_on_uv (np.ndarray.float64): (img_size, img_size, 2)
            --from_face_index_map (np.ndarray.int64): (img_size, img_size, 1)
            --to_face_index_map (np.ndarray.int64): (img_size, img_size, 1)
            --T (np.ndarray.int64): (img_size, img_size, 2)

        smpl_data (np.ndarray.float32): smpl data of both source and target.

        render:

        visualizer:
    Returns:

    """
    src_path, dst_path = pair_im_paths
    pair_data = load_pickle_file(pair_data_path)

    # load images
    # src_img = np.transpose(cv_utils.read_cv2_img(src_path).astype(np.float32) / 255 * 2 - 1, axes=(2, 0, 1))
    # dst_img = np.transpose(cv_utils.read_cv2_img(dst_path).astype(np.float32) / 255 * 2 - 1, axes=(2, 0, 1))
    src_img = load_images(src_path)
    dst_img = load_images(dst_path)

    # load pickle data
    src_bg_mask = (pair_data['from_face_index_map'][:, :, 0] == -1).astype(np.float32)     # (img_size, img_size)
    dst_bg_mask = (pair_data['to_face_index_map'][:, :, 0] == -1).astype(np.float32)       # (img_size, img_size)

    src_img_gpu = torch.tensor(src_img[None]).float()

    # if visualizer is not None:
    #
    #     front_mask = (src_fim != -1).astype(np.float32)
    #     front_mask = morph(front_mask, ks=7, mode='dilate')
    #     front_img = src_img * front_mask
    #     bg_img = src_img * (1 - front_mask)
    #
    #     visualizer.vis_named_img('src_fim', src_fim[None, None])
    #     visualizer.vis_named_img('dst_fim', dst_fim[None, None])
    #     visualizer.vis_named_img('front', front_img[None])
    #     visualizer.vis_named_img('bg', bg_img[None])
    #     visualizer.vis_named_img('src_img', src_img[None])
    #     visualizer.vis_named_img('dst_img', dst_img[None])
    #     visualizer.vis_named_img('warp_T', warp_T)
    #     # time.sleep(1)

    # process smpl data
    # dict_keys(['pose', 'shape', 'cams', 'vertices'])
    cams = torch.tensor(smpl_data['cams']).float().cuda()
    verts = torch.tensor(smpl_data['verts']).float().cuda()

    src_cams = cams[0:1]
    src_verts = verts[0:1]
    src_img_gpu = src_img_gpu.cuda()

    dst_cams = cams[1:]
    dst_verts = verts[1:]

    src_f2verts, src_fim, src_wim = render.render_fim_wim(src_cams, src_verts)
    src_f2pts = src_f2verts[:, :, :, 0:2]
    src_f2pts[:, :, :, 1] *= -1
    dst_f2verts, dst_fim, dst_wim = render.render_fim_wim(dst_cams, dst_verts)

    T = render.cal_bc_transform(src_f2pts, dst_fim, dst_wim)

    warp_img = F.grid_sample(src_img_gpu, T)

    smpls = np.concatenate([smpl_data['cams'], smpl_data['pose'], smpl_data['shape']], axis=-1)

    sample = {
        'src_fim': src_fim[0].cpu().numpy(),
        'dst_fim': dst_fim[0].cpu().numpy(),
        'src_bg_mask': src_bg_mask,
        'dst_bg_mask': dst_bg_mask,
        'T': T[0].cpu().numpy(),
        'warp': warp_img[0].cpu().numpy(),
        'smpls': smpls
    }

    if visualizer is not None:
        visualizer.vis_named_img('src_fim', sample['src_fim'][None, None, :, :])
        visualizer.vis_named_img('dst_fim', sample['dst_fim'][None, None, :, :])
        visualizer.vis_named_img('src_bg_mask', sample['src_bg_mask'][None, None, :, :])
        visualizer.vis_named_img('dst_bg_mask', sample['dst_bg_mask'][None, None, :, :])
        visualizer.vis_named_img('warp', sample['warp'][None, :, :, :])
        visualizer.vis_named_img('src_img', src_img[None, :, :, :])
        visualizer.vis_named_img('dst_im', dst_img[None, :, :, :])

        time.sleep(3)

    return sample


def run_batch(root_dir, pkl_dir, out_dir, pair_im_paths, all_pkl_paths, smpl_root=None, visual=False):

    flag_ids = len(root_dir) + 1
    all_smpl_data = dict()

    def get_cams_verts(im_path):
        # ipdb.set_trace()
        vid, img_name = os.path.split(im_path)      # root_dir/v_name, img_name
        v_name = vid[flag_ids:]                 # v_name

        if v_name not in all_smpl_data:
            smpl_dir = os.path.join(smpl_root, v_name, 'pose_shape.pkl')
            all_smpl_data[v_name] = load_pickle_file(smpl_dir)

        smpl_data = all_smpl_data[v_name]

        t = int(img_name.split('.')[0])
        cams = smpl_data['cams'][t]
        verts = smpl_data['vertices'][t]
        poses = smpl_data['pose'][t]
        shapes = smpl_data['shape'][t]

        return cams, verts, poses, shapes

    def sample_smpl_data(im_paths):
        cams = []
        verts = []
        poses = []
        shapes = []
        for im_path in im_paths:
            cam_t, vert_t, pose_t, shape_t = get_cams_verts(im_path)
            cams.append(cam_t)
            verts.append(vert_t)
            poses.append(pose_t)
            shapes.append(shape_t)

        cams = np.stack(cams)
        verts = np.stack(verts)
        poses = np.stack(poses)
        shapes = np.stack(shapes)

        smpl_data = {
            'cams': cams,
            'verts': verts,
            'pose': poses,
            'shape': shapes
        }
        return smpl_data

    assert len(pair_im_paths) == len(all_pkl_paths)
    if visual:
        visualizer = MotionImitationVisualizer(env='sample_data', ip='http://10.19.129.76', port=10086)
    else:
        visualizer = None

    if smpl_root:
        render = SMPLRendererV2(image_size=IMG_SIZE, tex_size=3, has_front=True, fill_back=False)
    else:
        render = None

    out_flag_ids = len(pkl_dir) + 1
    length = len(pair_im_paths)
    for i in tqdm(range(length)):
        im_paths = pair_im_paths[i]
        pkl_path = all_pkl_paths[i]
        if smpl_root:
            smpl_data = sample_smpl_data(im_paths)
        else:
            smpl_data = None

        sample = sample_pair_data(im_paths, pkl_path, smpl_data, render, visualizer)

        # for key, value in sample.items():
        #     print(key, value.shape)

        out_pkl_path = os.path.join(out_dir, pkl_path[out_flag_ids:])
        # print(out_pkl_path)
        write_pickle_file(out_pkl_path, sample)


class Runner(Process):
    def __init__(self, root_dir, pkl_dir, out_dir, pair_im_paths, all_pkl_paths, smpl_root=None, visual=False, gpu=0):
        self.root_dir = root_dir
        self.pkl_dir = pkl_dir
        self.out_dir = out_dir
        self.pair_im_paths = pair_im_paths
        self.all_pkl_paths = all_pkl_paths
        self.smpl_root = smpl_root
        self.visual = visual
        self.gpu = gpu

        super(Runner, self).__init__(name='runner_%d' % gpu)

    def run(self):
        os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)

        run_batch(self.root_dir, self.pkl_dir, self.out_dir, self.pair_im_paths, self.all_pkl_paths,
                  self.smpl_root, self.visual)


def check_splits(split_pkl_paths, split_im_paths, all_pkl_paths, pair_im_paths):
    def merge_list(split_list):
        new_list = []
        for sub_list in split_list:
            new_list.extend(sub_list)
        return new_list

    def compare_two_list(a_list, b_list):
        assert len(a_list) == len(b_list)
        for a, b in zip(a_list, b_list):
            assert a == b, '{} != {}'.format(a, b)

    new_pkl_paths = merge_list(split_pkl_paths)
    new_im_paths = merge_list(split_im_paths)

    compare_two_list(new_pkl_paths, all_pkl_paths)
    compare_two_list(new_im_paths, pair_im_paths)

    print('splits successfully...')


'''
python tools/sample_imper_hmr.py   --gpu 9 \
        --root_dir /public/liuwen/p300/ImPer/motion_transfer   \
        --out_dir  /public/liuwen/p300/ImPer/val_samples \
        --smpl_root /public/liuwen/p300/ImPer/motion_transfer_smpl  \
        --num_process 1

'''


if __name__ == '__main__':
    args = parse_args()
    root_dir = args.root_dir
    out_dir = args.out_dir
    num_process = args.num_process

    # pkl_dir = '/home/liuwen/ImPer/train_dp_hmr_pairs_divide_idx_15_results'
    pkl_dir = '/home/liuwen/ImPer/val_dp_hmr_pairs_divide_idx_15_results'
    all_pkl_paths = sorted(glob.glob((os.path.join(pkl_dir, '*.pkl'))))

    # pair_pkl_path = '/public/liuwen/p300/ImPer/pairs_train_divide_idx_15.pkl'
    pair_pkl_path = '/public/liuwen/p300/ImPer/pairs_val_divide_idx_15.pkl'
    pair_im_paths = convert_pair_im_paths(root_dir, pair_pkl_path)

    assert len(all_pkl_paths) == len(pair_im_paths)
    length = len(all_pkl_paths)

    if args.split_by_gpu:
        pass

    else:
        if num_process == 1:
            os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu[0])
            run_batch(root_dir, pkl_dir, out_dir, pair_im_paths, all_pkl_paths,
                      smpl_root=args.smpl_root, visual=False)
        else:
            batch_size = int(np.ceil(length / num_process))
            split_pkl_paths = [all_pkl_paths[i:min(length, i + batch_size)] for i in range(0, length, batch_size)]
            split_im_paths = [pair_im_paths[i:min(length, i + batch_size)] for i in range(0, length, batch_size)]

            # 2.2 check
            print('total length = {}, num process  = {}'.format(length, num_process))
            check_splits(split_pkl_paths, split_im_paths, all_pkl_paths, pair_im_paths)

            # root_dir, pkl_dir, out_dir, pair_im_paths, all_pkl_paths, smpl_root=None, visual=False, gpu=0
            runners = [Runner(root_dir, pkl_dir, out_dir, sub_im_paths, sub_pkl_paths,
                              smpl_root=args.smpl_root, visual=True, gpu=0) for (sub_im_paths, sub_pkl_paths) in
                       zip(split_im_paths, split_pkl_paths)]

            for runner in runners:
                runner.start()

            for runner in runners:
                runner.join()
