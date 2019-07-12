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
NUM_SAMPLES = 256
RNG = np.random.RandomState(2019)


def parse_args():
    parser = argparse.ArgumentParser(description='Options to run the inference scripts.')
    parser.add_argument('-g', '--gpu', type=int, required=True, help='the device id of gpu.')
    parser.add_argument('--root_dir', type=str, default='/public/liuwen/p300/ImPer', help='the path of video directory.')
    parser.add_argument('--out_dir', type=str, default='/public/liuwen/p300/ImPer', help='the path of video directory.')
    return parser.parse_args()


def read_lines(file_path):
    with open(file_path, 'r') as reader:
        lines = []
        for line in reader:
            line = line.rstrip()
            lines.append(line)
        return lines


def sample_all_pairs(root_dir, is_train=True):

    if is_train:
        dst_file_path = os.path.join(root_dir, 'pairs_train.pkl')
    else:
        dst_file_path = os.path.join(root_dir, 'pairs_val.pkl')

    if os.path.exists(dst_file_path):
        pairs_list = load_pickle_file(dst_file_path)
        return pairs_list

    vids_dir = os.path.join(root_dir, 'motion_transfer')
    smpls_dir = os.path.join(root_dir, 'motion_transfer_smpl')

    txt_file = 'MI_train.txt' if is_train else 'MI_val.txt'
    txt_path = os.path.join(root_dir, txt_file)

    vid_lines = read_lines(txt_path)

    pairs_list = []
    for i, line in enumerate(vid_lines):
        images_path = os.listdir(os.path.join(vids_dir, line))
        images_path.sort()
        smpl_data = load_pickle_file(os.path.join(smpls_dir, line, 'pose_shape.pkl'))

        cams = smpl_data['cams']

        assert len(images_path) == len(cams), '{} != {}'.format(len(images_path), len(cams))

        print(line)
        length = len(images_path)
        for t in tqdm(range(NUM_SAMPLES)):
            from_ids = RNG.randint(0, 15)
            to_ids = RNG.randint(0, length)

            from_im_name = line + '/' + images_path[from_ids]
            to_im_name = line + '/' + images_path[to_ids]

            pairs_list.append((from_im_name, to_im_name))

    write_pickle_file(dst_file_path, pairs_list)

    return pairs_list


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


def load_images(im_path):
    img = cv_utils.read_cv2_img(im_path)
    img = cv_utils.transform_img(img, IMG_SIZE, transpose=True)
    img = img * 2 - 1
    return img


def sample_pair_data(pair_im_paths, smpl_data, render, visualizer=None):
    """
    Args:
        pair_im_paths (tuple or list): (src_path, dst_path)

        smpl_data (np.ndarray.float32): smpl data of both source and target.

        render:

        visualizer:
    Returns:

    """
    src_path, dst_path = pair_im_paths
    # print(src_path, dst_path)

    # load images
    src_img = load_images(src_path)
    dst_img = load_images(dst_path)

    # process smpl data
    # dict_keys(['pose', 'shape', 'cams', 'vertices'])
    cams = torch.tensor(smpl_data['cams']).float().cuda()
    verts = torch.tensor(smpl_data['verts']).float().cuda()

    src_cams = cams[0:1]
    src_verts = verts[0:1]
    src_img_gpu = torch.tensor(src_img[None], dtype=torch.float32).cuda()

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
        'from_face_index_map': src_fim[0][:, :, None].cpu().numpy(),
        'to_face_index_map': dst_fim[0][:, :, None].cpu().numpy(),
        'T': T[0].cpu().numpy(),
        'warp': warp_img[0].cpu().numpy(),
        'smpls': smpls
    }

    if visualizer is not None:
        visualizer.vis_named_img('src_fim', sample['from_face_index_map'][None, None, :, :, 0])
        visualizer.vis_named_img('dst_fim', sample['to_face_index_map'][None, None, :, :, 0])
        visualizer.vis_named_img('warp', sample['warp'][None, :, :, :])
        visualizer.vis_named_img('src_img', src_img[None, :, :, :])
        visualizer.vis_named_img('dst_img', dst_img[None, :, :, :])
        time.sleep(3)

    return sample


def sample_data(pairs_list, root_dir, out_dir, visual=True):
    if visual:
        visualizer = MotionImitationVisualizer(env='sample_data', ip='http://10.19.129.76', port=10086)
    else:
        visualizer = None

    all_smpl_data = dict()
    smpls_dir = os.path.join(root_dir, 'motion_transfer_smpl')
    render = SMPLRendererV2(image_size=IMG_SIZE, tex_size=3, has_front=True, fill_back=False)

    def get_cams_verts(vid_img_name):
        # ipdb.set_trace()
        v_name, img_name = os.path.split(vid_img_name)      # root_dir/v_name

        if v_name not in all_smpl_data:
            smpl_dir = os.path.join(smpls_dir, v_name, 'pose_shape.pkl')
            all_smpl_data[v_name] = load_pickle_file(smpl_dir)

        smpl_data = all_smpl_data[v_name]

        t = int(img_name.split('.')[0])
        cams = smpl_data['cams'][t]
        verts = smpl_data['vertices'][t]
        pose = smpl_data['pose'][t]
        shape = smpl_data['shape'][t]

        return cams, verts, pose, shape

    def sample_smpl_data(vid_img_pair_names):
        cams = []
        verts = []
        poses = []
        shapes = []
        for vid_img_name in vid_img_pair_names:
            cam_t, vert_t, pose_t, shape_t = get_cams_verts(vid_img_name)
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
            'shape': shapes,
        }
        return smpl_data

    length = len(pairs_list)
    out_path_template = os.path.join(out_dir, '{:0>8}.pkl')
    for i in tqdm(range(length)):
        vid_img_pair_names = pairs_list[i]
        smpl_data = sample_smpl_data(vid_img_pair_names)

        im_pairs = [os.path.join(root_dir, 'motion_transfer_HD', vid_img_name) for vid_img_name in vid_img_pair_names]
        sample = sample_pair_data(im_pairs, smpl_data, render, visualizer=visualizer)

        # for key, value in sample.items():
        #     print(key, value.shape)

        out_pkl_path = out_path_template.format(i)
        # print(out_pkl_path)
        write_pickle_file(out_pkl_path, sample)


if __name__ == '__main__':
    import ipdb

    args = parse_args()
    root_dir = args.root_dir
    out_dir = args.out_dir

    os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    for is_train in [True, False]:
        pairs_list = sample_all_pairs(root_dir, is_train=is_train)

        train_test_name = 'train_pairs_results' if is_train else 'val_pairs_results'
        out_dir_path = os.path.join(root_dir, train_test_name)

        sample_data(pairs_list, root_dir, out_dir_path, visual=False)

        print(1)



