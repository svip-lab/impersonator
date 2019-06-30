from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import progressbar
from collections import OrderedDict

import itertools

import nnutils.mesh as mesh
import data_utils.util as util
from data_utils.visual_helper import MotionTransferUVInpaintVisualizer
from nnutils.batch_smpl import batch_orth_proj_idrot
from runners_factory import RunnerFactory
from options.mt_tex_options import OPTIONS
from nnutils.nmr import SMPLTextureRenderer

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def image_one_frame(image_path, image_size, mask_path=None):

    image = cv2.imread(image_path)
    # image = cv2.resize(image, (300, 300))

    image_resize = cv2.resize(image, (image_size, image_size))
    image_resize = image_resize.astype(np.float32)

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image_size, image_size))
        mask = mask.astype(np.float32)
        mask /= 255.0
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0

        image_resize *= mask[:, :, np.newaxis]
    # rescale image from [0, 1] to [-1, 1] and convert to NCHW
    image_resize = util.transform_image(image_resize, "NCHW", normalize=True, rgb=True)
    image_orig = util.transform_image(image, "NCHW", normalize=True, rgb=True)

    image_resize = torch.Tensor(image_resize).float()
    image_orig = torch.Tensor(image_orig).float()

    return image_resize, image_orig


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = OPTIONS.gpus[0]

    with torch.no_grad():

        # define visualizer
        visualizer = MotionTransferUVInpaintVisualizer(OPTIONS.model_name,
                                                       ip=OPTIONS.visdom_ip, port=OPTIONS.visdom_port)

        # define trainer
        runner = RunnerFactory.make_trainer(OPTIONS)
        render = runner.generator.render

        render.anti_aliasing = True
        render.fill_back = True
        render.background_color = (-1, -1, -1)
        # render.set_ambient_light(lia=0.5, lid=0.5)

        fps = 1
        counter = 0

        video_dir = '/p300/human_pose/processed/motion_transfer'
        dst_dir = '/p300/human_pose/processed/motion_transfer_smpl'

        for user_name in sorted(os.listdir(video_dir)):
            user_path = os.path.join(video_dir, user_name)

            for tex_name in sorted(os.listdir(user_path)):
                tex_path = os.path.join(user_path, tex_name)

                for pos in sorted(os.listdir(tex_path)):
                    counter += 1

                    pose_path = os.path.join(tex_path, pos)

                    images_names = sorted(os.listdir(pose_path))
                    length = len(images_names)

                    dst_utp_dir = os.path.join(dst_dir, user_name, tex_name, pos)
                    if not os.path.exists(dst_utp_dir):
                        os.makedirs(dst_utp_dir)

                    cam_list = []
                    pose_list = []
                    shape_list = []
                    vertices_list = []

                    for t in range(0, length, fps):
                        image_name = images_names[t]
                        image_path = os.path.join(pose_path, image_name)

                        resized_image, orig_image = image_one_frame(image_path, image_size=224)

                        resized_image = resized_image.cuda()[None, ...]

                        src_info = runner.inference_pose_shape(resized_image)

                        cams = src_info['cam'].cpu()
                        pose = src_info['pose'].cpu()
                        shape = src_info['shape'].cpu()
                        vertices = src_info['vertices'].cpu()

                        cam_list.append(cams)
                        pose_list.append(pose)
                        shape_list.append(shape)
                        vertices_list.append(vertices)
                        print(counter, t, length)

                    util.write_pickle_file(os.path.join(dst_utp_dir, 'pose_shape.pkl'), {
                        'pose': np.concatenate(pose_list, axis=0),
                        'shape': np.concatenate(shape_list, axis=0),
                        'cams': np.concatenate(cam_list, axis=0),
                        'vertices': np.concatenate(vertices_list, axis=0)
                    })


if __name__ == '__main__':
    main()

