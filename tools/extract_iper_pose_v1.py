import os
import argparse

# if not it will raise error
from openpose import pyopenpose as op
from tools.datum_wrapper import *
from utils.visualizer import MotionImitationVisualizer
import utils.pose_estimator as pose_estimator

import ipdb

# set gpu
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",
                        default="/public/liuwen/p300/human_pose/processed/motion_transfer_HD",
                        help="root video dir")
    parser.add_argument("--out_dir",
                        default="/public/liuwen/p300/human_pose/processed/motion_transfer_openpose/debug",
                        help="json output dir")
    parser.add_argument('--visual', action='store_true', help='using visualizer or not.')
    args = parser.parse_known_args()

    params = dict()

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    return args, params


def parse_dir(root_dir):

    print('parsing root dir {}'.format(root_dir))

    video_infos = dict()

    for p_id in os.listdir(root_dir):
        p_dir = os.path.join(root_dir, p_id)

        for c_id in os.listdir(p_dir):
            p_c_dir = os.path.join(p_dir, c_id)

            for a_id in os.listdir(p_c_dir):
                p_c_a_dir = os.path.join(p_c_dir, a_id)
                images = os.listdir(p_c_a_dir)
                images.sort()
                video_infos[p_id + '/' + c_id + '/' + a_id] = images

    return video_infos


# parse args
args, params = parse_args()

# visdom visualizer
visualizer = MotionImitationVisualizer(env='openpose', ip='10.19.125.183', port=10087)

pose_estimator.config()
src_img_path = '/public/liuwen/p300/human_pose/processed/motion_transfer_HD/001/1/1/0000.jpg'

for i in range(10):
    results = pose_estimator.estimate(src_img_path)
    print(i, src_img_path, results['pose_keypoints_2d'].shape)

# # dataset info
# root_dir = args[0].root_dir
# out_dir = args[0].out_dir
# video_infos = parse_dir(root_dir)
# number_is_not_one_list = []
#
# for v_id, video_name in enumerate(video_infos):
#     video_dir = os.path.join(root_dir, video_name)
#     pkl_dir = os.path.join(out_dir, video_name)
#
#     if not os.path.exists(pkl_dir):
#         os.makedirs(pkl_dir)
#
#     pkl_path = os.path.join(pkl_dir, 'openpose_outs.pkl')
#     if os.path.exists(pkl_path):
#         print('{} is exist.'.format(pkl_path))
#         continue
#
#     images = video_infos[video_name]
#     video_outputs = get_default_info()
#
#     for i, image_name in enumerate(images):
#         image_path = os.path.join(video_dir, image_name)
#
#         ipdb.set_trace()
#         result_info = pose_estimator.estimate(image_path)
#         print(i, image_path, result_info['pose_keypoints_2d'].shape)
#
#         number = write_datum_sample(datum_info=video_outputs, datum=result_info,
#                                     only_body=True, is_coco=False, is_first=(i == 0))
#
#         if number != 1:
#             number_is_not_one_list.append((image_path, number))
#
#         if args[0].visual:
#             cvOutputData = pose_estimator.datum.cvOutputData[None, ...]
#             visualizer.vis_named_img('cvOutputData', cvOutputData, transpose=True)
#             print(i, image_path, cvOutputData.shape)
#         else:
#             print(i, image_path)
#
#     write_datum_pkl(pkl_path, video_outputs)
#
# write_fail_case('fail_case_whole.txt', fail_case_list=number_is_not_one_list)
#
# pose_estimator.close()

