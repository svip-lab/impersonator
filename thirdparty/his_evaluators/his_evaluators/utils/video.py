# -*- coding: utf-8 -*-
# @Time    : 2019-08-02 18:31
# @Author  : Wen Liu
# @Email   : liuwen@shanghaitech.edu.cn

import os
import cv2
import glob
import shutil
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import numpy as np
import subprocess


def auto_unzip_fun(x, f):
    return f(*x)


def make_video(output_mp4_path, img_path_list, save_frames_dir=None, fps=24):
    """
    output_path is the final mp4 name
    img_dir is where the images to make into video are saved.
    """

    first_img = cv2.imread(img_path_list[0])
    h, w = first_img.shape[:2]

    pool_size = 40
    tmp_avi_video_path = '%s.avi' % output_mp4_path
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (w, h))
    args_list = [(img_path,) for img_path in img_path_list]
    with Pool(pool_size) as p:
        for img in tqdm(p.imap(partial(auto_unzip_fun, f=cv2.imread), args_list), total=len(args_list)):
            videoWriter.write(img)
    videoWriter.release()

    if save_frames_dir:
        for i, img_path in enumerate(img_path_list):
            shutil.copy(img_path, '%s/%.8d.jpg' % (save_frames_dir, i))

    os.system("ffmpeg -y -i %s -vcodec h264 %s > /dev/null 2>&1" % (tmp_avi_video_path, output_mp4_path))
    os.system("rm %s" % tmp_avi_video_path)


def fuse_image(img_path_list, row_num, col_num):
    assert len(img_path_list) == row_num * col_num

    img_list = [cv2.imread(img_path) for img_path in img_path_list]

    row_imgs = []
    for i in range(row_num):
        col_imgs = img_list[i * col_num: (i + 1) * col_num]
        col_img = np.concatenate(col_imgs, axis=1)
        row_imgs.append(col_img)

    fused_img = np.concatenate(row_imgs, axis=0)
    return fused_img


def fuse_video(video_frames_path_list, output_mp4_path, row_num, col_num, fps=24):
    assert len(video_frames_path_list) == row_num * col_num

    frame_num = len(video_frames_path_list[0])
    first_img = cv2.imread(video_frames_path_list[0][0])
    h, w = first_img.shape[:2]
    fused_h, fused_w = h * row_num, w * col_num

    args_list = []
    for frame_idx in range(frame_num):
        fused_frame_path_list = [video_frames[frame_idx] for video_frames in video_frames_path_list]
        args_list.append((fused_frame_path_list, row_num, col_num))

    pool_size = 40
    tmp_avi_video_path = '%s.avi' % output_mp4_path
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # for args in args_list:
    #     fuse_image(*args)
    # exit()

    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (fused_w, fused_h))
    with Pool(pool_size) as p:
        for img in tqdm(p.imap(partial(auto_unzip_fun, f=fuse_image), args_list), total=len(args_list)):
            videoWriter.write(img)
    videoWriter.release()

    os.system("ffmpeg -y -i %s -vcodec h264 %s > /dev/null 2>&1" % (tmp_avi_video_path, output_mp4_path))
    os.system("rm %s" % (tmp_avi_video_path))


def merge(src_img, ref_img_path, out_img_path, pad):
    h, w = src_img.shape[:2]
    image_size = h

    ref_img = cv2.imread(ref_img_path)
    out_img = cv2.imread(out_img_path)

    if ref_img.shape[0] != image_size and ref_img.shape[1] != image_size:
        ref_img = cv2.resize(ref_img, (image_size, image_size))

    if out_img.shape[0] != image_size and out_img.shape[1] != image_size:
        out_img = cv2.resize(out_img, (image_size, image_size))

    # print(src_img.shape, ref_img.shape, out_img.shape)
    merge_img = np.concatenate([src_img, pad, ref_img, pad, out_img], axis=1)

    return merge_img


def load_image(image_path, image_size=512):
    """

    Args:
        image_path (str):
        image_size (int):

    Returns:
        image (np.ndarray): (image_size, image_size, 3), BGR channel space, in the range of [0, 255], np.uint8.
    """

    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))

    return image


def fuse_one_image(img_paths, image_size):
    return load_image(img_paths[0], image_size)


def fuse_two_images(img_paths, image_size):
    """

    Args:
        img_paths (list of str):
        image_size (int):

    Returns:
        fuse_img (np.ndarray): (image_size // 2, image_size, 3), BGR channel space, in the range of [0, 255], np.uint8.
    """

    img_size = image_size // 2

    img_1 = load_image(img_paths[0], img_size)
    img_2 = load_image(img_paths[1], img_size)

    fuse_img = np.concatenate([img_1, img_2], axis=0)

    return fuse_img


def fuse_four_images(img_paths, image_size):
    """

    Args:
        img_paths (list of str):
        image_size (int):

    Returns:
        fuse_img (np.ndarray): (image_size, image_size, 3), BGR channel space, in the range of [0, 255], np.uint8.
    """

    fuse_img_1 = fuse_two_images(img_paths[0:2], image_size)
    fuse_img_2 = fuse_two_images(img_paths[2:4], image_size)

    fuse_img = np.concatenate([fuse_img_1, fuse_img_2], axis=1)
    return fuse_img


def fuse_eight_images(img_paths, image_size):
    """

    Args:
        img_paths (list of str):
        image_size (int):

    Returns:
        fuse_img (np.ndarray): (image_size // 2, image_size, 3), BGR channel space, in the range of [0, 255], np.uint8.
    """

    fuse_img_1 = fuse_two_images(img_paths[0:4], image_size // 2)
    fuse_img_2 = fuse_two_images(img_paths[4:8], image_size // 2)

    fuse_img = np.concatenate([fuse_img_1, fuse_img_2], axis=0)
    return fuse_img


def fuse_source(all_src_img_paths, image_size=512):
    """

    Args:
        all_src_img_paths (list of str): the list of source image paths, currently it only supports, 1, 2, 4, 8 number
            of source images.

        image_size (int): the final image resolution, (image_size, image_size, 3)

    Returns:
        fuse_img (np.ndarray): (image_size, image_size, 3), BGR channel space, in the range of [0, 255], np.uint8.
    """

    ns = len(all_src_img_paths)

    # TODO, currently it only supports, 1, 2, 4, 8 number of source images.
    assert ns in [1, 2, 4, 8], "{} must be in [1, 2, 4, 8], currently it only supports, " \
                               "1, 2, 4, 8 number of source images."

    if ns == 1:
        fuse_img = load_image(all_src_img_paths[0], image_size)

    elif ns == 2:
        fuse_img = fuse_two_images(all_src_img_paths, image_size)

    elif ns == 4:
        fuse_img = fuse_four_images(all_src_img_paths, image_size)

    elif ns == 8:
        fuse_img = fuse_eight_images(all_src_img_paths, image_size)

    else:
        raise ValueError("{} must be in [1, 2, 4, 8], currently it only supports, "
                         "1, 2, 4, 8 number of source images.")

    return fuse_img


def fuse_source_reference_output(output_mp4_path, src_img_paths, ref_img_paths, out_img_paths,
                                 image_size=512, pad=10, fps=25):
    total = len(ref_img_paths)
    assert total == len(out_img_paths), "{} != {}".format(total, len(out_img_paths))

    fused_src_img = fuse_source(src_img_paths, image_size)
    pad_region = np.zeros((image_size, pad, 3), dtype=np.uint8)

    pool_size = min(15, os.cpu_count())
    tmp_avi_video_path = '%s.avi' % output_mp4_path
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    W = fused_src_img.shape[1] + (image_size + pad) * 2
    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (W, image_size))

    with ProcessPoolExecutor(pool_size) as pool:
        for img in tqdm(pool.map(merge, [fused_src_img] * total,
                                 ref_img_paths, out_img_paths, [pad_region] * total)):
            videoWriter.write(img)

    videoWriter.release()

    os.system("ffmpeg -y -i %s -vcodec h264 %s > /dev/null 2>&1" % (tmp_avi_video_path, output_mp4_path))
    os.system("rm %s" % tmp_avi_video_path)
