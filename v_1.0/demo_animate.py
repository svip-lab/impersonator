import numpy as np
import cv2
import os
import glob

from models.models import ModelsFactory
from options.test_options import TestOptions
from utils.demo_visualizer import MotionImitationVisualizer

import ipdb


def get_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def im2disk(img, out_path):
    img += 1.0
    img /= 2.0
    img *= 255

    img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img)


def save_image(src_path, tgt_path, image_size=256):
    img = cv2.imread(src_path)
    img = cv2.resize(img, (image_size, image_size))
    cv2.imwrite(tgt_path, img)


def cvt_name(video_name):
    return '_'.join(video_name.split('/'))


def get_img_name(file_path):
    filename = os.path.split(file_path)[-1]
    return filename


def main(imitator, evaluator, output_dir='', visualizer=None):
    # 1. for-loop source info
    video_names = evaluator.dataset.video_names
    pair_info = evaluator.dataset.pair_info

    for v_name in video_names:
        src_info = pair_info['source'][v_name]

        src_outdir = get_dir(os.path.join(output_dir, cvt_name(v_name)))

        # 2. for-loop source image
        for i, src_path in enumerate(src_info['images']):
            src_smpl = src_info['smpls'][i] if src_info['smpls'] is not None else None
            src_img_name = os.path.split(src_path)[-1]
            src_name = src_img_name.split('.')[0]

            # personalize
            src_save_path = os.path.join(src_outdir, src_img_name) if output_dir else  ''
            imitator.personalize(src_path, src_smpl, src_save_path)

            if visualizer is not None:
                visualizer.vis_named_img('bg', imitator.src_info['bg'])

            # 3. cross(self) - imitator (for-loop)
            used_a = False
            for cross_v_name in video_names:
                cross_mi_info = pair_info['cross_mi'][cross_v_name]

                if cross_mi_info['a_pose'] and used_a:
                    continue

                if cross_mi_info['a_pose']:
                    used_a = True

                tgt_outdir = get_dir(os.path.join(src_outdir, src_name, cvt_name(cross_v_name))) if output_dir else ''

                is_self = v_name == cross_v_name
                imitator.imitate(tgt_paths=cross_mi_info['images'], tgt_smpls=cross_mi_info['smpls'],
                                 output_dir=tgt_outdir, visualizer=visualizer)


if __name__ == "__main__":

    opt = TestOptions().parse()

    # set animator
    animator = ModelsFactory.get_by_name(opt.model, opt)

    if opt.visual:
        visualizer = MotionImitationVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    src_path = opt.src_path
    ref_path = opt.ref_path
    tgt_path = opt.tgt_path

    animator.animate_setup(src_path, ref_path)

    imgs_paths = []
    if os.path.isdir(tgt_path):
        imgs_paths = glob.glob(os.path.join(tgt_path, '*.jpg'))
        imgs_paths.sort()
    else:
        imgs_paths = [tgt_path]

    animator.animate(img_paths=imgs_paths, cam_strategy=opt.cam_strategy, output_dir='', visualizer=visualizer)

