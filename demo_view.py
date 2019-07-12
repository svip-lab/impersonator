import numpy as np
import cv2
import os
import torch
import sys

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


def parse_view_params(view_params):
    """
    :param view_params: R=xxx,xxx,xxx/t=xxx,xxx,xxx
    :return:
        -R: np.ndarray, (3,)
        -t: np.ndarray, (3,)
    """

    params = dict()
    for segment in view_params.split('/'):
        # R=xxx,xxx,xxx -> (name, xxx,xxx,xxx)
        name, params_str = segment.split('=')

        vals = [float(val) for val in params_str.split(',')]

        params[name] = np.array(vals, dtype=np.float32)

    params['R'] = params['R'] / 180 * np.pi
    return params


if __name__ == "__main__":

    opt = TestOptions().parse()

    # set imitator
    viewer = ModelsFactory.get_by_name(opt.model, opt)

    if opt.visual:
        visualizer = MotionImitationVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    src_path = opt.src_path
    view_params = opt.view_params
    params = parse_view_params(view_params)

    viewer.setup(src_path)

    # length = 30
    # delta = 360 / length
    # pred_outs = []
    # for i in range(length):
    #     params['R'][0] = 10 / 180 * np.pi
    #     params['R'][1] = delta * i / 180.0 * np.pi
    #     params['R'][2] = 10 / 180 * np.pi
    #
    #     print(i, params['R'])
    #     preds = viewer.view(params['R'], params['t'], visualizer=None, name=str(i))
    #     pred_outs.append(preds)
    #
    # pred_outs = torch.cat(pred_outs, dim=0)
    # visualizer.vis_named_img('preds', pred_outs)

    def process(x):
        return float(x) / 180 * np.pi

    # while True:
    #     inputs = input('input thetas: ')
    #     if inputs == 'q':
    #         break
    #     thetas = list(map(process, inputs.split(' ')))
    #
    #     preds = viewer.view(thetas, params['t'], visualizer=None, name='0')
    #     visualizer.vis_named_img('pred', preds)

    src_names = ['001_9_1/000.jpg', '001_11_1/0000.jpg',
                 '011_1_1/0000.jpg', '024_7_1/000.jpg']



