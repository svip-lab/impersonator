import numpy as np
import cv2
import os
import sys

from models.models import ModelsFactory
from options.test_options import TestOptions
from utils.demo_visualizer import MotionImitationVisualizer


import ipdb


if __name__ == "__main__":

    opt = TestOptions().parse()

    # set imitator
    swapper = ModelsFactory.get_by_name(opt.model, opt)

    if opt.visual:
        visualizer = MotionImitationVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    src_path = opt.src_path
    tgt_path = opt.tgt_path

    swapper.swap_setup(src_path, tgt_path)

    if opt.post_tune:
        swapper.post_personalize(opt.output_dir, visualizer=visualizer)

    # if a->b
    swapper.swap(src_info=swapper.src_info, tgt_info=swapper.tsf_info, target_part=opt.swap_part, visualizer=visualizer)
    # else b->a
    # swapper.swap(src_info=swapper.tgt_info, tgt_info=swapper.src_info, target_part=opt.swap_part, visualizer=visualizer)

