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
    imitator = ModelsFactory.get_by_name(opt.model, opt)

    if opt.visual:
        visualizer = MotionImitationVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    src_path = opt.src_path
    tgt_path = opt.tgt_path

    # 1. imitation
    imitator.personalize(src_path, visualizer=visualizer)
    outputs = imitator.imitate([tgt_path], visualizer=visualizer)

    # 2. render the object face
    src_info = imitator.src_info
    tsf_info = imitator.tsf_info



