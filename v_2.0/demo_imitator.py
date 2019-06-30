import torch
import numpy as np
import cv2
import os

import networks
from models.models import ModelsFactory
from options.test_options import TestOptions
from utils.visualizer.demo_visualizer import MotionImitationVisualizer
from utils.util import load_pickle_file


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

    imitator.personalize(src_path, visualizer=visualizer)
    imitator.inference([tgt_path], visualizer=visualizer, cam_strategy=opt.cam_strategy)

