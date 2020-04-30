import unittest
import torch
import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm

from his_evaluators.protocols.iPER import IPERProtocol
from his_evaluators.protocols.MotionSynthetic import MotionSyntheticProtocol


class ProtocolTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.iPER_Protocol = IPERProtocol(data_dir="/p300/tpami/iPER")
        cls.MS_Protocol = MotionSyntheticProtocol(data_dir="/p300/tpami/datasets/motionSynthetic")

    def test_01_MS_Protocol(self):
        for vid_info in self.MS_Protocol:
            print(vid_info)


if __name__ == '__main__':
    unittest.main()



