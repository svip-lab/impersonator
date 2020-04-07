import unittest
import torch
import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm


from his_evaluators.evaluators.base import PairedMetricRunner, UnpairedMetricRunner


IMAGE_SIZE = 512
DEVICE = torch.device("cuda:0")
TEMPLATE_DIR = "./template_dir"


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


def make_template_files(number=100):
    global TEMPLATE_DIR

    TEMPLATE_DIR = mkdir(TEMPLATE_DIR)
    file_paths = []

    print("preparing {} samples and saving them into the template directory {}.".format(number, TEMPLATE_DIR))
    for i in tqdm(range(number)):
        pred = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3)
        pred *= 255
        pred = pred.astype(np.uint8)

        ref = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3)
        ref *= 255
        ref = ref.astype(np.uint8)

        pred_path = os.path.join(TEMPLATE_DIR, "{:0>8}.png".format(i))
        ref_path = os.path.join(TEMPLATE_DIR, "{:0>8}.png".format(i))

        cv2.imwrite(pred_path, pred)
        cv2.imwrite(ref_path, ref)

        file_paths.append((pred_path, ref_path))

    return file_paths


class MetricRunnerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.paired_metric_runner = PairedMetricRunner(
            metric_types=("ssim", "psnr", "lps"), device=DEVICE
        )

        cls.unpaired_metric_runner = UnpairedMetricRunner(
            metric_types=("is", "fid", "PCB-CS-reid", "PCB-freid", "OS-CS-reid", "OS-freid"), device=DEVICE
        )

        # cls.unpaired_metric_runner = UnpairedMetricRunner(
        #     metric_types=("is", "fid", "PCB-CS-reid", "PCB-freid"), device=DEVICE
        # )

        cls.file_paths = make_template_files(number=100)

    def test_01_paired_runner(self):
        self.paired_metric_runner.evaluate(file_paths=self.file_paths, image_size=IMAGE_SIZE, batch_size=32)

    def test_02_unpaired_runner(self):
        self.unpaired_metric_runner.evaluate(file_paths=self.file_paths, image_size=IMAGE_SIZE, batch_size=32)

    @classmethod
    def tearDownClass(cls):
        global TEMPLATE_DIR

        clean_dir(TEMPLATE_DIR)

        print("clean the template directory {}".format(TEMPLATE_DIR))


if __name__ == '__main__':
    unittest.main()
