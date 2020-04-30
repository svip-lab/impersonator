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

        # ref = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3)
        ref = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3))
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
            metric_types=("ssim", "psnr", "lps", "OS-CS-reid", "face-CS"), device=DEVICE
        )

        cls.unpaired_metric_runner = UnpairedMetricRunner(
            metric_types=("is", "fid", "OS-CS-reid", "OS-freid", "face-CS", "face-FD", "SSPE"), device=DEVICE
        )

        # cls.unpaired_metric_runner = UnpairedMetricRunner(
        #     metric_types=("is", "fid", "PCB-CS-reid", "PCB-freid", "OS-CS-reid", "OS-freid"), device=DEVICE
        # )

        # cls.unpaired_metric_runner = UnpairedMetricRunner(
        #     metric_types=("is", "fid", "PCB-CS-reid", "PCB-freid"), device=DEVICE
        # )

        cls.file_paths = make_template_files(number=100)

    def test_01_paired_runner(self):
        self.paired_metric_runner.evaluate(file_paths=self.file_paths, image_size=IMAGE_SIZE, batch_size=16)

    def test_02_unpaired_runner(self):
        self.unpaired_metric_runner.evaluate(file_paths=self.file_paths, image_size=IMAGE_SIZE, batch_size=16)

    def test_04_SSPE(self):
        from his_evaluators.evaluators.base import PairedEvaluationDataset, build_data_loader

        sample_dir = "./data"
        img_names = [
            "pred_00000000.jpg",
            "pred_00000114.jpg",
            "pred_00000175.jpg",
            "pred_00000423.jpg",
        ]
        all_img_paths = []
        for name in img_names:
            img_path = os.path.join(sample_dir, name)
            all_img_paths.append(img_path)

        dataset = PairedEvaluationDataset(list(zip(all_img_paths, all_img_paths)), image_size=512)
        dataloader = build_data_loader(dataset, batch_size=4)
        sample = next(iter(dataloader))

        pred = sample["pred"]
        ref = sample["ref"]
        ids = np.random.permutation(len(ref))
        ref = ref[ids]
        sspe = self.unpaired_metric_runner.metric_dict["SSPE"].calculate_score(pred, ref)
        print("sspe = {}".format(sspe))

    def test_03_face_detector(self):
        from his_evaluators.evaluators.base import PairedEvaluationDataset, build_data_loader

        sample_dir = "./data"
        img_names = [
            "pred_00000000.jpg",
            "pred_00000114.jpg",
            "pred_00000175.jpg",
            "pred_00000423.jpg",
        ]
        all_img_paths = []
        for name in img_names:
            img_path = os.path.join(sample_dir, name)
            all_img_paths.append(img_path)

        dataset = PairedEvaluationDataset(list(zip(all_img_paths, all_img_paths)), image_size=512)
        dataloader = build_data_loader(dataset, batch_size=4)
        sample = next(iter(dataloader))

        ref = sample["ref"]
        face_cropped, valid_ids = self.paired_metric_runner.metric_dict["face-CS"].detect_face(ref)
        print(valid_ids)
        has_detected_face = face_cropped[valid_ids]
        print(face_cropped.shape, has_detected_face.shape)
        face_cropped = face_cropped.cpu().numpy()
        for i, face in enumerate(face_cropped):
            face = (face + 1) / 2 * 255
            face = face.astype(np.uint8, copy=False)
            face = np.transpose(face, (1, 2, 0))
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

            print(face.shape, face.max(), face.min())
            cv2.imwrite(os.path.join(sample_dir, "face_{}".format(img_names[i])), face)

        print(face_cropped.shape, face_cropped.max(), face_cropped.min())

    @classmethod
    def tearDownClass(cls):
        global TEMPLATE_DIR

        clean_dir(TEMPLATE_DIR)

        print("clean the template directory {}".format(TEMPLATE_DIR))


if __name__ == '__main__':
    unittest.main()
