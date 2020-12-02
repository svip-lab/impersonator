import os
import torch
import numpy as np
from typing import Dict, Any, List
# evaluations
from his_evaluators import MotionImitationModel, IPERMotionImitationEvaluator

from models.imitator import Imitator
from options.test_options import TestOptions
from utils.visdom_visualizer import VisdomVisualizer
from run_imitator import adaptive_personalize
from utils import cv_utils


class LWGEvaluatorModel(MotionImitationModel):

    def __init__(self, opt, output_dir):
        super().__init__(output_dir)

        self.opt = opt

        if self.opt.ip:
            visualizer = VisdomVisualizer(env=self.opt.name, ip=self.opt.ip, port=self.opt.port)
        else:
            visualizer = None

        self.visualizer = visualizer
        self.model = None

    def imitate(self, src_infos: Dict[str, Any], ref_infos: Dict[str, Any]) -> List[str]:
        """
            Running the motion imitation of the self.model, based on the source information with respect to the
            provided reference information. It returns the full paths of synthesized images.
        Args:
            src_infos (dict): the source information contains:
                --images (list of str): the list of full paths of source images (the length is 1)
                --smpls (np.ndarray): (length of images, 85)
                --kps (np.ndarray): (length of images, 19, 2)
            ref_infos (dict): the reference information contains:
                --images (list of str): the list of full paths of reference images.
                --smpls (np.ndarray): (length of images, 85)
                --kps (np.ndarray): (length of images, 19, 2)
                --self_imitation (bool): the flag indicates whether it is self-imitation or not.

        Returns:
            preds_files (list of str): full paths of synthesized images with respects to the images in ref_infos.
        """

        tgt_paths = ref_infos["images"]
        tgt_smpls = np.copy(ref_infos["smpls"])
        self_imitation = ref_infos["self_imitation"]
        if self_imitation:
            cam_strategy = "copy"
            out_dir = self.si_out_dir
            count = self.num_preds_si
            self.num_preds_si += len(tgt_paths)
        else:
            cam_strategy = "smooth"
            out_dir = self.ci_out_dir
            count = self.num_preds_ci
            self.num_preds_ci += len(tgt_paths)
        outputs = self.model.inference(tgt_paths, tgt_smpls=tgt_smpls, cam_strategy=cam_strategy,
                                       visualizer=None, verbose=True)

        all_preds_files = []
        for i, preds in enumerate(outputs):
            filename = "{:0>8}.jpg".format(count)
            pred_file = os.path.join(out_dir, 'pred_' + filename)
            count += 1

            cv_utils.save_cv2_img(preds, pred_file, normalize=True)
            all_preds_files.append(pred_file)

        return all_preds_files

    def build_model(self):
        """
            You must define your model in this function, including define the graph and allocate GPU.
            This function will be called in @see `MotionImitationRunnerProcessor.run()`.
        Returns:
            None
        """
        # set imitator
        self.model = Imitator(self.opt)

    def personalization(self, src_infos):
        """
            some task/method specific data pre-processing or others.
        Args:
            src_infos (dict): the source information contains:
                --images (list of str): the list of full paths of source images (the length is 1)
                --smpls (np.ndarray): (length of images, 85)
                --kps (np.ndarray): (length of images, 19, 2)

        Returns:
            processed_src_infos (dict): the source information contains:
                --images (list of str): the list of full paths of source images (the length is 1)
                --smpls (np.ndarray): (length of images, 85)
                --kps (np.ndarray): (length of images, 19, 2)
                ...
        """

        # 1. load the pretrain model
        self.model._load_params(self.model.generator, self.opt.load_path)
        self.opt.src_path = src_infos["images"][0]

        # 2. post personalization
        if self.opt.post_tune:
            adaptive_personalize(self.opt, self.model, self.visualizer)

        self.model.personalize(self.opt.src_path, src_smpl=np.copy(src_infos["smpls"][0]), visualizer=None)
        processed_src_infos = src_infos
        return processed_src_infos

    def terminate(self):
        """
            Close the model session, like if the model is based on TensorFlow, it needs to call sess.close() to
            dealloc the resources.
        Returns:

        """
        pass


if __name__ == "__main__":
    opt = TestOptions().parse()

    model = LWGEvaluatorModel(opt, output_dir=opt.output_dir)
    # iPER_MI_evaluator = IPERMotionImitationEvaluator(dataset="iPER", data_dir=opt.data_dir)
    iPER_MI_evaluator = IPERMotionImitationEvaluator(dataset="iPER_ICCV", data_dir=opt.data_dir)

    iPER_MI_evaluator.evaluate(
        model=model,
        image_size=opt.image_size,
        pair_types=("ssim", "psnr", "lps", "face-CS", "OS-CS-reid"),
        unpair_types=("is", "fid", "OS-CS-reid", "OS-freid", "face-CS", "face-FD"),
        device=torch.device("cuda:0")
    )

