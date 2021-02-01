from abc import ABC
import torch
from multiprocessing import Process, Manager
from tqdm import tqdm
from typing import List, Dict, Any
import os
import glob

from his_evaluators.metrics import TYPES_QUALITIES

from .base import PairedMetricRunner, Evaluator
from ..utils.io import mkdir
from ..utils.video import fuse_source_reference_output


class AppearanceTransferModel(object):

    def __init__(self, output_dir):
        """

        Args:
            output_dir (str):
        """

        self.output_dir = mkdir(output_dir)
        self.si_out_dir = mkdir(os.path.join(output_dir, "self_imitation"))
        self.num_preds_si = 0

    def swap(self, src_infos: Dict[str, Any], app_infos: Dict[str, Any], motion_infos: Dict[str, Any]) -> List[str]:
        """
            Running the appearance transfer of the self.model, based on the source information with respect to the
            provided reference information. It returns the full paths of synthesized images.
        Args:
            src_infos (dict):
                --s_n (str): dict map of from number of source (s_n) to source images
                --name (str): the video name of source (`001/9/1`)
                --formated_name (str): the formated video name of source (`001_9_1`);
                --vid_path (str):
                --images (list of str):
                --smpls (np.ndarray or None):
                --kps (np.ndarray or None):

            app_infos (dict):
                --s_n (str): dict map of from number of source (s_n) to source images
                --name (str): the video name of source (`001/9/1`)
                --formated_name (str): the formated video name of source (`001_9_1`);
                --vid_path (str):
                --images (list of str):
                --smpls (np.ndarray or None):
                --kps (np.ndarray or None):

            motion_infos (dict):
                --name (str): the name of reference video,
                --formated_name (str): the formated reference name,
                --vid_path (str): the full video path,
                --images (list of str): the full images paths,
                --smpls (np.ndarray or None):
                --kps (np.ndarray or None):

        Returns:
            preds_files (list of str): full paths of synthesized images with respects to the images in ref_infos.
        """
        raise NotImplementedError

    def build_model(self):
        """
            You must define your model in this function, including define the graph and allocate GPU.
            This function will be called in @see `MotionImitationRunnerProcessor.run()`.
        Returns:
            None
        """
        raise NotImplementedError

    def terminate(self):
        """
            Close the model session, like if the model is based on TensorFlow, it needs to call sess.close() to
            dealloc the resources.
        Returns:

        """
        raise NotImplementedError

    def personalization(self, src_infos, app_infos):
        """
            some task/method specific data pre-processing or others.
        Args:
            src_infos (dict): the source information contains:
                --images (list of str): the list of full paths of source images (the length is 1)
                --smpls (np.ndarray): (length of images, 85)
                --kps (np.ndarray): (length of images, 19, 2)

            app_infos (dict): the appearance information contains:
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

        processed_src_infos = src_infos
        return processed_src_infos


class AppearanceTransferRunnerProcessor(Process):
    def __init__(self, model, protocols, return_dict: Manager):
        """
            The processor of running motion imitation models.
        Args:
            model (AppearanceTransferModel):
            protocols (Protocols):
            return_dict (Manager)
        """
        self.model = model
        self.protocols = protocols
        self.return_dict = return_dict

        super().__init__()

    def run(self):
        self.model.build_model()

        # si means self-imitation
        all_si_preds_ref_file_list = []

        for vid_info in tqdm(self.protocols):
            # source information, contains {"images", "smpls", "kps"},
            # here "images" are the list of full paths of source images (the length is 1)
            src_infos = vid_info["source"]
            app_infos = vid_info["appearance"]
            motion_infos = vid_info["motion"]

            # print(src_infos["name"], app_infos["name"])

            # run personalization
            personalized_infos = self.model.personalization(src_infos, app_infos)

            # si means (self-imitation)
            si_pred_files = self.model.swap(personalized_infos, app_infos, motion_infos)

            si_pred_ref_files = self.post_format_metric_file_list(si_pred_files, motion_infos["images"])

            all_si_preds_ref_file_list.extend(si_pred_ref_files)

            # break
        self.return_dict["all_si_preds_ref_file_list"] = all_si_preds_ref_file_list

    def terminate(self) -> None:
        self.model.terminate()

    def post_format_metric_file_list(self, si_preds_files, si_ref_files):
        """
            make [(si_pred, si_ref), ...], ...]
        Args:
            si_preds_files:
            si_ref_files:

        Returns:
            si_preds_ref_files:
        """
        si_preds_ref_files = list(zip(si_preds_files, si_ref_files))

        return si_preds_ref_files


class AppearanceTransferEvaluator(Evaluator, ABC):
    def __init__(self, dataset, data_dir):
        super().__init__(dataset, data_dir)

        # please call `build_metrics` to instantiate these two runners.
        self.paired_metrics_runner = None

    def reset_dataset(self, dataset, data_dir):
        super().__init__(dataset, data_dir)

    def build_metrics(
        self,
        pair_types=("ssim", "psnr", "lps"),
        device=torch.device("cpu")
    ):
        paired_metrics_runner = PairedMetricRunner(metric_types=pair_types, device=device)
        self.paired_metrics_runner = paired_metrics_runner

    def run_metrics(self, self_imitation_files, image_size=512):
        assert self.paired_metrics_runner is not None, \
            "please call `build_metrics(pair_types)` to instantiate metrics runners before calling this function."

        si_results = self.paired_metrics_runner.evaluate(self_imitation_files, image_size)

        return si_results

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def run_inference(self, *args, **kwargs):
        raise NotImplementedError


class IPERAppearanceTransferEvaluator(AppearanceTransferEvaluator):

    def __init__(self, data_dir, dataset="iPER_Appearance_Transfer"):
        """

        Args:
            data_dir (str): the data directory
            dataset (str): the dataset name, it can be
                --iPER_Appearance_Transfer: the iPER dataset;
        """
        super().__init__(dataset=dataset, data_dir=data_dir)

    def run_inference(self, model, src_infos, app_infos, motion_infos):
        """
        Args:
            model (AppearanceTransferModel): the model object, it must define and implements the function
                            `swap(src_infos, app_infos, motion_infos) -> List[str]`
            src_infos (dict):
                --s_n (str): dict map of from number of source (s_n) to source images
                --name (str): the video name of source (`001/9/1`)
                --formated_name (str): the formated video name of source (`001_9_1`);
                --vid_path (str):
                --images (list of str):
                --smpls (np.ndarray or None):
                --kps (np.ndarray or None):

            app_infos (dict):
                --s_n (str): dict map of from number of source (s_n) to source images
                --name (str): the video name of source (`001/9/1`)
                --formated_name (str): the formated video name of source (`001_9_1`);
                --vid_path (str):
                --images (list of str):
                --smpls (np.ndarray or None):
                --kps (np.ndarray or None):

            motion_infos (dict):
                --name (str): the name of reference video,
                --formated_name (str): the formated reference name,
                --vid_path (str): the full video path,
                --images (list of str): the full images paths,
                --smpls (np.ndarray or None):
                --kps (np.ndarray or None):

        Returns:
            file_paths (list of str): [pred_img_path_0, pred_img_path_1, ..., pred_img_path_i, ..., pred_img_path_n)]

        """

        assert hasattr(model, "swap"), '{} must implement swap(src_infos, app_infos, motion_infos) -> List[str]'

        file_paths = model.swap(src_infos, app_infos, motion_infos)

        return file_paths

    def evaluate(self, model, num_sources=1, image_size=512,
                 pair_types=("ssim", "psnr", "lps"), swap_parts=("head", "body"),
                 device=torch.device("cpu")):
        # 1. setup protocols
        self.protocols.setup(num_sources=num_sources, load_smpls=True, load_kps=True)

        # 2. declare runner processor for inference
        return_dict = Manager().dict({})
        runner = AppearanceTransferRunnerProcessor(model, self.protocols, return_dict)
        runner.start()
        runner.join()

        del model

        all_si_preds_ref_file_list = return_dict["all_si_preds_ref_file_list"]

        # run metrics
        self.build_metrics(pair_types, device)
        si_results = self.run_metrics(all_si_preds_ref_file_list, image_size)

        return si_results

    def preprocess(self, *args, **kwargs):
        pass

    def save_results(self, out_path, si_results):
        """
            save the the results into the out_path.
        Args:
            out_path (str): the full path to save the results.
            si_results (dict): the self-imitation results.

        Returns:
            None
        """

        with open(out_path, "w") as writer:
            writer.write("########################Self-imitation Results########################\n")
            for key, val in si_results.items():
                writer.write("{} = {}, quality = {}\n".format(key, val, TYPES_QUALITIES[key]))

    def make_video(self, output_dir, pred_files, src_infos, app_infos, motion_infos, image_size=512):
        """

        Args:
            output_dir (str): the directory to save the video.

            pred_files (list of str): the list of pred image paths.

            src_infos (dict): the source information contains:
                --images (list of str): the list of full paths of source images (the length is 1)
                --smpls (np.ndarray):
                --kps (np.ndarray):
                --formated_name (str):

            app_infos (dict): the appearance information contains:
                --images (list of str): the list of full paths of source images (the length is 1)
                --smpls (np.ndarray):
                --kps (np.ndarray):
                --formated_name (str):

            motion_infos (dict): the reference information contains:
                --images (list of str):
                --smpls (np.ndarray):
                --kps (np.ndarray):
                --formated_name (str):

            image_size (int)

        Returns:

        """

        src_paths = src_infos["images"]
        app_paths = app_infos["images"]
        motion_paths = motion_infos["images"]

        src_name = src_infos["formated_name"]
        app_name = app_infos["formated_name"]

        save_visual_path = os.path.join(output_dir, "{}-to-{}.mp4".format(src_name, app_name))

        src_app_paths = src_paths + app_paths
        fuse_source_reference_output(save_visual_path, src_app_paths, motion_paths,
                                     pred_files, image_size=image_size)

    def visual_video(self, output_dir, num_source, image_size=512):
        """

        Args:
            output_dir:
            num_source:
            image_size:

        Returns:

        """

        visual_dir = mkdir(os.path.join(output_dir, "visual"))
        si_visual_dir = mkdir(os.path.join(visual_dir, "self_imitation"))

        self.protocols.setup(num_sources=num_source, load_smpls=False, load_kps=False)

        si_pred_files = glob.glob(os.path.join(output_dir, "self_imitation", "*"))
        si_pred_files.sort()

        si_count = 0
        for vid_info in tqdm(self.protocols):
            # source information, contains {"images", "smpls", "kps"},
            # here "images" are the list of full paths of source images (the length is 1)
            src_infos = vid_info["source"]
            app_infos = vid_info["appearance"]
            motion_infos = vid_info["motion"]

            si_len = len(motion_infos["images"])
            self.make_video(si_visual_dir, si_pred_files[si_count:si_count + si_len],
                            src_infos, app_infos, motion_infos, image_size=image_size)
            si_count += si_len
