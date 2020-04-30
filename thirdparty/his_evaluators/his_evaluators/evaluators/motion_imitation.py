from abc import ABC
import torch
from multiprocessing import Process, Manager
from tqdm import tqdm
from typing import List, Dict, Any
import os

from his_evaluators.metrics import TYPES_QUALITIES

from .base import PairedMetricRunner, UnpairedMetricRunner, Evaluator
from ..utils.io import mkdir


class MotionImitationModel(object):

    def __init__(self, output_dir):
        """

        Args:
            output_dir:
        """

        self.output_dir = mkdir(output_dir)
        self.si_out_dir = mkdir(os.path.join(output_dir, "self_imitation"))
        self.ci_out_dir = mkdir(os.path.join(output_dir, "cross_imitation"))
        self.num_preds_si = 0
        self.num_preds_ci = 0

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

        processed_src_infos = src_infos
        return processed_src_infos


class MotionImitationRunnerProcessor(Process):
    def __init__(self, model, protocols, return_dict: Manager):
        """
            The processor of running motion imitation models.
        Args:
            model (MotionImitationModel):
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
        # ci means cross-imitation
        all_ci_preds_ref_file_list = []

        for vid_info in tqdm(self.protocols):
            # source information, contains {"images", "smpls", "kps"},
            # here "images" are the list of full paths of source images (the length is 1)
            src_infos = vid_info["source"]

            # run personalization
            src_infos = self.model.personalization(src_infos)

            # si means (self-imitation)
            si_infos = vid_info["self_imitation"]
            si_pred_files = self.model.imitate(src_infos, si_infos)

            # ci means (cross-imitation)
            ci_infos = vid_info["cross_imitation"]
            ci_pred_files = self.model.imitate(src_infos, ci_infos)

            si_pred_ref_files, ci_pred_ref_files = self.post_format_metric_file_list(
                si_pred_files, si_infos["images"],
                ci_pred_files, vid_info["flag"]
            )

            all_si_preds_ref_file_list.extend(si_pred_ref_files)
            all_ci_preds_ref_file_list.extend(ci_pred_ref_files)

            # break

        self.return_dict["all_si_preds_ref_file_list"] = all_si_preds_ref_file_list
        self.return_dict["all_ci_preds_ref_file_list"] = all_ci_preds_ref_file_list

    def terminate(self) -> None:
        self.model.terminate()

    def post_format_metric_file_list(self, si_preds_files, si_ref_files, ci_preds_files, ci_ref_files):
        """
            make [(si_pred, si_ref), ...], and [(ci_pred, ci_ref), ...]
        Args:
            si_preds_files:
            si_ref_files:
            ci_preds_files:
            ci_ref_files:

        Returns:
            si_preds_ref_files:
            ci_preds_ref_files:
        """
        si_preds_ref_files = list(zip(si_preds_files, si_ref_files))
        ci_preds_ref_files = list(zip(ci_preds_files, ci_ref_files))

        return si_preds_ref_files, ci_preds_ref_files


class MotionImitationEvaluator(Evaluator, ABC):
    def __init__(self, dataset, data_dir):
        super().__init__(dataset, data_dir)

        # please call `build_metrics` to instantiate these two runners.
        self.paired_metrics_runner = None
        self.unpaired_metrics_runner = None

    def reset_dataset(self, dataset, data_dir):
        super().__init__(dataset, data_dir)

    def build_metrics(
        self,
        pair_types=("ssim", "psnr", "lps"),
        unpair_types=("is", "fid", "PCB-freid", "PCB-CS-reid"),
        device=torch.device("cpu")
    ):
        paired_metrics_runner = PairedMetricRunner(metric_types=pair_types, device=device)
        unpaired_metrics_runner = UnpairedMetricRunner(metric_types=unpair_types, device=device)

        self.paired_metrics_runner = paired_metrics_runner
        self.unpaired_metrics_runner = unpaired_metrics_runner

    def run_metrics(self, self_imitation_files, cross_imitation_files, image_size=512):
        assert (self.paired_metrics_runner is not None or self.unpaired_metrics_runner is not None), \
            "please call `build_metrics(pair_types, unpair_types)` to instantiate metrics runners " \
            "before calling this function."

        si_results = self.paired_metrics_runner.evaluate(self_imitation_files, image_size)
        ci_results = self.unpaired_metrics_runner.evaluate(cross_imitation_files, image_size)

        return si_results, ci_results

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def run_inference(self, *args, **kwargs):
        raise NotImplementedError


class IPERMotionImitationEvaluator(MotionImitationEvaluator):

    def __init__(self, data_dir, dataset="iPER"):
        super().__init__(dataset=dataset, data_dir=data_dir)

    def run_inference(self, model, src_infos, ref_infos):
        """
        Args:
            model (MotionImitationModel): the model object, it must define and implements the function
                            `imitate(src_infos, ref_infos, is_self_imitation) -> List[str]`
            src_infos (dict): the source information contains:
                --images (list of str): the list of full paths of source images (the length is 1)
                --smpls (np.ndarray):
                --kps (np.ndarray):
            ref_infos (dict): the reference information contains:
                --images (list of str):
                --smpls (np.ndarray):
                --kps (np.ndarray):
                --self_imitation (bool):

        Returns:
            file_paths (list of str): [pred_img_path_0, pred_img_path_1, ..., pred_img_path_i, ..., pred_img_path_n)]

        """

        assert hasattr(model, "imitate"), '{} must implement imitate(src_infos, ref_infos) -> List[str]'

        file_paths = model.imitate(src_infos, ref_infos)

        return file_paths

    def evaluate(self, model, num_sources=1, image_size=512,
                 pair_types=("ssim", "psnr", "lps"),
                 unpair_types=("is", "fid", "PCB-freid", "PCB-CS-reid"),
                 device=torch.device("cpu")):
        # 1. setup protocols
        self.protocols.setup(num_sources=num_sources, load_smpls=True, load_kps=True)

        # 2. declare runner processor for inference
        return_dict = Manager().dict({})
        runner = MotionImitationRunnerProcessor(model, self.protocols, return_dict)
        runner.start()
        runner.join()

        del model

        all_si_preds_ref_file_list = return_dict["all_si_preds_ref_file_list"]
        all_ci_preds_ref_file_list = return_dict["all_ci_preds_ref_file_list"]

        # run metrics
        self.build_metrics(pair_types, unpair_types, device)
        si_results, ci_results = self.run_metrics(all_si_preds_ref_file_list, all_ci_preds_ref_file_list, image_size)

        return si_results, ci_results

    def preprocess(self, *args, **kwargs):
        pass

    def save_results(self, out_path, si_results, ci_result):
        """
            save the the results into the out_path.
        Args:
            out_path (str): the full path to save the results.
            si_results (dict): the self-imitation results.
            ci_result (dict): the cross-imitation results.

        Returns:
            None
        """

        with open(out_path, "w") as writer:
            writer.write("########################Self-imitation Results########################\n")
            for key, val in si_results.items():
                writer.write("{} = {}, quality = {}\n".format(key, val, TYPES_QUALITIES[key]))

            writer.write("########################Cross-imitation Results########################\n")
            for key, val in ci_result.items():
                writer.write("{} = {}, quality = {}\n".format(key, val, TYPES_QUALITIES[key]))

