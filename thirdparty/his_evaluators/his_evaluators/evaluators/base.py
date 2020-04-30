import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from scipy.special import softmax

from his_evaluators.utils.io import load_img

from his_evaluators.metrics import TYPES_QUALITIES, BaseMetric, register_metrics
from his_evaluators.protocols import create_dataset_protocols


def parse_arg():
    parser = argparse.ArgumentParser(description="Evaluate Motion Imitation.")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    parser.add_argument("--data_dir", type=str, help="data directory")
    parser.add_argument("--output_dir", type=str, help="output directory")

    args = parser.parse_args()

    return args


class PairedEvaluationDataset(Dataset):
    def __init__(self, pair_file_list, image_size=512):
        self.image_size = image_size
        self.pair_file_list = pair_file_list

    def __len__(self):
        return len(self.pair_file_list)

    def __getitem__(self, item):
        pred_file, ref_file = self.pair_file_list[item]

        pred_img = load_img(pred_file, self.image_size)
        ref_img = load_img(ref_file, self.image_size)

        sample = {
            "pred": pred_img,
            "ref": ref_img
        }

        return sample


def build_data_loader(dataset, batch_size=32):
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size,
        num_workers=4, shuffle=False, drop_last=False,
        pin_memory=True
    )

    return data_loader


class PairedMetricRunner(object):
    def __init__(self,
                 metric_types=("ssim", "psnr", "lps"),
                 device=torch.device("cuda:0")):

        self.metric_types = metric_types
        self.metric_dict = register_metrics(metric_types, device)

    def build_metric_results(self, metric_types):
        metric_results = dict()

        for name in metric_types:
            metric_results[name] = []

        return metric_results

    def evaluate(self, file_paths, image_size=512, batch_size=16):
        dataset = PairedEvaluationDataset(file_paths, image_size=image_size)
        dataloader = build_data_loader(dataset, batch_size=batch_size)

        metric_results = self.build_metric_results(self.metric_types)
        print("running {} metrics with paired samples = {}".format(self.metric_types, len(dataset)))

        for sample in tqdm(dataloader):
            pred_imgs = sample["pred"]
            ref_imgs = sample["ref"]

            for name in self.metric_types:
                score = self.metric_dict[name].calculate_score(pred_imgs, ref_imgs)
                metric_results[name].append(score)

        return self.post_process_results(metric_results)

    def post_process_results(self, metric_results):
        for name in metric_results:
            metric_results[name] = np.mean(metric_results[name])

            print("{} = {}, quality = {}".format(name, metric_results[name], self.metric_dict[name].quality()))

        return metric_results


class UnpairedMetricRunner(object):
    def __init__(self,
                 metric_types=("is", "fid", "OS-CS-reID", "OS-freid", "face-CS", "face-FD"),
                 device=torch.device("cpu")):

        metric_types = set(metric_types)

        if "is" in metric_types and "fid" not in metric_types:
            metric_types.add("fid")

        add_PCB = False
        add_OSNET = False
        add_FACE = False
        for m_t in metric_types:
            if "PCB" in m_t:
                add_PCB = True
            if "OS" in m_t:
                add_OSNET = True
            if "face" in m_t:
                add_FACE = True

        if add_PCB:
            metric_types.add("PCB-freid")

        if add_OSNET:
            metric_types.add("OS-freid")

        if add_FACE:
            metric_types.add("face-CS")

        self.metric_types = tuple(metric_types)
        self.metric_dict = register_metrics(self.metric_types, device, has_detector=True)

        self.get_is_feats = False
        self.get_fid_feats = False
        self.get_osnet_feats = False
        self.get_pcb_feats = False
        self.get_cs_reid = False
        self.get_face_feats = False
        self.get_face_cs = False
        self.get_sspe = False

    def build_metric_results(self, metric_types):
        metric_results = dict()

        for m_t in metric_types:
            if m_t == "is":
                self.get_is_feats = True
                metric_results["inception_softmax"] = []
            elif m_t == "fid":
                self.get_fid_feats = True
                metric_results["inception_feats"] = {
                    "pred": [],
                    "ref": []
                }
            elif "PCB" in m_t:
                self.get_pcb_feats = True
                metric_results["pcb_feats"] = {
                    "pred": [],
                    "ref": [],
                    "CS": []
                }
                if "-CS" in m_t:
                    self.get_cs_reid = True

            elif "OS" in m_t:
                self.get_osnet_feats = True
                metric_results["osnet_feats"] = {
                    "pred": [],
                    "ref": [],
                    "CS": []
                }
                if "-CS" in m_t:
                    self.get_cs_reid = True

            elif "face" in m_t:
                self.get_face_feats = True
                metric_results["face_feats"] = {
                    "pred": [],
                    "ref": [],
                    "CS": []
                }
                if "-CS" in m_t:
                    self.get_face_cs = True

            elif "SSPE" in m_t:
                self.get_sspe = True
                metric_results["SSPE"] = []

        return metric_results

    def evaluate(self, file_paths, image_size=512, batch_size=16):
        """
        Args:
            file_paths:
            image_size:
            batch_size:

        Returns:

        """

        dataset = PairedEvaluationDataset(file_paths, image_size=image_size)
        dataloader = build_data_loader(dataset, batch_size=batch_size)

        metric_results = self.build_metric_results(self.metric_types)

        print("running {} metrics with unpaired samples = {}".format(self.metric_types, len(dataset)))
        for sample in tqdm(dataloader):
            pred_imgs = sample["pred"]
            ref_imgs = sample["ref"]

            if self.get_fid_feats:
                inception_preds = self.metric_dict["fid"].forward(pred_imgs)
                inception_refs = self.metric_dict["fid"].forward(ref_imgs)
                metric_results["inception_feats"]["pred"].append(inception_preds)
                metric_results["inception_feats"]["ref"].append(inception_refs)

                if self.get_is_feats:
                    is_softmax = softmax(inception_preds, axis=1)
                    metric_results["inception_softmax"].append(is_softmax)

            if self.get_osnet_feats:
                osnet_preds = self.metric_dict["OS-freid"].forward(pred_imgs)
                osnet_refs = self.metric_dict["OS-freid"].forward(ref_imgs)
                metric_results["osnet_feats"]["pred"].append(osnet_preds)
                metric_results["osnet_feats"]["ref"].append(osnet_refs)

                if self.get_cs_reid:
                    cs_score = self.metric_dict["OS-freid"].cosine_similarity(osnet_preds, osnet_refs)
                    metric_results["osnet_feats"]["CS"].append(cs_score)

            if self.get_pcb_feats:
                pcb_preds = self.metric_dict["PCB-freid"].forward(pred_imgs)
                pcb_refs = self.metric_dict["PCB-freid"].forward(ref_imgs)
                metric_results["pcb_feats"]["pred"].append(pcb_preds)
                metric_results["pcb_feats"]["ref"].append(pcb_refs)

                if self.get_cs_reid:
                    cs_score = self.metric_dict["PCB-freid"].cosine_similarity(pcb_preds, pcb_refs)
                    metric_results["pcb_feats"]["CS"].append(cs_score)

            if self.get_face_feats:
                face_preds, _ = self.metric_dict["face-CS"].forward(pred_imgs)
                face_refs, valid_ids = self.metric_dict["face-CS"].forward(ref_imgs)

                metric_results["face_feats"]["pred"].append(face_preds[valid_ids])
                metric_results["face_feats"]["ref"].append(face_refs[valid_ids])

                if self.get_face_cs:
                    cs_score = self.metric_dict["face-CS"].cosine_similarity(face_preds, face_refs)
                    metric_results["face_feats"]["CS"].append(cs_score)

            if self.get_sspe:
                sspe = self.metric_dict["SSPE"].calculate_score(pred_imgs, ref_imgs)
                metric_results["SSPE"].append(sspe)

        results = self.post_process_results(metric_results)

        return results

    def template_results_dict(self):
        results = dict()

        for key in self.metric_types:
            results[key] = []

        return results

    def post_process_results(self, metric_results):

        for key in metric_results:

            if key == "SSPE":
                continue

            if key == "inception_softmax" :
                metric_results[key] = np.concatenate(metric_results[key], axis=0)
                continue

            for sub_key in metric_results[key]:
                if sub_key == "CS":
                    continue
                feats = metric_results[key][sub_key]
                metric_results[key][sub_key] = np.concatenate(feats, axis=0)

        results = dict()
        if self.get_is_feats:
            feats_norm = metric_results["inception_softmax"]
            results["is"] = self.metric_dict["is"].is_score_func(feats_norm)

            print("is = {}, quality = {}".format(results["is"], TYPES_QUALITIES["is"]))

        if self.get_fid_feats:
            pred_feats = metric_results["inception_feats"]["pred"]
            ref_feats = metric_results["inception_feats"]["ref"]
            results["fid"] = BaseMetric.fid_score_func(pred_feats, ref_feats)

            print("fid = {}, quality = {}".format(results["fid"], TYPES_QUALITIES["fid"]))

        if self.get_osnet_feats:
            pred_feats = metric_results["osnet_feats"]["pred"]
            ref_feats = metric_results["osnet_feats"]["ref"]
            results["OS-freid"] = BaseMetric.fid_score_func(pred_feats, ref_feats)

            print("OS-freid = {}, quality = {}".format(results["OS-freid"], TYPES_QUALITIES["OS-freid"]))

            if self.get_cs_reid:
                results["OS-CS-reid"] = np.mean(metric_results["osnet_feats"]["CS"])

                print("OS-CS-reid = {}, quality = {}".format(results["OS-CS-reid"], TYPES_QUALITIES["OS-CS-reid"]))

        if self.get_pcb_feats:
            pred_feats = metric_results["pcb_feats"]["pred"]
            ref_feats = metric_results["pcb_feats"]["ref"]
            results["PCB-freid"] = BaseMetric.fid_score_func(pred_feats, ref_feats)

            print("PCB-freid = {}, quality = {}".format(results["PCB-freid"], TYPES_QUALITIES["PCB-freid"]))

            if self.get_cs_reid:
                results["PCB-CS-reid"] = np.mean(metric_results["pcb_feats"]["CS"])

                print("PCB-CS-reid = {}, quality = {}".format(results["PCB-CS-reid"], TYPES_QUALITIES["PCB-CS-reid"]))

        if self.get_face_feats:
            pred_feats = metric_results["face_feats"]["pred"]
            ref_feats = metric_results["face_feats"]["ref"]

            length = len(ref_feats)
            if length == 0:
                print("there is no face detected! We can not compute face-FD and face-CS.")
            else:
                results["face-FD"] = BaseMetric.fid_score_func(pred_feats, ref_feats)
                print("face-FD = {}, quality = {}".format(results["face-FD"], TYPES_QUALITIES["face-FD"]))
                if self.get_face_cs:
                    results["face-CS"] = np.mean(metric_results["face_feats"]["CS"])
                    print("face-CS = {}, quality = {}".format(results["face-CS"], TYPES_QUALITIES["face-CS"]))

        if self.get_sspe:
            results["SSPE"] = np.mean(metric_results["SSPE"])
            print("SSPE = {}, quality = {}".format(results["SSPE"], TYPES_QUALITIES["SSPE"]))

        return results


class Evaluator(object):
    def __init__(self, dataset, data_dir):
        self.dataset = dataset
        self.protocols = create_dataset_protocols(dataset, data_dir)

    def build_metrics(self):
        raise NotImplementedError

    def run_inference(self, *args, **kwargs):
        raise NotImplementedError

    def run_metrics(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def preprocess(self, *args, **kwargs):
        raise NotImplementedError
