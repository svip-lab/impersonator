import os
import glob
import numpy as np

from .protocol import Protocol
from ..utils.io import load_json_file, load_pickle_file


def find_index_from_name(src_img_names, src_vid_path):
    vid_img_names = os.listdir(src_vid_path)
    vid_img_names.sort()
    num_names = len(vid_img_names)

    num_src = len(src_img_names)
    src_ids = sorted(range(num_src), key=lambda k: src_img_names[k])

    index_in_vid = []

    pi = 0

    for pj in range(num_names):
        if src_img_names[src_ids[pi]] == vid_img_names[pj]:
            index_in_vid.append(pj)

            pi += 1

        if pi == num_src:
            break

    return index_in_vid


class YoutubeDancer18Protocol(Protocol):

    def __init__(self, data_dir="/home/piaozx/liuwen/p300/human_pose/processed", eval_path="Youtube-Dancer-18.json"):
        super().__init__()

        # the root directory of iPER, need to be replaced!
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.train_ids_file = "train.txt"
        self.test_ids_file = "val.txt"
        self.eval_path = eval_path

        """
        "001/9/1": {
          "source": ["000.jpg", "035.jpg", "091.jpg", "120.jpg", "155.jpg", "195.jpg", "219.jpg", "251.jpg"],
          "view angle": [0, 45, 90, 135, 180, 225, 270, 315],
          "s_n" : {
              "1": ["000.jpg"],
              "2": ["000.jpg", "155.jpg"],
              "4": ["000.jpg", "091.jpg", "155.jpg", "219.jpg"],
              "8": ["000.jpg", "035.jpg", "091.jpg", "120.jpg", "155.jpg", "195.jpg", "219.jpg", "251.jpg"]
          },
          "mask": [],
          "novel view": false,
          "self_imitation": {
            "target": "001/9/1",
            "range": [0, 300]
          },
          "cross_imitation": {
            "target": "007/1/2",
            "range": [180, 255]
          },
          "flag": [180, 255]
        },
        
        """

        full_eval_path = os.path.join(
            os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "data", self.eval_path
        )

        self.eval_info = load_json_file(full_eval_path)["val"]
        self.vid_names = list(self.eval_info.keys())

        self._all_vid_smpls = {}
        self._all_vid_offsets = {}
        self._all_vid_kps = {}

        # setups
        self._num_source = 1
        self._load_smpls = False
        self._load_kps = False

    def __len__(self):
        return len(self.vid_names)

    def take_images_paths(self, vid_name, start, end):
        """
        Args:
            vid_name:
            start:
            end:

        Returns:

        """
        vid_path = os.path.join(self.processed_dir, vid_name, "images")
        vid_images_paths = glob.glob(os.path.join(vid_path, "*"))
        vid_images_paths.sort()
        images_paths = vid_images_paths[start: end + 1]
        return images_paths

    def setup(self, num_sources=1, load_smpls=False, load_kps=False):
        self._num_source = num_sources
        self._load_smpls = load_smpls
        self._load_kps = load_kps

    def __getitem__(self, item):
        """

        Args:
            item:

        Returns:
            eval_info (dict): the information for evaluation, it contains:

                --source (dict):
                    --s_n (str): dict map of from number of source (s_n) to source images
                    --name (str): the video name of source (`001/9/1`)
                    --formated_name (str): the formated video name of source (`001_9_1`);
                    --vid_path (str):
                    --images (list of str):
                    --smpls (np.ndarray or None):
                    --kps (np.ndarray or None):

                --self_imitation (dict):
                    --images (list of str):
                    --smpls (np.ndarray or None):
                    --kps (np.ndarray or None):
                    --self_imitation (bool): True

                --cross_imitation (dict):
                    --images (list of str):
                    --smpls (np.ndarray or None):
                    --kps (np.ndarray or None):
                    --self_imitation (bool): False

                --flag (list of str):
        """

        num_sources = self._num_source
        load_smpls = self._load_smpls
        load_kps = self._load_kps

        vid_name = self.vid_names[item]
        vid_info = self.eval_info[vid_name]

        eval_info = dict()

        # 1. source information
        src_vid_smpls = self.get_smpls(vid_name)
        src_vid_kps = self.get_kps(vid_name)

        src_vid_path = os.path.join(self.processed_dir, vid_name, "images")
        src_img_paths = glob.glob(os.path.join(src_vid_path, "*"))
        src_img_paths.sort()

        src_img_names = vid_info["s_n"][str(num_sources)]

        # "frame_00000000.png" -> "00000000" -> int("00000000")
        src_img_ids = find_index_from_name(src_img_names, src_vid_path)

        eval_info["source"] = {
            "s_n": num_sources,
            "name": vid_name,
            "formated_name": self.format_name(vid_name),
            "vid_path": os.path.join(self.processed_dir, vid_name, "images"),
            "images": [src_img_paths[t] for t in src_img_ids],
            "smpls": src_vid_smpls[src_img_ids] if load_smpls else None,
            "kps": src_vid_kps[src_img_ids] if load_kps else None
        }

        # 2. self-imitation
        self_imitation = vid_info["self_imitation"]

        eval_info["self_imitation"] = {
            "name": self_imitation["target"],
            "formated_name": self.format_name(self_imitation["target"]),
            "images": src_img_paths[self_imitation["range"][0]: self_imitation["range"][1] + 1],
            "smpls": src_vid_smpls[self_imitation["range"][0]: self_imitation["range"][1] + 1] if load_smpls else None,
            "kps": src_vid_kps[self_imitation["range"][0]: self_imitation["range"][1] + 1] if load_kps else None,
            "self_imitation": True
        }

        # 2. cross-imitation
        cross_imitation = vid_info["cross_imitation"]
        target_vid_name = cross_imitation["target"]
        target_vid_smpls = self.get_smpls(target_vid_name)
        target_vid_kps = self.get_kps(target_vid_name)
        cross_images_paths = self.take_images_paths(
            vid_name=target_vid_name,
            start=cross_imitation["range"][0],
            end=cross_imitation["range"][1]
        )
        eval_info["cross_imitation"] = {
            "name": target_vid_name,
            "formated_name": self.format_name(target_vid_name),
            "images": cross_images_paths,
            "smpls": target_vid_smpls[
                     cross_imitation["range"][0]: cross_imitation["range"][1] + 1] if load_smpls else None,
            "kps": target_vid_kps[
                   cross_imitation["range"][0]: cross_imitation["range"][1] + 1] if load_kps else None,
            "self_imitation": False
        }

        eval_info["flag"] = self.take_images_paths(
            vid_name=vid_name,
            start=vid_info["flag"][0],
            end=vid_info["flag"][1]
        )

        # print(vid_name, cross_imitation["range"][1] - cross_imitation["range"][0],
        #       vid_info["flag"][1] - vid_info["flag"][0])
        assert cross_imitation["range"][1] - cross_imitation["range"][0] == vid_info["flag"][1] - vid_info["flag"][0]

        return eval_info

    def get_smpl_path(self, name):
        """

        Args:
            name (str):

        Returns:
            smpl_path (str):
        """

        smpl_path = os.path.join(self.processed_dir, name, "pose_shape.pkl")
        return smpl_path

    def get_kps_path(self, name):
        """

        Args:
            name (str):

        Returns:
            kps_path (str):
        """
        smpl_path = os.path.join(self.processed_dir, name, "kps.pkl")
        return smpl_path

    def get_smpls(self, name):
        smpls = None
        if name in self.eval_info:
            if name not in self._all_vid_smpls:
                smpl_path = self.get_smpl_path(name)
                smpl_data = load_pickle_file(smpl_path)
                cams = smpl_data['cams']
                thetas = smpl_data['pose']
                betas = smpl_data["shape"]
                smpls = np.concatenate([cams, thetas, betas], axis=1)
                self._all_vid_smpls[name] = smpls
            else:
                smpls = self._all_vid_smpls[name]

        return smpls

    def get_kps(self, name):
        kps = None
        if name in self.eval_info:
            if name not in self._all_vid_kps:
                kps_path = self.get_kps_path(name)
                kps = load_pickle_file(kps_path)
                self._all_vid_kps[name] = kps
            else:
                kps = self._all_vid_kps[name]
        return kps

    @property
    def total_frames(self):
        total = 0
        for vid_name, vid_info in self.eval_info.items():
            src_vid = os.path.join(self.processed_dir, vid_name)
            length = len(os.listdir(src_vid))
            total += length
        return total

