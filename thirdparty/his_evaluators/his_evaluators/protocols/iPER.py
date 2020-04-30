import os
import glob
import numpy as np

from .protocol import Protocol
from ..utils.io import load_json_file, load_pickle_file


class IPERProtocol(Protocol):

    def __init__(self, data_dir="/p300/iPER"):
        super().__init__()

        # the root directory of iPER, need to be replaced!
        self.data_dir = data_dir
        self.train_ids_file = "train.txt"
        self.test_ids_file = "val.txt"
        self.eval_path = "iPER_protocol.json"
        self.images_folder = "images_HD"
        self.smpls_folder = "smpls"

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
            "data", "iPER_protocol.json"
        )

        self.eval_info = load_json_file(full_eval_path)["val"]
        self.vid_names = list(self.eval_info.keys())

        self._all_vid_smpls = {}
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
        vid_path = os.path.join(self.data_dir, self.images_folder, vid_name)
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

        src_vid_path = os.path.join(self.data_dir, self.images_folder, vid_name)
        src_img_paths = glob.glob(os.path.join(src_vid_path, "*"))
        src_img_paths.sort()

        src_img_names = vid_info["s_n"][str(num_sources)]
        src_img_ids = [int(t.split(".")[0]) for t in src_img_names]
        eval_info["source"] = {
            "s_n": num_sources,
            "name": vid_name,
            "formated_name": self.format_name(vid_name),
            "vid_path": os.path.join(self.data_dir, self.images_folder, vid_name),
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

    def format_name(self, name):
        """
            convert `001/9/1` to `001_9_1`
        Args:
            name (str): such as `001/9/1`.

        Returns:
            formated_name (str): such as `001_9_1`
        """

        formated_name = "_".join(name.split("/"))
        return formated_name

    def original_name(self, formated_name):
        """

        Args:
            formated_name:

        Returns:

        """
        original_name = "/".join(formated_name.split("_"))

        return original_name

    def get_smpl_path(self, name):
        """

        Args:
            name (str): such as `001/9/1`.

        Returns:
            smpl_path (str):
        """

        smpl_path = os.path.join(self.data_dir, self.smpls_folder, name, "pose_shape.pkl")
        return smpl_path

    def get_kps_path(self, name):
        """

        Args:
            name (str): such as `001/9/1`.

        Returns:
            kps_path (str):
        """
        smpl_path = os.path.join(self.data_dir, self.smpls_folder, name, "kps.pkl")
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
                kps = load_pickle_file(kps_path)["kps"]
                self._all_vid_kps[name] = kps
            else:
                kps = self._all_vid_kps[name]
        return kps

    @property
    def total_frames(self):
        total = 0
        for vid_name, vid_info in self.eval_info.items():
            src_vid = os.path.join(self.data_dir, self.images_folder, vid_name)
            length = len(os.listdir(src_vid))
            total += length
        return total


class ICCVIPERProtocol(IPERProtocol):
    def __init__(self, data_dir="/p300/iPER"):
        super().__init__(data_dir)

    def __len__(self):
        return len(self.vid_names) * 3

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
        self._num_source = 1
        self._load_smpls = True
        self._load_kps = True
        num_sources = self._num_source
        load_smpls = self._load_smpls
        load_kps = self._load_kps

        vid_ids = item // 3
        src_ids = item % 3

        vid_name = self.vid_names[vid_ids]
        vid_info = self.eval_info[vid_name]

        eval_info = dict()

        # 1. source information
        src_vid_smpls = self.get_smpls(vid_name)
        src_vid_kps = self.get_kps(vid_name)

        src_vid_path = os.path.join(self.data_dir, self.images_folder, vid_name)
        src_img_paths = glob.glob(os.path.join(src_vid_path, "*"))
        src_img_paths.sort()

        src_img_names = vid_info["s_n"]["3"][src_ids:src_ids+1]
        src_img_ids = [int(t.split(".")[0]) for t in src_img_names]
        eval_info["source"] = {
            "s_n": num_sources,
            "name": vid_name,
            "formated_name": self.format_name(vid_name),
            "vid_path": os.path.join(self.data_dir, self.images_folder, vid_name),
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
