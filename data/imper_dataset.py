import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import numpy as np
from utils import cv_utils
from utils.util import load_pickle_file, ToTensor, ImageTransformer
import glob


__all__ = ['ImPerBaseDataset', 'ImPerDataset']


class ImPerBaseDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(ImPerBaseDataset, self).__init__(opt, is_for_train)
        self._name = 'ImPerBaseDataset'

        self._intervals = opt.intervals

        # read dataset
        self._read_dataset_paths()

    def __getitem__(self, index):
        # assert (index < self._dataset_size)

        # start_time = time.time()
        # get sample data
        v_info = self._vids_info[index % self._num_videos]
        images, smpls = self._load_pairs(v_info)

        # pack data
        sample = {
            'images': images,
            'smpls': smpls
        }

        sample = self._transform(sample)
        # print(time.time() - start_time)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._vids_dir = os.path.join(self._root, self._opt.images_folder)
        self._smpls_dir = os.path.join(self._root, self._opt.smpls_folder)

        # read video list
        self._num_videos = 0
        self._dataset_size = 0
        use_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        use_ids_filepath = os.path.join(self._root, use_ids_filename)
        self._vids_info = self._read_vids_info(use_ids_filepath)

    def _read_vids_info(self, file_path):
        vids_info = []
        with open(file_path, 'r') as reader:

            lines = []
            for line in reader:
                line = line.rstrip()
                lines.append(line)

            total = len(lines)
            for i, line in enumerate(lines):
                images_path = glob.glob(os.path.join(self._vids_dir, line, '*.jpg'))
                images_path.sort()
                smpl_data = load_pickle_file(os.path.join(self._smpls_dir, line, 'pose_shape.pkl'))
                kps_data = load_pickle_file(os.path.join(self._smpls_dir, line, 'kps.pkl'))

                cams = smpl_data['cams']
                kps = (kps_data['kps'] + 1) / 2.0 * 1024

                assert len(images_path) == len(cams), '{} != {}'.format(len(images_path), len(cams))

                info = {
                    'images': images_path,
                    'cams': cams,
                    'thetas': smpl_data['pose'],
                    'betas': smpl_data['shape'],
                    'j2ds': kps,
                    'length': len(images_path)
                }
                vids_info.append(info)
                self._dataset_size += info['length'] // self._intervals
                # self._dataset_size += info['length']
                self._num_videos += 1
                print('loading video = {}, {} / {}'.format(line, i, total))

                if self._opt.debug:
                    if i > 1:
                        break

        return vids_info

    @property
    def video_info(self):
        return self._vids_info

    def _load_pairs(self, vid_info):
        length = vid_info['length']
        pair_ids = np.random.choice(length, size=2, replace=False)

        smpls = np.concatenate((vid_info['cams'][pair_ids],
                                vid_info['thetas'][pair_ids],
                                vid_info['betas'][pair_ids]), axis=1)

        images = []
        images_paths = vid_info['images']
        for t in pair_ids:
            image_path = images_paths[t]
            image = cv_utils.read_cv2_img(image_path)

            images.append(image)

        return images, smpls

    def _create_transform(self):
        transform_list = [
            ImageTransformer(output_size=self._opt.image_size),
            ToTensor()]
        self._transform = transforms.Compose(transform_list)


class ImPerDataset(ImPerBaseDataset):

    def __init__(self, opt, is_for_train):
        super(ImPerDataset, self).__init__(opt, is_for_train)
        self._name = 'ImPerDataset'

    def _load_pairs(self, vid_info):
        length = vid_info['length']

        start = np.random.randint(0, 15)
        end = np.random.randint(0, length)
        pair_ids = np.array([start, end], dtype=np.int32)

        smpls = np.concatenate((vid_info['cams'][pair_ids],
                                vid_info['thetas'][pair_ids],
                                vid_info['betas'][pair_ids]), axis=1)

        images = []
        images_paths = vid_info['images']
        for t in pair_ids:
            image_path = images_paths[t]
            image = cv_utils.read_cv2_img(image_path)

            images.append(image)

        return images, smpls


