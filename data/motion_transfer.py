import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import numpy as np
from utils import cv_utils
from utils.util import load_pickle_file
import glob

from .place_dataset import PlaceDataset


class MIDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(MIDataset, self).__init__(opt, is_for_train)
        self._name = 'MIDataset'

        if 'intervals' in self._opt.__dict__:
            self._intervals = opt.intervals[-1]
        else:
            self._intervals = 20

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

                cams = smpl_data['cams']

                assert len(images_path) == len(cams), '{} != {}'.format(len(images_path), len(cams))

                info = {
                    'images': images_path,
                    'cams': cams,
                    'thetas': smpl_data['pose'],
                    'betas': smpl_data['shape'],
                    'length': len(images_path)
                }
                vids_info.append(info)
                self._dataset_size += info['length'] // self._intervals
                # self._dataset_size += info['length']
                self._num_videos += 1
                print('loading video = {}, {} / {}'.format(line, i, total))

                # if i > 5:
                #     break

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
            cv_utils.ImageTransformer(output_size=self._opt.image_size),
            cv_utils.ToTensor()]
        self._transform = transforms.Compose(transform_list)


class MIDatasetV2(MIDataset):

    def __init__(self, opt, is_for_train):
        super(MIDatasetV2, self).__init__(opt, is_for_train)
        self._name = 'MIDatasetV2'

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


class FastLoadMIDataset(MIDataset):

    def __init__(self, opt, is_for_train):
        super(FastLoadMIDataset, self).__init__(opt, is_for_train)
        self._name = 'FastMIDataset'

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

                cams = smpl_data['cams']

                assert len(images_path) == len(cams), '{} != {}'.format(len(images_path), len(cams))

                length = len(images_path)
                ids = np.arange(0, length, self._intervals)

                info = {
                    'images': images_path[0:length:self._intervals],
                    'cams': cams[ids],
                    'thetas': smpl_data['pose'][ids],
                    'betas': smpl_data['shape'][ids],
                    'length': len(ids)
                }
                vids_info.append(info)
                self._dataset_size += info['length']
                self._num_videos += 1
                print('loading video = {}, {} / {}'.format(line, i, total))

                if i > 4:
                    break

        return vids_info

    def _load_pairs(self, vid_info):
        length = vid_info['length']

        start = 0
        # start = np.random.randint(0, 15)
        end = np.random.randint(0, length)
        # pair_ids = np.array([0, length // 2], dtype=np.int32)
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


class MIv2PlaceDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(MIv2PlaceDataset, self).__init__(opt, is_for_train)
        self._name = 'MIv2PlaceDataset'

        # self.mi = FastLoadMIDataset(opt, is_for_train)
        self.mi = MIDatasetV2(opt, is_for_train)
        self.place = PlaceDataset(opt, is_for_train)

        self._dataset_size = len(self.mi)
        num_places = len(self.place)
        interval = num_places // self._dataset_size
        self.sample_ids = np.arange(0, num_places, interval)[0:self._dataset_size]

        print(self._dataset_size, len(self.sample_ids))

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, item):
        sample = self.mi[item]
        bg = self.place[self.sample_ids[item]]

        sample['bg'] = bg
        return sample


if __name__ == '__main__':
    import torch
    from utils.demo_visualizer import MotionImitationVisualizer
    import ipdb

    class Object(object):
        def __init__(self):
            self.data_dir = '/public/liuwen/p300/human_pose/processed'
            self.images_folder = 'motion_transfer_HD'
            self.smpls_folder = 'motion_transfer_smpl'
            self.train_ids_file = 'MI_train.txt'
            self.test_ids_file = 'MI_val.txt'
            self.image_size = 256
            self.place_dir = '/home/piaozx/liuwen/p300/places365_standard'
            self.time_step = 10
            self.intervals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    opts = Object()

    mi_place = MIv2PlaceDataset(opts, is_for_train=True)

    print(len(mi_place))

    _dataloader = torch.utils.data.DataLoader(
        mi_place,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    visualizer = MotionImitationVisualizer(env='debug', ip='http://10.19.125.13', port=10087)

    for i, sample in enumerate(_dataloader):
        print('batch iter = {}'.format(i))

        # for key, value in sample.items():
        #     print('\t {}, {}'.format(key, value.shape))

        bg = sample['bg']
        visualizer.vis_named_img('bg', bg)