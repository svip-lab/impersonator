import os.path
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import numpy as np
from utils import cv_utils
from utils.util import load_pickle_file
import time


RNG = np.random.RandomState(2019)


class PlaceDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(PlaceDataset, self).__init__(opt, is_for_train)
        self._name = 'PlaceDataset'

        self._read_dataset_paths()

    def _read_dataset_paths(self):
        if self._is_for_train:
            sub_folder = 'train'
        else:
            sub_folder = 'val'

        self._data_dir = os.path.join(self._opt.place_dir, sub_folder)

        self.dataset = datasets.ImageFolder(self._data_dir, transform=self._transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item][0]

    def _create_transform(self):
        transforms.ToTensor()
        transform_list = [
            transforms.RandomResizedCrop(self._opt.image_size),
            transforms.RandomHorizontalFlip(),
            cv_utils.ImageNormalizeToTensor()]
        self._transform = transforms.Compose(transform_list)


class SMPLPlaceDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(SMPLPlaceDataset, self).__init__(opt, is_for_train)
        self._name = 'SMPLPlaceDataset'

        self._smpls = []
        self._read_dataset_paths()

    def _read_smpl_info(self, file_path):
        smpls = []
        with open(file_path, 'r') as reader:

            lines = []
            for line in reader:
                line = line.rstrip()
                lines.append(line)

            total = len(lines)
            for i, line in enumerate(lines):
                smpl_data = load_pickle_file(os.path.join(self._smpls_dir, line, 'pose_shape.pkl'))
                smpls.append(np.concatenate([smpl_data['cams'], smpl_data['pose'], smpl_data['shape']], axis=1))
                print(i, total, line)

        smpls = np.concatenate(smpls, axis=0)
        return smpls

    def _read_dataset_paths(self):
        if self._is_for_train:
            sub_folder = 'train'
        else:
            sub_folder = 'val'

        self._smpls_dir = os.path.join(self._opt.data_dir, self._opt.smpls_folder)
        self._place_dir = os.path.join(self._opt.place_dir, sub_folder)

        use_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        use_ids_filepath = os.path.join(self._opt.data_dir, use_ids_filename)
        self._smpls = self._read_smpl_info(use_ids_filepath)
        self._smpls_total = self._smpls.shape[0]
        self.dataset = datasets.ImageFolder(self._place_dir, transform=self._transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        images = self.dataset[item][0]
        smpls = self._smpls[RNG.randint(0, self._smpls_total)]
        # print(images.shape, smpls.shape)
        sample = {
            'images': images,
            'smpls': smpls
        }
        return sample

    def _create_transform(self):
        transforms.ToTensor()
        transform_list = [
            transforms.RandomResizedCrop(self._opt.image_size),
            transforms.RandomHorizontalFlip(),
            cv_utils.ImageNormalizeToTensor()]
        self._transform = transforms.Compose(transform_list)


if __name__ == '__main__':
    from utils.visualizer.demo_visualizer import MotionImitationVisualizer

    class Object(object):
        def __init__(self):
            self.place_dir = '/home/piaozx/liuwen/p300/places365_standard'
            self.image_size = 256

            self.data_dir = '/public/liuwen/p300/human_pose/processed'
            self.images_folder = 'motion_transfer_HD'
            self.smpls_folder = 'motion_transfer_smpl'
            self.train_ids_file = 'MI_train.txt'
            self.test_ids_file = 'MI_val.txt'

    opts = Object()
    smpl_place_dataset = SMPLPlaceDataset(opt=opts, is_for_train=True)

    visualizer = MotionImitationVisualizer(env='debug', ip='http://10.19.125.13', port=10087)

    print(len(smpl_place_dataset))
    for i in range(1000):
        sample = smpl_place_dataset[i]
        images = sample['images']
        smpls = sample['smpls']
        print(images.shape, smpls.shape)

        visualizer.vis_named_img('image', images[None, ...])
        time.sleep(1)



