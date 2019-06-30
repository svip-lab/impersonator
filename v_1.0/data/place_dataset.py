import os.path
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import numpy as np
from utils import cv_utils
from utils.util import load_pickle_file
import glob
import time


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


if __name__ == '__main__':
    import ipdb
    import torch.utils.data
    from utils.demo_visualizer import MotionImitationVisualizer

    class Object(object):
        def __init__(self):
            self.place_dir = '/p300/places365_standard'
            self.image_size = 256

    opts = Object()
    train_place_dataset = PlaceDataset(opts, is_for_train=True)
    test_place_datast = PlaceDataset(opts, is_for_train=False)

    total_datast = torch.utils.data.ConcatDataset([train_place_dataset, test_place_datast])
    print(len(train_place_dataset), len(test_place_datast), len(total_datast))

    # visualizer = MotionImitationVisualizer(env='debug', ip='http://10.10.10.100', port=31100)
    #
    # for i in range(1000):
    #     image = train_place_dataset[i]
    #
    #     visualizer.vis_named_img('image', image[None, ...])
    #
    #     print(i, image.shape)
    #     time.sleep(1)


