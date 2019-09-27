import numpy as np

from .dataset import DatasetBase
from .imper_dataset import ImPerDataset
from .place_dataset import PlaceDataset
from .fashion_dataset import FashionPairDataset


class ImPerPlaceDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(ImPerPlaceDataset, self).__init__(opt, is_for_train)
        self._name = 'ImPerPlaceDataset'

        # self.mi = FastLoadMIDataset(opt, is_for_train)
        self.mi = ImPerDataset(opt, is_for_train)
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


class ImPerFashionPlaceDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(ImPerFashionPlaceDataset, self).__init__(opt, is_for_train)
        self._name = 'ImPerFashionPlaceDataset'

        # self.mi = FastLoadMIDataset(opt, is_for_train)
        self.mi = ImPerDataset(opt, is_for_train)
        self.place = PlaceDataset(opt, is_for_train)
        self.fashion = FashionPairDataset(opt, is_for_train)

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
        fashion = self.fashion[item]

        sample['bg'] = bg

        sample['fashion_images'] = fashion['images']
        sample['fashion_masks'] = fashion['pseudo_masks']
        sample['fashion_head_bbox'] = fashion['head_bbox']
        sample['fashion_body_bbox'] = fashion['body_bbox']
        sample['fashion_bg_inputs'] = fashion['bg_inputs']
        sample['fashion_src_inputs'] = fashion['src_inputs']
        sample['fashion_tsf_inputs'] = fashion['tsf_inputs']
        sample['fashion_T'] = fashion['T']

        return sample




