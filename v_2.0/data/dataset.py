import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path


class DatasetFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train):
        if dataset_name == 'mi':
            from data.motion_transfer import MIDataset
            dataset = MIDataset(opt, is_for_train)

        elif dataset_name == 'mi_v2':
            from data.motion_transfer import MIDatasetV2
            dataset = MIDatasetV2(opt, is_for_train)

        elif dataset_name == 'fast_mi':
            from data.motion_transfer import FastLoadMIDataset
            dataset = FastLoadMIDataset(opt, is_for_train)

        elif dataset_name == 'miv2_place':
            from data.motion_transfer import MIv2PlaceDataset
            dataset = MIv2PlaceDataset(opt, is_for_train)

        elif dataset_name == 'smpl_place':
            from data.place_dataset import SMPLPlaceDataset
            dataset = SMPLPlaceDataset(opt, is_for_train)

        elif dataset_name == 'imper_sample':
            from data.motion_transfer import ImPerSampleDataset
            dataset = ImPerSampleDataset(opt, is_for_train)

        elif dataset_name == 'deep_fashion':
            from data.motion_transfer import DeepFashionPairSampleDataset
            dataset = DeepFashionPairSampleDataset(opt, is_for_train)

        elif dataset_name == 'df_imper':
            from data.motion_transfer import create_imper_fashion_pair_dataset
            dataset = create_imper_fashion_pair_dataset(opt, is_for_train)

        elif dataset_name == 'imper_pair':
            from data.motion_transfer import ImperPairSampleDataset
            dataset = ImperPairSampleDataset(opt, is_for_train)

        elif dataset_name == 'deep_fashion_hmr':
            from data.motion_transfer import DeepFashionHmrPairSampleDataset
            dataset = DeepFashionHmrPairSampleDataset(opt, is_for_train)

        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, opt, is_for_train):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._opt = opt
        self._is_for_train = is_for_train
        self._create_transform()

        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        self._transform = transforms.Compose([])

    def get_transform(self):
        return self._transform

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

    def _is_csv_file(self, filename):
        return filename.endswith('.csv')

    def _get_all_files_in_subfolders(self, dir, is_file):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

        return images

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


