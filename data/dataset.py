import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path


class DatasetFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train):
        if dataset_name == 'imper':
            from data.imper_dataset import ImPerDataset
            dataset = ImPerDataset(opt, is_for_train)

        elif dataset_name == 'fashion':
            from data.fashion_dataset import FashionPairDataset
            dataset = FashionPairDataset(opt, is_for_train)

        elif dataset_name == 'imper_place':
            from data.imper_fashion_place_dataset import ImPerPlaceDataset
            dataset = ImPerPlaceDataset(opt, is_for_train)

        elif dataset_name == 'imper_fashion_place':
            from data.imper_fashion_place_dataset import ImPerFashionPlaceDataset
            dataset = ImPerFashionPlaceDataset(opt, is_for_train)

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


