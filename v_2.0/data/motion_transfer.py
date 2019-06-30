import os.path
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import numpy as np
from utils import cv_utils
from utils.util import load_pickle_file, morph, cal_head_bbox
import glob
import h5py
import networks.bodymesh.mesh as mesh

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

                if i > 10:
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


class ImPerSampleDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(ImPerSampleDataset, self).__init__(opt, is_for_train)
        self._name = 'ImperSampleDataset'

        self.is_both_bg = self._opt.is_both

        # read dataset
        self._read_dataset_paths()

        # prepare mapping function
        self.map_fn = mesh.create_mapping(map_name=opt.map_name, mapping_path=opt.uv_mapping,
                                          contain_bg=True, fill_back=False)

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._sample_dir = self._opt.sample_dir

        # read video list
        self._num_videos = 0
        self._num_samples = 0
        self._dataset_size = 0
        self._sample_files = []

        use_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        use_ids_filepath = os.path.join(self._root, use_ids_filename)
        self._read_vids_info(use_ids_filepath)

    def _read_vids_info(self, file_path):

        def load_num_sample(h5file_path):
            """

            Args:
                h5file_path (str): the full path of hdh5 file of each video, the data items are followings:

                'smpls': (N, 85),
                'new_cams': (N, 3),
                'fims': (N, 2, 256, 256),
                'imgs': (N, 2, 3, 256, 256),
                'T':    (N, 256, 256, 2),
                'warp'  (N, 3, 256, 256):

            Returns:
                N (int): the number of samples.
            """
            with h5py.File(h5file_path, 'r') as sample_info:
                return sample_info['new_cams'].shape[0]

        with open(file_path, 'r') as reader:
            lines = []
            for line in reader:
                line = line.rstrip()
                lines.append(line)

            self._num_videos = len(lines)
            num_sample = 0
            for i, line in enumerate(lines):
                h5_path = os.path.join(self._sample_dir, line, 'sample.h5')
                self._sample_files.append(h5_path)

                cur_num = load_num_sample(h5_path)
                if i != 0:
                    assert cur_num == num_sample, \
                        '{} has {} samples, but others has {} samples'.format(line, cur_num, num_sample)

                self._dataset_size += cur_num
                num_sample = cur_num
                print('loading video = {}, {} / {}'.format(line, i, self._num_videos))

            # check total size % num videos == 0
            assert self._dataset_size % self._num_videos == 0, \
                '{} % {} != 0'.format(self._dataset_size, self._num_videos)

            self._num_samples = self._dataset_size // self._num_videos

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, item):
        """
        Args:
            item (int):  index of self._dataset_size

        Returns:
            new_sample (dict): items contain
                --src_inputs (torch.FloatTensor): (3+3, h, w)
                --tsf_inputs (torch.FloatTensor): (3+3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --head_bbox (torch.FloatTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]
                --images (torch.FloatTensor): (2, 3, h, w)
                --pseudo_masks (torch.FloatTensor) : (2, 1, h, w)
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both_bg is True
        """
        vid = item // self._num_samples
        sid = item % self._num_samples
        sample_path = self._sample_files[vid]

        sample = self.load_sample(sample_path, sid)

        sample = self.preprocess(sample)

        return sample

    def load_sample(self, sample_path, sid):
        with h5py.File(sample_path, 'r') as data:
            images = data['imgs'][sid, ...]
            T = data['T'][sid, ...]
            warp = data['warp'][sid, ...]
            fims = data['fims'][sid, ...]
            head_bbox = data['head_bbox'][sid, 1, ...]

            fims_enc = self.map_fn[fims]    # (2, h, w, c)
            fims_enc = np.transpose(fims_enc, axes=(0, 3, 1, 2))   # (2, c, h, w)

            sample = {
                'images': torch.FloatTensor(images),
                'fims': torch.FloatTensor(fims_enc),
                'T': torch.FloatTensor(T),
                'warp': torch.FloatTensor(warp),
                'head_bbox': torch.IntTensor(head_bbox),
            }

            return sample

    def preprocess(self, sample):
        """
        Args:
           sample (dict): items contain
                --images (torch.FloatTensor): (2, 3, h, w)
                --fims (torch.FloatTensor): (2, 3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --warp (torch.FloatTensor): (3, h, w)
                --head_bbox (torch.FloatTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]

        Returns:
            new_sample (dict): items contain
                --src_inputs (torch.FloatTensor): (3+3, h, w)
                --tsf_inputs (torch.FloatTensor): (3+3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --head_bbox (torch.FloatTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]
                --images (torch.FloatTensor): (2, 3, h, w)
                --pseudo_masks (torch.FloatTensor) : (2, 1, h, w)
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both_bg is True
        """
        with torch.no_grad():
            images = sample['images']
            fims = sample['fims']

            # 1. process the bg inputs
            src_fim = fims[0]
            src_img = images[0]
            src_mask = src_fim[None, -1:, :, :]   # (1, h, w)
            src_bg_mask = morph(src_mask, ks=13, mode='dilate')[0]      # bg is 0, front is 1
            src_bg_inputs = torch.cat([src_img * (1 - src_bg_mask), src_bg_mask], dim=0)

            # 2. process the src inputs
            src_crop_mask = morph(src_mask, ks=3, mode='dilate')[0]
            src_fim[-1] = src_crop_mask
            src_inputs = torch.cat([src_img * src_crop_mask, src_fim])

            # 3. process the tsf inputs
            tsf_fim = fims[1]
            tsf_mask = tsf_fim[None, -1:, :, :]     # (1, h, w), bg is 0, front is 1
            tsf_crop_mask = morph(tsf_mask, ks=3, mode='dilate')[0]
            tsf_fim[-1] = tsf_crop_mask
            tsf_inputs = torch.cat([sample['warp'], tsf_fim], dim=0)

            if self.is_both_bg:
                tsf_img = images[1]
                tsf_bg_mask = morph(tsf_mask, ks=13, mode='dilate')[0]  # bg is 0, front is 1
                tsf_bg_inputs = torch.cat([tsf_img * (1 - tsf_bg_mask), tsf_bg_mask], dim=0)
                bg_inputs = torch.stack([src_bg_inputs, tsf_bg_inputs], dim=0)
            else:
                bg_inputs = src_bg_inputs

            # 4. concat pseudo mask
            pseudo_masks = torch.stack([1 - src_crop_mask, 1 - tsf_crop_mask], dim=0)

            new_sample = {
                'images': images,
                'pseudo_masks': pseudo_masks,
                'head_bbox': sample['head_bbox'],
                'T': sample['T'],
                'bg_inputs': bg_inputs,
                'src_inputs': src_inputs,
                'tsf_inputs': tsf_inputs
            }

            return new_sample


class DeepFashionPairSampleDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(DeepFashionPairSampleDataset, self).__init__(opt, is_for_train)
        self._name = 'DeepFashionPairSampleDataset'

        self.is_both_bg = self._opt.is_both
        self.bg_ks = self._opt.bg_ks
        self.ft_ks = self._opt.ft_ks

        # read dataset
        self._read_dataset_paths()

        # prepare mapping function
        self.map_fn = mesh.create_mapping(map_name=opt.map_name, mapping_path=opt.uv_mapping,
                                          contain_bg=True, fill_back=False)
        # prepare head mapping function
        self.head_fn = mesh.create_mapping('head', head_info='pretrains/head.json',
                                           contain_bg=True, fill_back=False)

        self.bg_kernel = torch.ones(1, 1, self.bg_ks, self.bg_ks, dtype=torch.float32)
        self.ft_kernel = torch.ones(1, 1, self.ft_ks, self.ft_ks, dtype=torch.float32)

        # self.bg_kernel = None
        # self.ft_kernel = None

    def _read_dataset_paths(self):
        # /public/liuwen/p300/deep_fashion
        self._root = self._opt.data_dir

        # read pair list
        self._dataset_size = 0
        self._sample_files = []

        pair_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        pair_ids_filepath = os.path.join(self._root, pair_ids_filename)

        pkl_filename = self._opt.train_pkl_folder if self._is_for_train else self._opt.test_pkl_folder
        pkl_dir = os.path.join(self._root, pkl_filename)

        im_dir = os.path.join(self._root, self._opt.images_folder)
        if not os.path.exists(im_dir):
            vid_name = 'train_256' if self._is_for_train else 'test_256'
            im_dir = os.path.join(self._root, vid_name)
        self._read_samples_info(im_dir, pkl_dir, pair_ids_filepath)

    def _read_samples_info(self, im_dir, pkl_dir, pair_ids_filepath):
        """
        Args:
            im_dir:
            pkl_dir:
            pair_ids_filepath:

        Returns:

        """
        # 1. load image pair list
        self.im_pair_list = self._read_pair_list(im_dir, pair_ids_filepath)

        # 2. load pkl file paths
        self.all_pkl_paths = sorted(glob.glob((os.path.join(pkl_dir, '*.pkl'))))

        assert len(self.im_pair_list) == len(self.all_pkl_paths), '{} != {}'.format(
            len(self.im_pair_list), len(self.all_pkl_paths)
        )
        self._dataset_size = len(self.im_pair_list)

    def _read_pair_list(self, im_dir, pair_pkl_path):
        pair_list = load_pickle_file(pair_pkl_path)
        new_pair_list = []

        for i, pairs in enumerate(pair_list):
            src_path = os.path.join(im_dir, pairs[0])
            dst_path = os.path.join(im_dir, pairs[1])

            new_pair_list.append((src_path, dst_path))

        return new_pair_list

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, item):
        """
        Args:
            item (int):  index of self._dataset_size

        Returns:
            new_sample (dict): items contain
                --src_inputs (torch.FloatTensor): (3+3, h, w)
                --tsf_inputs (torch.FloatTensor): (3+3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --head_bbox (torch.IntTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]
                --valid_bbox (torch.FloatTensor): (1), 1.0 valid and 0.0 invalid.
                --images (torch.FloatTensor): (2, 3, h, w)
                --pseudo_masks (torch.FloatTensor) : (2, 1, h, w)
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both_bg is True
        """
        im_pairs = self.im_pair_list[item]
        pkl_path = self.all_pkl_paths[item]

        sample = self.load_sample(im_pairs, pkl_path)
        sample = self.preprocess(sample)

        return sample

    def load_images(self, im_pairs):
        imgs = []
        for im_path in im_pairs:
            img = cv_utils.read_cv2_img(im_path)
            img = cv_utils.transform_img(img, self._opt.image_size, transpose=True)
            img = img * 2 - 1
            imgs.append(img)
        imgs = np.stack(imgs)
        return imgs

    def load_sample(self, im_pairs, pkl_path):
        # 1. load images
        imgs = self.load_images(im_pairs)
        # 2.load pickle data
        pkl_data = load_pickle_file(pkl_path)
        src_fim = pkl_data['from_face_index_map'][:, :, 0]  # (img_size, img_size)
        dst_fim = pkl_data['to_face_index_map'][:, :, 0]  # (img_size, img_size)
        T = pkl_data['T']  # (img_size, img_size, 2)
        fims = np.stack([src_fim, dst_fim], axis=0)

        fims_enc = self.map_fn[fims]  # (2, h, w, c)
        fims_enc = np.transpose(fims_enc, axes=(0, 3, 1, 2))  # (2, c, h, w)

        heads_mask = self.head_fn[fims[1:]]  # (1, h, w, 1)
        heads_mask = np.transpose(heads_mask, axes=(0, 3, 1, 2))    # (1, 1, h, w)
        head_bbox, valid_bbox = cal_head_bbox(heads_mask, factor=1.1)
        head_bbox = head_bbox[0]
        valid_bbox = valid_bbox[0]
        sample = {
            'images': torch.tensor(imgs).float(),
            'fims': torch.tensor(fims_enc).float(),
            'T': torch.tensor(T).float(),
            'head_bbox': torch.tensor(head_bbox).int(),
            'valid_bbox': torch.tensor(valid_bbox).float()
        }

        if 'warp' in pkl_data:
            if len(pkl_data['warp'].shape) == 4:
                sample['warp'] = torch.tensor(pkl_data['warp'][0], dtype=torch.float32)
            else:
                sample['warp'] = torch.tensor(pkl_data['warp'], dtype=torch.float32)
        elif 'warp_R' in pkl_data:
            sample['warp'] = torch.tensor(pkl_data['warp_R'][0], dtype=torch.float32)
        elif 'warp_T' in pkl_data:
            sample['warp'] = torch.tensor(pkl_data['warp_T'][0], dtype=torch.float32)

        return sample

    def preprocess(self, sample):
        """
        Args:
           sample (dict): items contain
                --images (torch.FloatTensor): (2, 3, h, w)
                --fims (torch.FloatTensor): (2, 3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --warp (torch.FloatTensor): (3, h, w)
                --head_bbox (torch.FloatTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]

        Returns:
            new_sample (dict): items contain
                --src_inputs (torch.FloatTensor): (3+3, h, w)
                --tsf_inputs (torch.FloatTensor): (3+3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --head_bbox (torch.FloatTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]
                --images (torch.FloatTensor): (2, 3, h, w)
                --pseudo_masks (torch.FloatTensor) : (2, 1, h, w)
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both_bg is True
        """
        with torch.no_grad():
            images = sample['images']
            fims = sample['fims']

            # 1. process the bg inputs
            src_fim = fims[0]
            src_img = images[0]
            src_mask = src_fim[None, -1:, :, :]   # (1, h, w)
            src_bg_mask = morph(src_mask, ks=self.bg_ks, mode='dilate', kernel=self.bg_kernel)[0]  # bg is 0, front is 1
            src_bg_inputs = torch.cat([src_img * (1 - src_bg_mask), src_bg_mask], dim=0)

            # 2. process the src inputs
            src_crop_mask = morph(src_mask, ks=self.ft_ks, mode='dilate', kernel=self.ft_kernel)[0]
            src_fim[-1] = src_crop_mask
            src_inputs = torch.cat([src_img * src_crop_mask, src_fim])

            # 3. process the tsf inputs
            tsf_fim = fims[1]
            tsf_mask = tsf_fim[None, -1:, :, :]     # (1, h, w), bg is 0, front is 1
            tsf_crop_mask = morph(tsf_mask, ks=self.ft_ks, mode='dilate', kernel=self.ft_kernel)[0]
            tsf_fim[-1] = tsf_crop_mask

            if 'warp' not in sample:
                warp = F.grid_sample(src_img[None], sample['T'][None])[0]
            else:
                warp = sample['warp']
            tsf_inputs = torch.cat([warp, tsf_fim], dim=0)

            if self.is_both_bg:
                tsf_img = images[1]
                tsf_bg_mask = morph(tsf_mask, ks=self.bg_ks, mode='dilate', kernel=self.bg_kernel)[0]  # bg is 0, front is 1
                tsf_bg_inputs = torch.cat([tsf_img * (1 - tsf_bg_mask), tsf_bg_mask], dim=0)
                bg_inputs = torch.stack([src_bg_inputs, tsf_bg_inputs], dim=0)
            else:
                bg_inputs = src_bg_inputs

            # 4. concat pseudo mask
            pseudo_masks = torch.stack([1 - src_crop_mask, 1 - tsf_crop_mask], dim=0)

            new_sample = {
                'images': images,
                'pseudo_masks': pseudo_masks,
                'head_bbox': sample['head_bbox'],
                'valid_bbox': sample['valid_bbox'],
                'T': sample['T'],
                'bg_inputs': bg_inputs,
                'src_inputs': src_inputs,
                'tsf_inputs': tsf_inputs
            }

            return new_sample


class ImperPairSampleDataset(DeepFashionPairSampleDataset):

    def __init__(self, opt, is_for_train):
        super(ImperPairSampleDataset, self).__init__(opt, is_for_train)
        self._name = 'ImperPairSampleDataset'

    def _read_dataset_paths(self):
        # /public/liuwen/p300/deep_fashion
        self._root = self._opt.data_dir

        # read pair list
        self._dataset_size = 0
        self._sample_files = []

        pair_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        pair_ids_filepath = os.path.join(self._root, pair_ids_filename)

        # pkl_filename = 'train_dp_hmr_pairs_divide_idx_15_results' if self._is_for_train \
        #     else 'val_dp_hmr_pairs_divide_idx_15_results'
        #
        pkl_filename = 'train_samples' if self._is_for_train \
            else 'val_samples'
        pkl_dir = os.path.join(self._root, pkl_filename)

        im_dir = os.path.join(self._root, 'motion_transfer')
        self._read_samples_info(im_dir, pkl_dir, pair_ids_filepath)

    def _read_pair_list(self, im_dir, pair_pkl_path):
        pair_list = load_pickle_file(pair_pkl_path)
        new_pair_list = []

        def cvt(im_path):
            im_path_splits = im_path.split('/')
            vid = str(im_path_splits[-2])
            cvt_vid = '/'.join(vid.split('_'))
            new_im_path = os.path.join(im_dir, cvt_vid, im_path_splits[-1])
            # print(im_path, new_im_path)
            return new_im_path

        for i, pairs in enumerate(pair_list):
            src_path = cvt(pairs[0])
            dst_path = cvt(pairs[1])

            new_pair_list.append((src_path, dst_path))

        return new_pair_list

    def load_images(self, im_pairs):
        imgs = []
        for im_path in im_pairs:
            img = cv_utils.read_cv2_img(im_path)
            img = cv_utils.transform_img(img, self._opt.image_size, transpose=True)
            img = img * 2 - 1
            imgs.append(img)
        imgs = np.stack(imgs)
        return imgs

    def load_sample(self, im_pairs, pkl_path):
        # 1. load images
        imgs = self.load_images(im_pairs)
        # 2.load pickle data
        pkl_data = load_pickle_file(pkl_path)
        # src_fim = pkl_data['from_face_index_map'][:, :, 0]  # (img_size, img_size)

        # dst_fim = pkl_data['to_face_index_map'][:, :, 0]  # (img_size, img_size)
        src_fim = pkl_data['src_fim']  # (img_size, img_size)
        dst_fim = pkl_data['dst_fim']  # (img_size, img_size)

        T = pkl_data['T']  # (img_size, img_size, 2)
        fims = np.stack([src_fim, dst_fim], axis=0)

        fims_enc = self.map_fn[fims]  # (2, h, w, c)
        fims_enc = np.transpose(fims_enc, axes=(0, 3, 1, 2))  # (2, c, h, w)

        heads_mask = self.head_fn[fims[1:]]  # (1, h, w, 1)
        heads_mask = np.transpose(heads_mask, axes=(0, 3, 1, 2))    # (1, 1, h, w)
        head_bbox, valid_bbox = cal_head_bbox(heads_mask, factor=1.2)
        head_bbox = head_bbox[0]
        valid_bbox = valid_bbox[0]
        sample = {
            'images': torch.tensor(imgs).float(),
            'fims': torch.tensor(fims_enc).float(),
            'T': torch.tensor(T).float(),
            'head_bbox': torch.tensor(head_bbox).int(),
            'valid_bbox': torch.tensor(valid_bbox).float(),
            'src_bg_mask': torch.tensor(pkl_data['src_bg_mask']).float(),
            'dst_bg_mask': torch.tensor(pkl_data['dst_bg_mask']).float()
        }

        if 'warp' in pkl_data:
            sample['warp'] = torch.tensor(pkl_data['warp'][0]).float()
        elif 'warp_R' in pkl_data:
            sample['warp'] = torch.tensor(pkl_data['warp_R'][0]).float()
        elif 'warp_T' in pkl_data:
            sample['warp'] = torch.tensor(pkl_data['warp_T'][0]).float()

        return sample

    def preprocess(self, sample):
        """
        Args:
           sample (dict): items contain
                --images (torch.FloatTensor): (2, 3, h, w)
                --fims (torch.FloatTensor): (2, 3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --head_bbox (torch.FloatTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]

        Returns:
            new_sample (dict): items contain
                --src_inputs (torch.FloatTensor): (3+3, h, w)
                --tsf_inputs (torch.FloatTensor): (3+3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --head_bbox (torch.IntTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]
                --vali_bbox (torch.FloatTensor): (1), hear 4 = [lt_x, lt_y, rt_x, rt_y]
                --images (torch.FloatTensor): (2, 3, h, w)
                --pseudo_masks (torch.FloatTensor) : (2, 1, h, w)
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both_bg is True
        """
        with torch.no_grad():
            images = sample['images']
            fims = sample['fims']
            # 1. process the bg inputs
            src_fim = fims[0]
            src_img = images[0]
            init_src_bg_mask = sample['src_bg_mask'][None, None, :, :]   # (1, 1, h, w)
            src_bg_mask = morph(init_src_bg_mask, ks=self.bg_ks, mode='erode', kernel=self.bg_kernel)[0] # bg is 1, front is 0
            src_bg_inputs = torch.cat([src_img * src_bg_mask, 1 - src_bg_mask], dim=0)

            # 2. process the src inputs
            src_crop_mask = morph(init_src_bg_mask, ks=self.ft_ks, mode='erode', kernel=self.ft_kernel)[0]
            src_inputs = torch.cat([src_img * (1 - src_crop_mask), src_fim])

            # 3. process the tsf inputs
            tsf_fim = fims[1]
            init_tsf_bg_mask = sample['dst_bg_mask'][None, None, :, :]
            tsf_crop_mask = morph(init_tsf_bg_mask, ks=self.ft_ks, mode='erode', kernel=self.ft_kernel)[0]

            if 'warp' not in sample:
                warp = F.grid_sample(src_img[None], sample['T'][None])[0]
            else:
                warp = sample['warp']

            tsf_inputs = torch.cat([warp, tsf_fim], dim=0)

            if self.is_both_bg:
                tsf_img = images[1]
                tsf_bg_mask = morph(init_tsf_bg_mask, ks=self.bg_ks, mode='erode', kernel=self.bg_kernel)[0]# bg is 1, front is 0
                tsf_bg_inputs = torch.cat([tsf_img * tsf_bg_mask, 1 - tsf_bg_mask], dim=0)
                bg_inputs = torch.stack([src_bg_inputs, tsf_bg_inputs], dim=0)
            else:
                bg_inputs = src_bg_inputs

            # 4. concat pseudo mask
            pseudo_masks = torch.stack([src_crop_mask, tsf_crop_mask], dim=0)

            new_sample = {
                'images': images,
                'pseudo_masks': pseudo_masks,
                'head_bbox': sample['head_bbox'],
                'valid_bbox': sample['valid_bbox'],
                'T': sample['T'],
                'bg_inputs': bg_inputs,
                'src_inputs': src_inputs,
                'tsf_inputs': tsf_inputs
            }

            return new_sample


class DeepFashionHmrPairSampleDataset(DeepFashionPairSampleDataset):

    def __init__(self, opt, is_for_train):
        super(DeepFashionHmrPairSampleDataset, self).__init__(opt, is_for_train)
        self._name = 'DeepFashionHmrPairSampleDataset'

    def _read_dataset_paths(self):
        # /public/liuwen/p300/deep_fashion
        self._root = self._opt.data_dir

        # read pair list
        self._dataset_size = 0
        self._sample_files = []

        pair_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        pair_ids_filepath = os.path.join(self._root, pair_ids_filename)

        pkl_filename = 'train_256_v2_max_bbox_dp_hmr_pairs_results' if self._is_for_train \
            else 'test_256_v2_max_bbox_dp_hmr_pairs_results'
        pkl_dir = os.path.join(self._root, pkl_filename)

        im_dir = os.path.join(self._root, 'img_256')
        self._read_samples_info(im_dir, pkl_dir, pair_ids_filepath)

    def load_images(self, im_pairs):
        imgs = []
        for im_path in im_pairs:
            img = cv_utils.read_cv2_img(im_path)
            img = cv_utils.transform_img(img, self._opt.image_size, transpose=True)
            img = img * 2 - 1
            imgs.append(img)
        imgs = np.stack(imgs)
        return imgs

    def load_sample(self, im_pairs, pkl_path):
        # 1. load images
        imgs = self.load_images(im_pairs)
        # 2.load pickle data
        pkl_data = load_pickle_file(pkl_path)

        # import ipdb
        # ipdb.set_trace()

        src_fim = pkl_data['from_face_index_map']  # (img_size, img_size)
        dst_fim = pkl_data['to_face_index_map']    # (img_size, img_size)

        T = pkl_data['T']  # (img_size, img_size, 2)
        fims = np.stack([src_fim, dst_fim], axis=0)

        fims_enc = self.map_fn[fims]  # (2, h, w, c)
        fims_enc = np.transpose(fims_enc, axes=(0, 3, 1, 2))  # (2, c, h, w)

        heads_mask = self.head_fn[fims[1:]]  # (1, h, w, 1)
        heads_mask = np.transpose(heads_mask, axes=(0, 3, 1, 2))    # (1, 1, h, w)
        head_bbox, valid_bbox = cal_head_bbox(heads_mask, factor=1.2)
        head_bbox = head_bbox[0]
        valid_bbox = valid_bbox[0]
        sample = {
            'images': torch.tensor(imgs).float(),
            'fims': torch.tensor(fims_enc).float(),
            'T': torch.tensor(T).float(),
            'head_bbox': torch.tensor(head_bbox).int(),
            'valid_bbox': torch.tensor(valid_bbox).float()
        }

        if 'warp' in pkl_data:
            sample['warp'] = torch.tensor(pkl_data['warp'][0]).float()
        elif 'warp_R' in pkl_data:
            sample['warp'] = torch.tensor(pkl_data['warp_R'][0]).float()
        elif 'warp_T' in pkl_data:
            sample['warp'] = torch.tensor(pkl_data['warp_T'][0]).float()

        return sample


def create_imper_fashion_pair_dataset(opt, is_for_train):
    from torch.utils.data import ConcatDataset
    import copy
    # deep fashion
    df_opt = opt
    df_dataset = DeepFashionPairSampleDataset(df_opt, is_for_train)
    print('deep fashion dataset')
    # imper option
    imper_opt = copy.copy(opt)

    imper_opt.bg_ks = 15
    imper_opt.ft_ks = 3
    imper_opt.data_dir = '/public/liuwen/p300/ImPer'
    imper_opt.train_ids_file = 'pairs_train.pkl'
    imper_opt.test_ids_file = 'pairs_val.pkl'
    imper_opt.train_pkl_folder = 'train_pairs_results'
    imper_opt.test_pkl_folder = 'val_pairs_results'
    imper_opt.val_pkl_folder = 'val_pairs_results'
    imper_opt.images_folder = 'motion_transfer'

    imper_dataset = DeepFashionPairSampleDataset(imper_opt, is_for_train)
    print('imper dataset')
    dataset = ConcatDataset([df_dataset, imper_dataset])
    dataset.__setattr__('name', 'DeepFashion_ImPer')
    return dataset


if __name__ == '__main__':
    from utils.visualizer.demo_visualizer import MotionImitationVisualizer

    def debug_miplace_dataset(opts):
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

    def debug_ImPerSampleDataset(opts):
        import time
        imper_sampler = ImPerSampleDataset(opts, is_for_train=True)
        print(len(imper_sampler))

        visualizer = MotionImitationVisualizer(env='process', ip='http://10.19.129.76', port=10086)

        for i, sample in enumerate(imper_sampler):
            visualizer.vis_named_img('images', sample['images'])
            visualizer.vis_named_img('pseudo_masks', sample['pseudo_masks'])
            visualizer.vis_named_img('warp', sample['tsf_inputs'][None, 0:3, ...])
            visualizer.vis_named_img('src_fim', sample['src_inputs'][None, 3:, ...])
            visualizer.vis_named_img('tsf_fim', sample['tsf_inputs'][None, 3:, ...])
            visualizer.vis_named_img('bg_input', sample['bg_inputs'][None, 0:3, ...])
            visualizer.vis_named_img('src_input', sample['src_inputs'][None, 0:3, ...])

            T = sample['T']
            T[T == -1.0] = -2.0
            T = T[None, ...]
            new_warp = F.grid_sample(sample['images'][0:1], T)
            visualizer.vis_named_img('new_warp', new_warp, denormalize=True)
            print(i, sample['head_bbox'].shape)
            # time.sleep(1)

    def debug_DeepFashionPairSampleDataset():
        import time

        class Object(object):
            def __init__(self):
                # self.data_dir = '/public/liuwen/p300/human_pose/processed'
                # self.data_dir = '/public/liuwen/p300/deep_fashion'
                self.data_dir = '/public/liuwen/p300/ImPer'
                self.sample_dir = '/public/impersonator_dataset_v2/frame_samples/imper'
                self.place_dir = '/home/piaozx/liuwen/p300/places365_standard'
                self.images_folder = 'motion_transfer_HD'
                self.smpls_folder = 'motion_transfer_smpl'
                # self.train_ids_file = 'MI_train.txt'
                # self.test_ids_file = 'MI_val.txt'
                self.train_ids_file = 'pairs_train.pkl'
                self.test_ids_file = 'pairs_test.pkl'
                self.image_size = 256
                self.time_step = 10
                self.intervals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                self.map_name = 'uv_seg'
                self.uv_mapping = 'pretrains/mapper.txt'
                self.is_both = True
                self.is_both = True
                self.bg_ks = 13
                self.ft_ks = 3

        opts = Object()

        imper_sampler = DeepFashionPairSampleDataset(opts, is_for_train=True)
        length = len(imper_sampler)

        visualizer = MotionImitationVisualizer(env='df_loader', ip='http://10.19.129.76', port=10086)

        start = time.time()
        for i in range(length):
            sample = imper_sampler[i]
            visualizer.vis_named_img('images', sample['images'])
            visualizer.vis_named_img('pseudo_masks', sample['pseudo_masks'])
            visualizer.vis_named_img('warp', sample['tsf_inputs'][None, 0:3, ...])
            visualizer.vis_named_img('src_fim', sample['src_inputs'][None, 3:, ...])
            visualizer.vis_named_img('tsf_fim', sample['tsf_inputs'][None, 3:, ...])
            visualizer.vis_named_img('src_input', sample['src_inputs'][None, 0:3, ...])
            visualizer.vis_named_img('bg_input', sample['bg_inputs'][:, 0:3, ...])

            head_bbox = sample['head_bbox'].numpy()
            dst_bbox = head_bbox
            visualizer.vis_named_img('dst_head', sample['images'][1:2, :, dst_bbox[1]: dst_bbox[3], dst_bbox[0]: dst_bbox[2]])

            print(i, sample['head_bbox'].shape)
            # time.sleep(1)
        print(time.time() - start)

    def debug_DeepFashionHmrPairSampleDataset():
        import time

        class Object(object):
            def __init__(self):
                # self.data_dir = '/public/liuwen/p300/human_pose/processed'
                self.data_dir = '/public/deep_fashion/intrinsic'
                self.sample_dir = '/public/impersonator_dataset_v2/frame_samples/imper'
                self.place_dir = '/home/piaozx/liuwen/p300/places365_standard'
                self.images_folder = 'motion_transfer_HD'
                self.smpls_folder = 'motion_transfer_smpl'
                # self.train_ids_file = 'MI_train.txt'
                # self.test_ids_file = 'MI_val.txt'
                self.train_ids_file = 'pairs_train.pkl'
                self.test_ids_file = 'pairs_test.pkl'
                self.image_size = 256
                self.time_step = 10
                self.intervals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                self.map_name = 'uv_seg'
                self.uv_mapping = 'pretrains/mapper.txt'
                self.is_both = True
                self.is_both = True
                self.bg_ks = 21
                self.ft_ks = 9

        opts = Object()

        imper_sampler = DeepFashionHmrPairSampleDataset(opts, is_for_train=True)
        length = len(imper_sampler)

        visualizer = MotionImitationVisualizer(env='df_loader', ip='http://10.19.129.76', port=10086)

        start = time.time()
        for i in range(length):
            sample = imper_sampler[i]
            visualizer.vis_named_img('images', sample['images'])
            visualizer.vis_named_img('pseudo_masks', sample['pseudo_masks'])
            visualizer.vis_named_img('warp', sample['tsf_inputs'][None, 0:3, ...])
            visualizer.vis_named_img('src_fim', sample['src_inputs'][None, 3:, ...])
            visualizer.vis_named_img('tsf_fim', sample['tsf_inputs'][None, 3:, ...])
            visualizer.vis_named_img('src_input', sample['src_inputs'][None, 0:3, ...])
            visualizer.vis_named_img('bg_input', sample['bg_inputs'][:, 0:3, ...])

            head_bbox = sample['head_bbox'].numpy()
            dst_bbox = head_bbox
            visualizer.vis_named_img('dst_head', sample['images'][1:2, :, dst_bbox[1]: dst_bbox[3], dst_bbox[0]: dst_bbox[2]])

            print(i, sample['head_bbox'].shape)
            time.sleep(1)
        print(time.time() - start)

    def debug_ImperPairSampleDataset():
        import time

        class Object(object):
            def __init__(self):
                self.data_dir = '/public/liuwen/p300/ImPer'
                # self.data_dir = '/public/liuwen/p300/deep_fashion'
                self.sample_dir = '/public/impersonator_dataset_v2/frame_samples/imper'
                self.place_dir = '/home/piaozx/liuwen/p300/places365_standard'
                self.images_folder = 'motion_transfer_HD'
                self.smpls_folder = 'motion_transfer_smpl'
                # self.train_ids_file = 'MI_train.txt'
                # self.test_ids_file = 'MI_val.txt'
                self.train_ids_file = 'pairs_train_divide_idx_15.pkl'
                self.test_ids_file = 'pairs_val_divide_idx_15.pkl'
                self.image_size = 256
                self.time_step = 10
                self.intervals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                self.map_name = 'uv_seg'
                self.uv_mapping = 'pretrains/mapper.txt'
                self.is_both = True
                self.bg_ks = 9
                self.ft_ks = 3

        opts = Object()

        imper_sampler = ImperPairSampleDataset(opts, is_for_train=True)
        length = len(imper_sampler)

        visualizer = MotionImitationVisualizer(env='df_loader', ip='http://10.19.129.76', port=10086)

        start = time.time()
        for i in range(length):
            sample = imper_sampler[i]

            new_warp_img = F.grid_sample(sample['images'][0:1], sample['T'][None])

            visualizer.vis_named_img('images', sample['images'])
            visualizer.vis_named_img('new_warp_img', new_warp_img)
            visualizer.vis_named_img('pseudo_masks', sample['pseudo_masks'])
            visualizer.vis_named_img('warp', sample['tsf_inputs'][None, 0:3, ...])
            visualizer.vis_named_img('src_fim', sample['src_inputs'][None, 3:, ...])
            visualizer.vis_named_img('tsf_fim', sample['tsf_inputs'][None, 3:, ...])
            visualizer.vis_named_img('src_input', sample['src_inputs'][None, 0:3, ...])
            visualizer.vis_named_img('bg_input', sample['bg_inputs'][:, 0:3, ...])
            head_bbox = sample['head_bbox'].numpy()
            dst_bbox = head_bbox
            visualizer.vis_named_img('dst_head', sample['images'][1:2, :, dst_bbox[1]: dst_bbox[3], dst_bbox[0]: dst_bbox[2]])

            print(i, sample['head_bbox'].shape)
            # time.sleep(1)
        print(time.time() - start)

    def debug_df_imper():
        import time
        import torch.utils.data

        class Object(object):
            def __init__(self):
                self.data_dir = '/public/deep_fashion/intrinsic'
                self.sample_dir = '/public/impersonator_dataset_v2/frame_samples/imper'
                self.place_dir = '/home/piaozx/liuwen/p300/places365_standard'
                self.images_folder = 'img_256'
                self.smpls_folder = 'motion_transfer_smpl'
                self.train_pkl_folder = 'train_256_v2_max_bbox_dp_hmr_pairs_results'
                self.test_pkl_folder = 'test_256_v2_max_bbox_dp_hmr_pairs_results'
                self.train_ids_file = 'pairs_train.pkl'
                self.test_ids_file = 'pairs_test.pkl'
                self.image_size = 256
                self.time_step = 10
                self.intervals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                self.map_name = 'uv_seg'
                self.uv_mapping = 'pretrains/mapper.txt'
                self.is_both = True
                self.bg_ks = 21
                self.ft_ks = 9

        opts = Object()

        imper_dataset = create_imper_fashion_pair_dataset(opts, is_for_train=True)
        imper_sampler = torch.utils.data.dataloader.DataLoader(
            dataset=imper_dataset,
            batch_size=2,
            num_workers=2,
            shuffle=True,
            drop_last=True
        )
        length = len(imper_sampler)

        visualizer = MotionImitationVisualizer(env='df_loader', ip='http://10.19.129.76', port=10086)

        start = time.time()
        for i, sample in enumerate(imper_sampler):
            visualizer.vis_named_img('src_img', sample['images'][:, 0])
            visualizer.vis_named_img('tsf_img', sample['images'][:, 1])
            visualizer.vis_named_img('src_mask', sample['pseudo_masks'][:, 0])
            visualizer.vis_named_img('tsf_mask', sample['pseudo_masks'][:, 1])
            visualizer.vis_named_img('warp', sample['tsf_inputs'][:, 0:3, ...])
            visualizer.vis_named_img('src_fim', sample['src_inputs'][:, 3:, ...])
            visualizer.vis_named_img('tsf_fim', sample['tsf_inputs'][:, 3:, ...])
            visualizer.vis_named_img('src_input', sample['src_inputs'][:, 0:3, ...])
            # visualizer.vis_named_img('bg_input', sample['bg_inputs'][:, 0:3, ...])

            # head_bbox = sample['head_bbox'].numpy()
            # dst_bbox = head_bbox
            # visualizer.vis_named_img('dst_head', sample['images'][1:2, :, dst_bbox[1]: dst_bbox[3], dst_bbox[0]: dst_bbox[2]])
            #
            # print(i, sample['head_bbox'].shape)
            time.sleep(1)
        print(time.time() - start)

    # debug_DeepFashionHmrPairSampleDataset()
    # debug_DeepFashionPairSampleDataset()
    # debug_ImperPairSampleDataset()
    debug_df_imper()
