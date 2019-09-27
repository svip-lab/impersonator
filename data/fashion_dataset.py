import glob
import os.path
import torch
import torch.nn.functional as F
from data.dataset import DatasetBase
import numpy as np
from utils import cv_utils
from utils.util import load_pickle_file, morph, cal_mask_bbox

import utils.mesh as mesh


class FashionPairDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(FashionPairDataset, self).__init__(opt, is_for_train)
        self._name = 'FashionPairDataset'

        self.use_src_bg = False
        self.bg_ks = 21
        self.ft_ks = 7

        # read dataset
        self._read_dataset_paths()

        # prepare mapping function
        self.map_fn = mesh.create_mapping(map_name=opt.map_name, mapping_path=opt.uv_mapping,
                                          contain_bg=True, fill_back=False)
        # prepare head mapping function
        self.head_fn = mesh.create_mapping('head', head_info='assets/pretrains/head.json',
                                           contain_bg=True, fill_back=False)

        self.bg_kernel = torch.ones(1, 1, self.bg_ks, self.bg_ks, dtype=torch.float32)
        self.ft_kernel = torch.ones(1, 1, self.ft_ks, self.ft_ks, dtype=torch.float32)

    def _read_dataset_paths(self):
        # /public/liuwen/p300/deep_fashion
        self._root = self._opt.fashion_dir

        # read pair list
        self._dataset_size = 0
        self._sample_files = []

        pair_ids_filename = 'pairs_train.pkl' if self._is_for_train else 'pairs_test.pkl'
        pair_ids_filepath = os.path.join(self._root, pair_ids_filename)

        pkl_filename = 'train_256_v2_max_bbox_dp_hmr_pairs_results' if self._is_for_train \
            else 'test_256_v2_max_bbox_dp_hmr_pairs_results'
        pkl_dir = os.path.join(self._root, pkl_filename)

        im_dir = os.path.join(self._root, 'img_256')
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
        im_pair_list = load_pickle_file(pair_ids_filepath)

        # 2. load pkl file paths
        all_pkl_paths = sorted(glob.glob((os.path.join(pkl_dir, '*.pkl'))))

        # 3. filters the source image is not front
        self.im_pair_list = []
        self.all_pkl_paths = []

        for pairs, pkl_path in zip(im_pair_list, all_pkl_paths):
            src_path = os.path.join(im_dir, pairs[0])
            dst_path = os.path.join(im_dir, pairs[1])

            if 'side' in src_path or 'back' in src_path:
                continue

            src_path = os.path.join(im_dir, src_path)
            dst_path = os.path.join(im_dir, dst_path)

            self.im_pair_list.append((src_path, dst_path))
            self.all_pkl_paths.append(pkl_path)

        assert len(self.im_pair_list) == len(self.all_pkl_paths), '{} != {}'.format(
            len(self.im_pair_list), len(self.all_pkl_paths)
        )
        self._dataset_size = len(self.im_pair_list)

        del im_pair_list
        del all_pkl_paths

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
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both is True
        """

        item = item % self._dataset_size
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
        head_bbox, _ = cal_mask_bbox(heads_mask, factor=1.05)
        body_bbox, _ = cal_mask_bbox(1 - fims_enc[1:, -1:], factor=1.2)

        # print(head_bbox.shape, valid_bbox.shape)
        sample = {
            'images': torch.tensor(imgs).float(),
            'fims': torch.tensor(fims_enc).float(),
            'T': torch.tensor(T).float(),
            'head_bbox': torch.tensor(head_bbox[0]).long(),
            'body_bbox': torch.tensor(body_bbox[0]).long()
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

    @torch.no_grad()
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
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both is True
        """
        images = sample['images']
        fims = sample['fims']

        # 1. process the bg inputs
        src_fim = fims[0]
        src_img = images[0]
        src_mask = src_fim[None, -1:, :, :]   # (1, h, w)

        # 2. process the src inputs
        src_crop_mask = morph(src_mask, ks=self.ft_ks, mode='erode', kernel=self.ft_kernel)[0]
        src_inputs = torch.cat([src_img * (1 - src_crop_mask), src_fim])

        # 3. process the tsf inputs
        tsf_fim = fims[1]
        tsf_mask = tsf_fim[None, -1:, :, :]     # (1, h, w), bg is 0, front is 1
        tsf_crop_mask = morph(tsf_mask, ks=self.ft_ks, mode='erode', kernel=self.ft_kernel)[0]

        if 'warp' not in sample:
            warp = F.grid_sample(src_img[None], sample['T'][None])[0]
        else:
            warp = sample['warp']

        tsf_inputs = torch.cat([warp, tsf_fim], dim=0)

        if self.use_src_bg:
            src_bg_mask = morph(src_mask, ks=self.bg_ks, mode='erode', kernel=self.bg_kernel)[0]  # bg is 0, front is 1
            bg_inputs = torch.cat([src_img * src_bg_mask, src_bg_mask], dim=0)
        else:
            tsf_img = images[1]
            tsf_bg_mask = morph(tsf_mask, ks=self.bg_ks, mode='erode', kernel=self.bg_kernel)[0]
            bg_inputs = torch.cat([tsf_img * tsf_bg_mask, tsf_bg_mask], dim=0)

        # 4. concat pseudo mask
        pseudo_masks = torch.stack([src_crop_mask, tsf_crop_mask], dim=0)

        new_sample = {
            'images': images,
            'pseudo_masks': pseudo_masks,
            'head_bbox': sample['head_bbox'],
            'body_bbox': sample['body_bbox'],
            'bg_inputs': bg_inputs,
            'src_inputs': src_inputs,
            'tsf_inputs': tsf_inputs,
            'T': sample['T'],
        }

        return new_sample
