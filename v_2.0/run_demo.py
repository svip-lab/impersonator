import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
import networks
from models.models import ModelsFactory
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from data.dataset import DatasetBase
from options.eval_options import EvalOptions
from utils.visualizer.demo_visualizer import MotionImitationVisualizer
from utils.util import load_pickle_file, morph, cal_head_bbox
from utils import cv_utils
import networks.bodymesh.mesh as mesh


def save_batch_images(save_template, batch_images, count):
    bs = batch_images.shape[0]

    for i in range(bs):
        image = batch_images[i]
        image = (image + 1.0) / 2 * 255
        image = image.astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_path = save_template.format(count + i)
        cv2.imwrite(image_path, image)
        print(image_path)


def save_results(out_dir, src_images, ref_images, tsf_images, count):
    src_images = src_images.numpy()
    ref_images = ref_images.numpy()
    tsf_images = tsf_images.numpy()

    src_save_temp = out_dir + '/src_{:0>8}.jpg'
    ref_save_temp = out_dir + '/ref_{:0>8}.jpg'
    tsf_save_temp = out_dir + '/tsf_{:0>8}.jpg'

    save_batch_images(src_save_temp, batch_images=src_images, count=count)
    save_batch_images(ref_save_temp, batch_images=ref_images, count=count)
    save_batch_images(tsf_save_temp, batch_images=tsf_images, count=count)


class DemoDataset(DatasetBase):
    def __init__(self, opt, is_for_train=False):
        super(DemoDataset, self).__init__(opt, is_for_train)
        self._name = 'DemoDataset'

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

        # source image dir
        im_dir = os.path.join(self._root, 'all_img')
        pkl_dir = os.path.join(self._root, 'pairs_result')

        self._read_samples_info(im_dir, pkl_dir)

    def _read_samples_info(self, im_dir, pkl_dir):
        """
        Args:
            im_dir:
            pkl_dir:

        Returns:

        """
        # 1. load image pair and pickle list
        self.im_pair_list, self.all_pkl_paths = self._read_pair_list(im_dir, pkl_dir)

        assert len(self.im_pair_list) == len(self.all_pkl_paths), '{} != {}'.format(
            len(self.im_pair_list), len(self.all_pkl_paths)
        )
        self._dataset_size = len(self.im_pair_list)

    def _read_pair_list(self, im_dir, pkl_dir):
        im_pair_list = []
        pkl_list = []

        for img_name in os.listdir(pkl_dir):
            img_dir_path = os.path.join(pkl_dir, img_name)

            for pkl_filename in os.listdir(img_dir_path):
                pkl_name = str(pkl_filename.split('.pkl')[0])
                from_im_name, to_im_name = pkl_name.split('_to_')

                im_pair_list.append((os.path.join(im_dir, from_im_name),
                                     os.path.join(im_dir, to_im_name)))

                pkl_list.append(os.path.join(pkl_dir, img_name, pkl_filename))

        return im_pair_list, pkl_list

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
        heads_mask = np.transpose(heads_mask, axes=(0, 3, 1, 2))  # (1, 1, h, w)
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
            src_mask = src_fim[None, -1:, :, :]  # (1, h, w)
            src_bg_mask = morph(src_mask, ks=self.bg_ks, mode='dilate', kernel=self.bg_kernel)[
                0]  # bg is 0, front is 1
            src_bg_inputs = torch.cat([src_img * (1 - src_bg_mask), src_bg_mask], dim=0)

            # 2. process the src inputs
            src_crop_mask = morph(src_mask, ks=self.ft_ks, mode='dilate', kernel=self.ft_kernel)[0]
            src_fim[-1] = src_crop_mask
            src_inputs = torch.cat([src_img * src_crop_mask, src_fim])

            # 3. process the tsf inputs
            tsf_fim = fims[1]
            tsf_mask = tsf_fim[None, -1:, :, :]  # (1, h, w), bg is 0, front is 1
            tsf_crop_mask = morph(tsf_mask, ks=self.ft_ks, mode='dilate', kernel=self.ft_kernel)[0]
            tsf_fim[-1] = tsf_crop_mask

            if 'warp' not in sample:
                warp = F.grid_sample(src_img[None], sample['T'][None])[0]
            else:
                warp = sample['warp']
            tsf_inputs = torch.cat([warp, tsf_fim], dim=0)

            if self.is_both_bg:
                tsf_img = images[1]
                tsf_bg_mask = morph(tsf_mask, ks=self.bg_ks, mode='dilate', kernel=self.bg_kernel)[
                    0]  # bg is 0, front is 1
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


if __name__ == "__main__":
    import time
    opt = EvalOptions().parse()
    demo_dataset = DemoDataset(opt, is_for_train=False)

    demo_loader = torch.utils.data.DataLoader(
        demo_dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=2,
        drop_last=False
    )
    # set imitator
    imitator = ModelsFactory.get_by_name(opt.model, opt)
    imitator.set_eval()
    # imitator.set_G_train()

    if opt.visual:
        visualizer = MotionImitationVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    # pair
    out_dir = opt.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    count = 0
    for i, sample in enumerate(demo_loader):
        imitator.set_input(sample)

        with torch.no_grad():
            fake_tsf_imgs, fake_imgs, fake_tsf_color = imitator.forward()
            fake_tsf_imgs = fake_tsf_imgs.cpu()
            bs = fake_tsf_imgs.shape[0]

            save_results(out_dir, sample['images'][:, 0, ...], sample['images'][:, 1, ...], fake_tsf_imgs, count)
            count += bs

        print(i)
        if visualizer is not None:
            visualizer.vis_named_img('src_img', sample['images'][:, 0, ...])
            visualizer.vis_named_img('ref_img', sample['images'][:, 1, ...])
            visualizer.vis_named_img('tsf_img', fake_tsf_imgs)

            time.sleep(3)


