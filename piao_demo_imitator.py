import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
import os
import glob

from data.dataset import DatasetBase
from models.models import ModelsFactory
from options.test_options import TestOptions
from utils.demo_visualizer import MotionImitationVisualizer
from utils.util import load_pickle_file, write_pickle_file, mkdirs, mkdir, morph, cal_head_bbox
import utils.cv_utils as cv_utils
import utils.mesh as mesh
import pickle
from utils.video import make_video
from scipy.spatial.transform import Rotation as R


@torch.no_grad()
def write_pair_info(src_info, tsf_info, out_file, imitator, only_vis):
    """
    Args:
        src_info:
        tsf_info:
        out_file:
        imitator:
    Returns:

    """
    pair_data = dict()

    pair_data['from_face_index_map'] = src_info['fim'][0][:, :, None].cpu().numpy()
    pair_data['to_face_index_map'] = tsf_info['fim'][0][:, :, None].cpu().numpy()
    pair_data['T'] = tsf_info['T'][0].cpu().numpy()
    pair_data['warp'] = tsf_info['tsf_img'][0].cpu().numpy()
    pair_data['smpls'] = torch.cat([src_info['theta'], tsf_info['theta']], dim=0).cpu().numpy()
    pair_data['j2d'] = torch.cat([src_info['j2d'], tsf_info['j2d']], dim=0).cpu().numpy()

    tsf_f2verts, tsf_fim, tsf_wim = imitator.render.render_fim_wim(tsf_info['cam'], tsf_info['verts'])
    tsf_p2verts = tsf_f2verts[:, :, :, 0:2]
    tsf_p2verts[:, :, :, 1] *= -1

    T_cycle = imitator.render.cal_bc_transform(tsf_p2verts, src_info['fim'], src_info['wim'])
    pair_data['T_cycle'] = T_cycle[0].cpu().numpy()

    # back_face_ids = mesh.get_part_face_ids(part_type='head_back')
    # tsf_p2verts[:, back_face_ids] = -2
    # T_cycle_vis = imitator.render.cal_bc_transform(tsf_p2verts, src_info['fim'], src_info['wim'])
    # pair_data['T_cycle_vis'] = T_cycle_vis[0].cpu().numpy()

    # for key, val in pair_data.items():
    #     print(key, val.shape)

    write_pickle_file(out_file, pair_data)


def scan_tgt_paths(tgt_path, itv=20):
    if os.path.isdir(tgt_path):
        all_tgt_paths = glob.glob(os.path.join(tgt_path, '*'))
        all_tgt_paths.sort()
        all_tgt_paths = all_tgt_paths[::itv]
    else:
        all_tgt_paths = [tgt_path]

    return all_tgt_paths


def meta_imitate(opt, imitator, prior_tgt_path, save_imgs=True, visualizer=None):
    src_path = opt.src_path

    all_tgt_paths = scan_tgt_paths(prior_tgt_path, itv=40)
    output_dir = opt.output_dir

    out_img_dir, out_pair_dir = mkdirs([os.path.join(output_dir, 'imgs'), os.path.join(output_dir, 'pairs')])

    img_pair_list = []

    for t in tqdm(range(len(all_tgt_paths))):
        tgt_path = all_tgt_paths[t]
        preds = imitator.inference([tgt_path], visualizer=visualizer, cam_strategy=opt.cam_strategy, verbose=False)

        tgt_name = os.path.split(tgt_path)[-1]
        out_path = os.path.join(out_img_dir, 'pred_' + tgt_name)

        if save_imgs:
            cv_utils.save_cv2_img(preds[0], out_path, normalize=True)
            write_pair_info(imitator.src_info, imitator.tsf_info,
                            os.path.join(out_pair_dir, '{:0>8}.pkl'.format(t)), imitator=imitator,
                            only_vis=opt.only_vis)

            img_pair_list.append((src_path, tgt_path))

    if save_imgs:
        write_pickle_file(os.path.join(output_dir, 'pairs_meta.pkl'), img_pair_list)


class PairSampleDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(PairSampleDataset, self).__init__(opt, is_for_train)
        self._name = 'PairSampleDataset'

        self.is_both = self._opt.is_both
        self.bg_ks = self._opt.bg_ks
        self.ft_ks = self._opt.ft_ks

        # read dataset
        self._read_dataset_paths()

        # prepare mapping function
        self.map_fn = mesh.create_mapping(map_name=opt.map_name, mapping_path=opt.uv_mapping,
                                          contain_bg=True, fill_back=False)
        # prepare head mapping function
        # self.head_fn = mesh.create_mapping('head', head_info='pretrains/head.json',
        #                                    contain_bg=True, fill_back=False)

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

        pair_ids_filepath = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        # pair_ids_filepath = os.path.join(self._root, pair_ids_filename)
        # pair_ids_filepath = os.path.join(self._root, pair_ids_filename)

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
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both is True
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

        sample = {
            'images': torch.tensor(imgs).float(),
            'src_fim': torch.tensor(src_fim).float(),
            'tsf_fim': torch.tensor(dst_fim).float(),
            'fims': torch.tensor(fims_enc).float(),
            'T': torch.tensor(T).float(),
            'j2d': torch.tensor(pkl_data['j2d']).float()
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

        if 'T_cycle' in pkl_data:
            sample['T_cycle'] = torch.tensor(pkl_data['T_cycle']).float()

        if 'T_cycle_vis' in pkl_data:
            sample['T_cycle_vis'] = torch.tensor(pkl_data['T_cycle_vis']).float()

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
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both is True
        """
        with torch.no_grad():
            images = sample['images']
            fims = sample['fims']

            # 1. process the bg inputs
            src_fim = fims[0]
            src_img = images[0]
            src_mask = src_fim[None, -1:, :, :]  # (1, h, w)
            src_bg_mask = morph(src_mask, ks=self.bg_ks, mode='erode', kernel=self.bg_kernel)[0]  # bg is 0, front is 1
            src_bg_inputs = torch.cat([src_img * src_bg_mask, src_bg_mask], dim=0)

            # 2. process the src inputs
            src_crop_mask = morph(src_mask, ks=self.ft_ks, mode='erode', kernel=self.ft_kernel)[0]
            src_inputs = torch.cat([src_img * (1 - src_crop_mask), src_fim])

            # 3. process the tsf inputs
            tsf_fim = fims[1]
            tsf_mask = tsf_fim[None, -1:, :, :]  # (1, h, w), bg is 0, front is 1
            tsf_crop_mask = morph(tsf_mask, ks=self.ft_ks, mode='erode', kernel=self.ft_kernel)[0]

            if 'warp' not in sample:
                warp = F.grid_sample(src_img[None], sample['T'][None])[0]
            else:
                warp = sample['warp']
            tsf_inputs = torch.cat([warp, tsf_fim], dim=0)

            if self.is_both:
                tsf_img = images[1]
                tsf_bg_mask = morph(tsf_mask, ks=self.bg_ks, mode='dilate', kernel=self.bg_kernel)[0]  # bg is 0, front is 1
                tsf_bg_inputs = torch.cat([tsf_img * (1 - tsf_bg_mask), tsf_bg_mask], dim=0)
                bg_inputs = torch.stack([src_bg_inputs, tsf_bg_inputs], dim=0)
            else:
                bg_inputs = src_bg_inputs

            # 4. concat pseudo mask
            pseudo_masks = torch.stack([src_crop_mask, tsf_crop_mask], dim=0)

            new_sample = {
                'images': images,
                'pseudo_masks': pseudo_masks,
                'j2d': sample['j2d'],
                'T': sample['T'],
                'bg_inputs': bg_inputs,
                'src_inputs': src_inputs,
                'tsf_inputs': tsf_inputs,
                'src_fim': sample['src_fim'],
                'tsf_fim': sample['tsf_fim']
            }

            if 'T_cycle' in sample:
                new_sample['T_cycle'] = sample['T_cycle']

            if 'T_cycle_vis' in sample:
                new_sample['T_cycle_vis'] = sample['T_cycle_vis']

            return new_sample


class MetaCycleDataSet(PairSampleDataset):
    def __init__(self, opt, is_for_train):
        super(MetaCycleDataSet, self).__init__(opt, is_for_train)
        self._name = 'MetaCycleDataSet'

    def _read_samples_info(self, im_dir, pkl_dir, pair_ids_filepath):
        """
        Args:
            im_dir:
            pkl_dir:
            pair_ids_filepath:

        Returns:

        """
        # 1. load image pair list
        self.im_pair_list = load_pickle_file(pair_ids_filepath)

        # 2. load pkl file paths
        self.all_pkl_paths = sorted(glob.glob((os.path.join(pkl_dir, '*.pkl'))))

        assert len(self.im_pair_list) == len(self.all_pkl_paths), '{} != {}'.format(
            len(self.im_pair_list), len(self.all_pkl_paths)
        )
        self._dataset_size = len(self.im_pair_list)

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
        im_pairs = self.im_pair_list[item]
        pkl_path = self.all_pkl_paths[item]

        sample = self.load_sample(im_pairs, pkl_path)
        sample = self.preprocess(sample)

        sample['preds'] = torch.tensor(self.load_init_preds(im_pairs[1])).float()

        return sample

    def load_init_preds(self, pred_path):
        pred_img_name = os.path.split(pred_path)[-1]
        pred_img_path = os.path.join(self._opt.preds_img_folder, 'pred_' + pred_img_name)

        img = cv_utils.read_cv2_img(pred_img_path)
        img = cv_utils.transform_img(img, self._opt.image_size, transpose=True)
        img = img * 2 - 1

        return img


def make_dataset(opt):
    class Config(object):
        pass

    config = Config()

    output_dir = opt.output_dir

    config.data_dir = output_dir
    config.images_folder = 'motion_transfer_HD'
    config.smpls_folder = 'motion_transfer_smpl'
    config.train_pkl_folder = 'pairs'
    config.train_ids_file = os.path.join(output_dir, 'pairs_meta.pkl')
    config.preds_img_folder = os.path.join(output_dir, 'imgs')
    config.image_size = opt.image_size
    config.map_name = opt.map_name
    config.uv_mapping = opt.uv_mapping
    config.is_both = False
    config.bg_ks = opt.bg_ks
    config.ft_ks = opt.ft_ks

    meta_cycle_ds = MetaCycleDataSet(opt=config, is_for_train=True)
    length = len(meta_cycle_ds)

    data_loader = torch.utils.data.DataLoader(
        meta_cycle_ds,
        batch_size=min(length, opt.batch_size),
        shuffle=False,
        num_workers=4,
        drop_last=True)

    return data_loader


def adaptive_personalize(opt, imitator, visualizer):
    output_dir = opt.output_dir
    out_img_dir, out_pair_dir = mkdirs([os.path.join(output_dir, 'imgs'), os.path.join(output_dir, 'pairs')])

    # TODO check if it has been computed.
    print('\n\t\t\tPersonalization: meta imitation...')
    imitator.personalize(opt.src_path, visualizer=None)
    meta_imitate(opt, imitator, prior_tgt_path=opt.pri_path, visualizer=None, save_imgs=True)

    # post tune
    print('\n\t\t\tPersonalization: meta cycle finetune...')
    loader = make_dataset(opt)
    imitator.post_personalize(opt.output_dir, loader, visualizer=None, verbose=True)


def load_mixamo_smpl(mixamo_idx):
    mixamo_root_path = '/root/A_dataset/mixamo'
    dir_name = '%.4d' % mixamo_idx
    pkl_path = os.path.join(mixamo_root_path, dir_name, 'result.pkl')

    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    anim_len = result['anim_len']
    pose_array = result['smpl_array'].reshape(anim_len, -1)
    cam_array = result['cam_array']
    shape_array = np.ones((anim_len, 10))
    smpl_array = np.concatenate((cam_array, pose_array, shape_array), axis=1)

    return smpl_array


MIXAMO_DANCE_ACTION_IDX_LIST = [78, 79, 102, 155, 159]
MIXAMO_BASE_ACTION_IDX_LIST = [0, 8, 10, 20, 22, 32, 70, 96, 104, 148, 196, 228, 229]
MIXAMO_ACROBAT_ACTION_IDX_LIST = [7, 24, 29, 31, 76, 83, 87, 120, 129, 130, 131, 132, 133, 134, 141, 142, 145, 161, 166, 177]


def generate_actor_result(test_opt, src_img_path):
    imitator = ModelsFactory.get_by_name(test_opt.model, test_opt)
    src_img_name = os.path.split(src_img_path)[-1][:-4]
    test_opt.src_path = src_img_path

    if test_opt.post_tune:
        adaptive_personalize(test_opt, imitator, visualizer=None)
    else:
        imitator.personalize(test_opt.src_path, visualizer=None)

    action_list_dict = {'dance': MIXAMO_DANCE_ACTION_IDX_LIST,
                        'base': MIXAMO_BASE_ACTION_IDX_LIST,
                        'acrobat': MIXAMO_ACROBAT_ACTION_IDX_LIST}

    for action_type in ['dance', 'base', 'acrobat']:
        for i in action_list_dict[action_type]:
            if test_opt.output_dir:
                pred_output_dir = os.path.join(test_opt.output_dir, 'mixamo_preds')
                os.system("rm -r %s" % pred_output_dir)
                mkdir(pred_output_dir)
            else:
                pred_output_dir = None

            print(pred_output_dir)
            tgt_smpls = load_mixamo_smpl(i)

            # dance_demo_id = 7
            # video_info_pkl_path = '/root/impersonator_piao/input_video_data/dance_demo_%d_smooth_hmr_low_pass_smpl.pkl' % dance_demo_id
            # with open(video_info_pkl_path, 'rb') as f:
            #     video_info_list = pickle.load(f)
            # tgt_smpls = [video_info['smpl_param'] for video_info in video_info_list]

            imitator.inference_by_smpls(tgt_smpls, cam_strategy='smooth', output_dir=pred_output_dir, visualizer=None)

            save_dir = os.path.join(test_opt.output_dir, src_img_name, action_type)
            mkdir(save_dir)

            output_mp4_path = os.path.join(save_dir, 'mixamo_%.4d_%s.mp4' % (i, src_img_name))
            img_path_list = sorted(glob.glob('%s/*.jpg' % pred_output_dir))
            make_video(output_mp4_path, img_path_list, save_frames_dir=None, fps=30)


def create_T_pose_novel_view_smpl():
    # cam + pose + shape
    smpls = np.zeros((180, 75))

    for i in range(180):
        r1 = R.from_rotvec([0, 0, 0])
        r2 = R.from_euler("xyz", [180, i * 2, 0], degrees=True)
        r = (r1 * r2).as_rotvec()

        smpls[i, 3:6] = r

    return smpls


def generate_T_pose_novel_view_result(test_opt, src_img_path):
    imitator = ModelsFactory.get_by_name(test_opt.model, test_opt)
    src_img_name = os.path.split(src_img_path)[-1][:-4]
    test_opt.src_path = src_img_path

    if test_opt.post_tune:
        adaptive_personalize(test_opt, imitator, visualizer=None)
    else:
        imitator.personalize(test_opt.src_path, visualizer=None)

    if test_opt.output_dir:
        pred_output_dir = os.path.join(test_opt.output_dir, 'T_novel_view_preds')
        os.system("rm -r %s" % pred_output_dir)
        mkdir(pred_output_dir)
    else:
        pred_output_dir = None

    print(pred_output_dir)
    tgt_smpls = create_T_pose_novel_view_smpl()

    imitator.inference_by_smpls(tgt_smpls, cam_strategy='smooth', output_dir=pred_output_dir, visualizer=None)

    save_dir = os.path.join(test_opt.output_dir, src_img_name)
    mkdir(save_dir)

    output_mp4_path = os.path.join(save_dir, 'T_novel_view_%s.mp4' % (src_img_name))
    img_path_list = sorted(glob.glob('%s/*.jpg' % pred_output_dir))
    make_video(output_mp4_path, img_path_list, save_frames_dir=None, fps=30)


def main():
    # meta imitator
    test_opt = TestOptions().parse()
    test_opt.front_warp = False
    test_opt.post_tune = False
    test_opt.post_tune = True
    test_opt.name = 'impersonator_mi_fashion_place'
    test_opt.checkpoints_dir = '/public/liuwen/p300/models'

    """
    mixamo
    """
    demo_img_dir = 'good_actor/imper'
    test_opt.output_dir = 'meta_train/result_%s' % os.path.split(demo_img_dir)[-1]

    for src_img_path in tqdm(sorted(glob.glob("%s/*" % demo_img_dir))):
        generate_actor_result(test_opt, src_img_path)

    """
    T pose novel view
    """
    # demo_img_dir = 'good_actor/man'
    # test_opt.output_dir = 'meta_train/T_novel_view_result_%s' % os.path.split(demo_img_dir)[-1]
    #
    # for src_img_path in tqdm(sorted(glob.glob("%s/*" % demo_img_dir))):
    #     generate_T_pose_novel_view_result(test_opt, src_img_path)


if __name__ == "__main__":
    main()
