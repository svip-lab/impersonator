import torch
import torch.nn
import torch.utils.data
import torchvision.utils
import numpy as np
from tqdm import tqdm
import os
import glob

from models.viewer import Viewer
from data.dataset import PairSampleDataset
from options.test_options import TestOptions
from utils.visdom_visualizer import VisdomVisualizer
from utils.util import load_pickle_file, write_pickle_file, mkdir, mkdirs
import utils.cv_utils as cv_utils


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


class MetaCycleDataSet(PairSampleDataset):
    def __init__(self, opt):
        super(MetaCycleDataSet, self).__init__(opt, True)
        self._name = 'MetaCycleDataSet'

    def _read_dataset_paths(self):
        # read pair list
        self._dataset_size = 0
        self._read_samples_info(None, self._opt.pkl_dir, self._opt.pair_ids_filepath)

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

    config.pair_ids_filepath = os.path.join(output_dir, 'pairs_meta.pkl')
    config.pkl_dir = os.path.join(output_dir, 'pairs')
    config.preds_img_folder = os.path.join(output_dir, 'imgs')
    config.image_size = opt.image_size
    config.map_name = opt.map_name
    config.uv_mapping = opt.uv_mapping
    config.is_both = False
    config.bg_ks = opt.bg_ks
    config.ft_ks = opt.ft_ks

    meta_cycle_ds = MetaCycleDataSet(opt=config)
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
    mkdirs([os.path.join(output_dir, 'imgs'), os.path.join(output_dir, 'pairs')])

    # TODO check if it has been computed.
    print('\n\t\t\tPersonalization: meta imitation...')
    imitator.personalize(opt.src_path, visualizer=None)
    meta_imitate(opt, imitator, prior_tgt_path=opt.pri_path, visualizer=None, save_imgs=True)

    # post tune
    print('\n\t\t\tPersonalization: meta cycle finetune...')
    loader = make_dataset(opt)
    imitator.post_personalize(opt.output_dir, loader, visualizer=None, verbose=False)


def parse_view_params(view_params):
    """
    :param view_params: R=xxx,xxx,xxx/t=xxx,xxx,xxx
    :return:
        -R: np.ndarray, (3,)
        -t: np.ndarray, (3,)
    """

    params = dict()
    for segment in view_params.split('/'):
        # R=xxx,xxx,xxx -> (name, xxx,xxx,xxx)
        name, params_str = segment.split('=')

        vals = [float(val) for val in params_str.split(',')]

        params[name] = np.array(vals, dtype=np.float32)

    params['R'] = params['R'] / 180 * np.pi
    return params


if __name__ == "__main__":

    opt = TestOptions().parse()

    # set imitator
    viewer = Viewer(opt=opt)

    if opt.ip:
        visualizer = VisdomVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    if opt.post_tune:
        adaptive_personalize(opt, viewer, visualizer)

    viewer.personalize(opt.src_path, visualizer=visualizer)
    print('\n\t\t\tPersonalization: completed...')

    src_path = opt.src_path
    view_params = opt.view_params
    params = parse_view_params(view_params)

    length = 16
    delta = 360 / length
    pred_outs = []
    logger = tqdm(range(length))

    print('\n\t\t\tSynthesizing {} novel views'.format(length))
    for i in logger:
        params['R'][0] = 10 / 180 * np.pi
        params['R'][1] = delta * i / 180.0 * np.pi
        params['R'][2] = 10 / 180 * np.pi

        preds = viewer.view(params['R'], params['t'], visualizer=None, name=str(i))
        pred_outs.append(preds)

        logger.set_description(
            'view = ({:.3f}, {:.3f}, {:.3f})'.format(params['R'][0], params['R'][1], params['R'][2])
        )

    pred_outs = torch.cat(pred_outs, dim=0)
    pred_outs = (pred_outs + 1) / 2.0

    if opt.ip:
        visualizer.vis_named_img('preds', pred_outs, denormalize=False)

    if opt.save_res:
        pred_output_dir = mkdir(os.path.join(opt.output_dir, 'viewers'))
        filepath = os.path.join(pred_output_dir, src_path.split('/')[-1])
        torchvision.utils.save_image(pred_outs, filepath)

    # def process(x):
    #     return float(x) / 180 * np.pi
    #
    # while True:
    #     inputs = input('input thetas: ')
    #     if inputs == 'q':
    #         break
    #     thetas = list(map(process, inputs.split(' ')))
    #
    #     preds = viewer.view(thetas, params['t'], visualizer=None, name='0')
    #     visualizer.vis_named_img('pred', preds)




