import numpy as np
from tqdm import tqdm
import cv2
import os
import glob

from models.imitator import Imitator
from models.viewer import Viewer
from options.test_options import TestOptions

from utils.visdom_visualizer import VisdomVisualizer
from utils.video import make_video
from utils.util import mkdir

from run_imitator import adaptive_personalize


def clean(output_dir):

    for item in ['imgs', 'pairs', 'mixamo_preds', 'pairs_meta.pkl', 'T_novel_view_preds']:
        filepath = os.path.join(output_dir, item)
        if os.path.exists(filepath):
            os.system("rm -r %s" % filepath)


def tensor2cv2(img_tensor):
    img = (img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2
    img = img[:, :, ::-1]
    img = (img * 255).astype(np.uint8)

    return img


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


def create_T_pose_novel_view_smpl():
    from scipy.spatial.transform import Rotation as R
    # cam + pose + shape
    smpls = np.zeros((180, 75))

    for i in range(180):
        r1 = R.from_rotvec([0, 0, 0])
        r2 = R.from_euler("xyz", [180, i * 2, 0], degrees=True)
        r = (r1 * r2).as_rotvec()

        smpls[i, 3:6] = r

    return smpls


def generate_T_pose_novel_view_result(test_opt, src_img_path):
    imitator = Imitator(test_opt)
    src_img_name = os.path.split(src_img_path)[-1][:-4]
    test_opt.src_path = src_img_path

    if test_opt.post_tune:
        adaptive_personalize(test_opt, imitator, visualizer=None)
    else:
        imitator.personalize(test_opt.src_path, visualizer=None)

    if test_opt.output_dir:
        pred_output_dir = os.path.join(test_opt.output_dir, 'T_novel_view_preds')
        if os.path.exists(pred_output_dir):
            os.system("rm -r %s" % pred_output_dir)
        mkdir(pred_output_dir)
    else:
        pred_output_dir = None

    print(pred_output_dir)
    tgt_smpls = create_T_pose_novel_view_smpl()

    imitator.inference_by_smpls(tgt_smpls, cam_strategy='smooth', output_dir=pred_output_dir, visualizer=None)

    save_dir = os.path.join(test_opt.output_dir, src_img_name)
    mkdir(save_dir)

    output_mp4_path = os.path.join(save_dir, 'T_novel_view_%s.mp4' % src_img_name)
    img_path_list = sorted(glob.glob('%s/*.jpg' % pred_output_dir))
    make_video(output_mp4_path, img_path_list, save_frames_dir=None, fps=30)

    # clean other left
    clean(test_opt.output_dir)


def generate_orig_pose_novel_view_result(opt, src_path):
    opt.src_path = src_path
    # set imitator
    viewer = Viewer(opt)

    if opt.ip:
        visualizer = VisdomVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    if opt.post_tune:
        adaptive_personalize(opt, viewer, visualizer)

    viewer.personalize(opt.src_path, visualizer=visualizer)
    print('\n\t\t\tPersonalization: completed...')

    view_params = opt.view_params
    params = parse_view_params(view_params)

    length = 180
    delta = 360 / length
    logger = tqdm(range(length))

    src_img_true_name = os.path.split(opt.src_path)[-1][:-4]
    save_dir = os.path.join(opt.output_dir, src_img_true_name)
    mkdir(os.path.join(save_dir, 'imgs'))

    print('\n\t\t\tSynthesizing {} novel views'.format(length))
    for i in logger:
        params['R'][0] = 0
        params['R'][1] = delta * i / 180.0 * np.pi
        params['R'][2] = 0

        preds = viewer.view(params['R'], params['t'], visualizer=None, name=str(i))
        # pred_outs.append(preds)

        save_img_name = '%s.%d.jpg' % (os.path.split(opt.src_path)[-1], delta * i)

        cv2.imwrite('%s/imgs/%s' % (save_dir, save_img_name), tensor2cv2(preds))

    """
    make video
    """
    img_path_list = glob.glob("%s/imgs/*.jpg" % save_dir)
    output_mp4_path = '%s/%s.mp4' % (save_dir, src_img_true_name)
    make_video(output_mp4_path, img_path_list, save_frames_dir=None, fps=30)

    clean(opt.output_dir)
    clean(save_dir)


if __name__ == "__main__":

    opt = TestOptions().parse()
    opt.bg_ks = 31
    opt.T_pose = False
    opt.front_warp = False
    opt.bg_replace = True
    opt.post_tune = True
    opt.output_dir = './outputs/results/demos/viewers'

    src_path_list = [
        ('iPER', './assets/src_imgs/imper_Random_Pose/novel_3.jpg'),
        ('Fashion', './assets/src_imgs/fashion_woman/fashionWOMENDressesid0000271801_4full.jpg'),
        ('Fashion', './assets/src_imgs/fashion_man/Jackets_Vests-id_0000071603_4_full.jpg')
    ]

    for dataset, src_path in src_path_list:
        if dataset == 'Fashion':
            opt.T_pose = True
            generate_T_pose_novel_view_result(opt, src_path)
        else:
            opt.T_pose = False
            generate_orig_pose_novel_view_result(opt, src_path)
