import torch.utils.data
import torchvision.utils
import numpy as np
from tqdm import tqdm
import os

from models.viewer import Viewer
from options.test_options import TestOptions
from utils.visdom_visualizer import VisdomVisualizer
from utils.util import mkdir

from run_imitator import adaptive_personalize


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




