import os
from models.swapper import Swapper
from options.test_options import TestOptions
from utils.visdom_visualizer import VisdomVisualizer
from utils.util import mkdir


def get_img_name(img_path: str):
    """
        Get the name from the image path.

    Args:
        img_path (str): a/b.jpg or a/b.png ...

    Returns:
        name (str): a/b.jpg -> b
    """
    image_name = os.path.split(img_path)[-1].split('.')[0]
    return image_name


def save_results(src_path, tgt_path, output_dir, preds):
    """
        Save the results.
    """
    import utils.cv_utils as cv_utils

    src_name = get_img_name(src_path)
    tgt_name = get_img_name(tgt_path)

    preds = preds[0].permute(1, 2, 0)
    preds = preds.cpu().numpy()

    filepath = os.path.join(output_dir, '{}->{}.png'.format(src_name, tgt_name))
    cv_utils.save_cv2_img(preds, filepath, normalize=True)
    print('\n\t\t\tSaving results to {}'.format(filepath))


if __name__ == "__main__":
    opt = TestOptions().parse()

    # set imitator
    swapper = Swapper(opt=opt)

    if opt.ip:
        visualizer = VisdomVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    src_path = opt.src_path
    tgt_path = opt.tgt_path

    swapper.swap_setup(src_path, tgt_path)

    if opt.post_tune:
        print('\n\t\t\tPersonalization: meta cycle finetune...')
        swapper.post_personalize(opt.output_dir, visualizer=None, verbose=False)

    print('\n\t\t\tPersonalization: completed...')

    # if a->b
    print('\n\t\t\tSwapping: {} wear the clothe of {}...'.format(src_path, tgt_path))
    preds = swapper.swap(src_info=swapper.src_info, tgt_info=swapper.tsf_info,
                         target_part=opt.swap_part, visualizer=visualizer)

    if opt.save_res:
        pred_output_dir = mkdir(os.path.join(opt.output_dir, 'swappers'))
        save_results(src_path, tgt_path, pred_output_dir, preds)

    # # else b->a
    # preds = swapper.swap(src_info=swapper.tgt_info, tgt_info=swapper.src_info,
    #                      target_part=opt.swap_part, visualizer=visualizer)

