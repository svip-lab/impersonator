from tqdm import tqdm
from models.models import ModelsFactory
from options.test_options import TestOptions
from utils.visdom_visualizer import VisdomVisualizer


import ipdb


if __name__ == "__main__":

    opt = TestOptions().parse()

    # set imitator
    swapper = ModelsFactory.get_by_name(opt.model, opt)

    if opt.visual:
        visualizer = VisdomVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    src_path = opt.src_path
    tgt_path = opt.tgt_path

    swapper.swap_setup(src_path, tgt_path)

    if opt.post_tune:
        print('\n\t\t\tPersonalization: meta cycle finetune...')
        swapper.post_personalize(opt.output_dir, visualizer=None, verbose=True)

    print('\n\t\t\tPersonalization: completed...')

    # if a->b
    print('\n\t\t\tSwapping: {} wear the clothe of {}...'.format(src_path, tgt_path))
    swapper.swap(src_info=swapper.src_info, tgt_info=swapper.tsf_info, target_part=opt.swap_part, visualizer=visualizer)
    # else b->a
    # swapper.swap(src_info=swapper.tgt_info, tgt_info=swapper.src_info, target_part=opt.swap_part, visualizer=visualizer)

