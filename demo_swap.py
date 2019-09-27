import cv2
import os
import numpy as np

from models.swapper import Swapper
from options.test_options import TestOptions
from utils.visdom_visualizer import VisdomVisualizer
from utils.util import mkdir


def tensor2cv2(img_tensor):
    img = (img_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
    img = img[:, :, ::-1]
    img = (img * 255).astype(np.uint8)

    return img


if __name__ == "__main__":

    opt = TestOptions().parse()
    opt.bg_ks = 25
    opt.front_warp = True
    opt.post_tune = True

    src_path_list = [('iPER',    './assets/src_imgs/imper_A_Pose/009_5_1_000.jpg'),
                     ('Fashion', './assets/src_imgs/fashion_man/Jackets_Vests-id_0000008408_4_full.jpg'),
                     ('Fashion', './assets/src_imgs/fashion_woman/Sweaters-id_0000088807_4_full.jpg')]

    tgt_path_list = ['./assets/src_imgs/fashion_woman/fashionWOMENBlouses_Shirtsid0000666802_4full.jpg',
                     './assets/src_imgs/fashion_man/Sweatshirts_Hoodies-id_0000680701_4_full.jpg',
                     './assets/src_imgs/fashion_man/Sweatshirts_Hoodies-id_0000097801_4_full.jpg']

    for (dataset, src_path) in src_path_list:
        opt.src_path = src_path

        for tgt_path in tgt_path_list:
            opt.tgt_path = tgt_path

            # set imitator
            swapper = Swapper(opt)

            if opt.ip:
                visualizer = VisdomVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
            else:
                visualizer = None

            src_path = opt.src_path
            tgt_path = opt.tgt_path

            swapper.swap_setup(src_path, tgt_path)

            if opt.post_tune:
                print('\n\t\t\tPersonalization: meta cycle finetune...')
                swapper.post_personalize(opt.output_dir, visualizer=visualizer, verbose=True)

            print('\n\t\t\tPersonalization: completed...')

            # if a->b
            print('\n\t\t\tSwapping: {} wear the clothe of {}...'.format(src_path, tgt_path))
            result = swapper.swap(src_info=swapper.src_info, tgt_info=swapper.tsf_info, target_part=opt.swap_part,
                                  visualizer=visualizer)
            # else b->a
            # swapper.swap(src_info=swapper.tsf_info, tgt_info=swapper.src_info, target_part=opt.swap_part,
            #              visualizer=visualizer)

            src_img_true_name = os.path.split(opt.src_path)[-1][:-4]

            save_dir = mkdir('./outputs/results/demos/swappers/%s' % src_img_true_name)
            save_img_name = '%s.%s' % (os.path.split(opt.src_path)[-1], os.path.split(opt.tgt_path)[-1])

            cv2.imwrite('%s/%s' % (save_dir, save_img_name), tensor2cv2(result))
