from tqdm import tqdm
from models.models import ModelsFactory
from options.test_options import TestOptions
from utils.demo_visualizer import MotionImitationVisualizer

import ipdb
import cv2
import glob
import os
import numpy as np
from utils.video import make_video


def tensor2cv2(img_tensor):
    img = (img_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
    img = img[:, :, ::-1]
    img = (img * 255).astype(np.uint8)

    return img


if __name__ == "__main__":

    opt = TestOptions().parse()
    # opt.src_path = 'meta_train/samples/all_img/men1_256.jpg'
    # opt.tgt_path = 'meta_train/samples/all_img/8_256.jpg'

    # opt.src_path = 'good_actor/woman/fashionWOMENBlouses_Shirtsid0000694703_4full.jpg'
    # opt.src_path = 'good_actor/woman/Sweaters-id_0000363204_4_full.jpg'
    # opt.src_path = 'good_actor/man/Jackets_Vests-id_0000190301_4_full.jpg'
    # opt.src_path = 'good_actor/woman/Sweaters-id_0000337302_4_full.jpg'
    opt.src_path = 'good_actor/imper/000.jpg'

    # opt.tgt_path = 'good_actor/woman/fashionWOMENBlouses_Shirtsid0000695303_4full.jpg'
    # opt.tgt_path = 'good_actor/woman/fashionWOMENDressesid0000271801_4full.jpg'
    # opt.tgt_path = 'good_actor/man/Jackets_Vests-id_0000190301_4_full.jpg'
    # opt.tgt_path = 'good_actor/man/Jackets_Vests-id_0000009401_4_full.jpg'
    opt.tgt_path = 'good_actor/woman/fashionWOMENDressesid0000271801_4full.jpg'
    # opt.tgt_path = 'good_actor/imper/000.jpg'
    # opt.tgt_path = 'good_actor/woman/Sweaters-id_0000337302_4_full.jpg'

    opt.front_warp = True
    opt.visual = True
    opt.post_tune = True

    tgt_path_list = sorted(glob.glob("good_actor/*/*"))
    src_path_list = sorted(glob.glob("good_actor/imper/*"))

    for src_path in src_path_list:
        opt.src_path = src_path

        for tgt_path in tgt_path_list:
            opt.tgt_path = tgt_path

            # set imitator
            swapper = ModelsFactory.get_by_name(opt.model, opt)

            if opt.visual:
                visualizer = MotionImitationVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
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
            result = swapper.swap(src_info=swapper.src_info, tgt_info=swapper.tsf_info, target_part=opt.swap_part, visualizer=visualizer)
            # else b->a
            # swapper.swap(src_info=swapper.tgt_info, tgt_info=swapper.src_info, target_part=opt.swap_part, visualizer=visualizer)

            src_img_true_name = os.path.split(opt.src_path)[-1][:-4]

            save_dir = 'meta_train/swap_result/%s' % src_img_true_name
            os.makedirs(save_dir, exist_ok=True)
            save_img_name = '%s.%s' % (os.path.split(opt.src_path)[-1], os.path.split(opt.tgt_path)[-1])

            cv2.imwrite('%s/%s' % (save_dir, save_img_name), tensor2cv2(result))
