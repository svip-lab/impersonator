import torch
import numpy as np
import cv2
import os

import networks
from models.models import ModelsFactory
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from options.eval_options import EvalOptions
from utils.visualizer.demo_visualizer import MotionImitationVisualizer
from utils.util import load_pickle_file


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


if __name__ == "__main__":
    import time
    opt = EvalOptions().parse()
    # set imitator
    imitator = ModelsFactory.get_by_name(opt.model, opt)
    imitator.set_eval()
    # imitator.set_G_train()

    if opt.visual:
        visualizer = MotionImitationVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    pair_dataloader = CustomDatasetDataLoader(opt, is_for_train=True, drop_last=False).load_data()
    unpair_dataloader = CustomDatasetDataLoader(opt, is_for_train=False, drop_last=False).load_data()

    # pair
    out_dir = opt.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    count = 0
    for i, sample in enumerate(pair_dataloader):
        imitator.set_input(sample)

        with torch.no_grad():
            fake_tsf_imgs, fake_imgs, fake_masks = imitator.forward()
            fake_tsf_imgs = fake_tsf_imgs.cpu()
            bs = fake_tsf_imgs.shape[0]

        # print('pair', i, sample['images'].shape, fake_tsf_imgs.shape)

        save_results(out_dir, sample['images'][:, 0, ...], sample['images'][:, 1, ...], fake_tsf_imgs, count)
        count += bs

        if visualizer is not None:
            visualizer.vis_named_img('src_img', sample['images'][:, 0, ...])
            visualizer.vis_named_img('ref_img', sample['images'][:, 1, ...])
            visualizer.vis_named_img('tsf_img', fake_tsf_imgs)

            time.sleep(1)

    # unpair
    for i, sample in enumerate(unpair_dataloader):
        imitator.set_input(sample)
        with torch.no_grad():
            fake_tsf_imgs, fake_imgs, fake_masks = imitator.forward()
            fake_tsf_imgs = fake_tsf_imgs.cpu()
            bs = fake_tsf_imgs.shape[0]

        # print('unpair', i, sample['images'].shape, fake_tsf_imgs.shape)

        save_results(out_dir, sample['images'][:, 0, ...], sample['images'][:, 1, ...], fake_tsf_imgs, count)
        count += bs
        if visualizer is not None:
            visualizer.vis_named_img('src_img', sample['images'][:, 0, ...])
            visualizer.vis_named_img('ref_img', sample['images'][:, 1, ...])
            visualizer.vis_named_img('tsf_img', fake_tsf_imgs)
            time.sleep(1)






