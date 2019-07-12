import os
import cv2
from utils import cv_utils
import torchvision.transforms as transforms
import torch
import numpy as np
from models import ModelsFactory
from data.dataset import DatasetFactory
from utils.visualizer.demo_visualizer import MotionImitationVisualizer
from options.test_options import TestOptions


class MotionImitator(object):
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()

        # self._hmr = self._model.bdr.hmr
        self._hmr = self._model.hmr

        self._transform = transforms.Compose([
            cv_utils.ImageTransformer(output_size=self._opt.image_size)]
        )

        if self._opt.visual:
            self._visualizer = MotionImitationVisualizer(env=self._opt.name, ip=self._opt.ip, port=self._opt.port)

    def _extract_smpls(self, input_file):
        img = cv_utils.read_cv2_img(input_file)
        img = cv_utils.transform_img(img, image_size=224) * 2 - 1.0  # hmr receive [-1, 1]
        img = img.transpose((2, 0, 1))
        img = torch.FloatTensor(img).cuda()[None, ...]
        theta = self._hmr(img)

        return theta

    def _set_smpls(self, tgt_file, get_detail=False):
        if isinstance(tgt_file, str):
            tgt_smpl = self._extract_smpls(tgt_file)

        elif isinstance(tgt_file, torch.FloatTensor):
            tgt_smpl = tgt_file[None, ...].cuda()

        elif isinstance(tgt_file, np.ndarray):
            tgt_smpl = torch.FloatTensor(tgt_file[None, ...]).cuda()

        else:
            raise ValueError('type error of {}'.format(type(tgt_file)))

        smpl_out = tgt_smpl
        if get_detail:
            smpl_out = self._hmr.get_details(tgt_smpl)

        return smpl_out

    def imitate(self, src_img_file, tgt_seqs_file):
        with torch.no_grad():
            # 1. load source image
            org_img = cv_utils.read_cv2_img(src_img_file)
            src_imgs = cv_utils.transform_img(org_img, image_size=self._opt.image_size) * 2 - 1.0  # hmr receive [-1, 1]
            src_imgs = torch.FloatTensor(src_imgs.transpose((2, 0, 1)))[None, ...].cuda()
            src_smpl = self._set_smpls(src_img_file)

            inputs = {
                'src_img': src_imgs,
                'src_smpl': src_smpl,
                'desired_smpl': None
            }

            if self._opt.visual:
                self._visualizer.vis_named_img('source image', src_imgs)

            # 2. load target sequences
            for t, tgt_file in enumerate(tgt_seqs_file):
                # torch.cuda.FloatTensor, (1, 85)
                tgt_smpl = self._set_smpls(tgt_file)
                inputs['desired_smpl'] = tgt_smpl

                # set input
                self._model.set_test_input(inputs)

                # generate cur image
                # imgs = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
                # return imgs['concat']
                cur_imgs = self._model.forward(keep_data_for_visuals=False, return_estimates=False)[0]

                if self._opt.visual:
                    # self._visualizer.vis_preds_gts(cur_imgs, gts=None)
                    self._visualizer.vis_named_img('transfer', cur_imgs)

                print(t, cur_imgs.shape)

                # if self._opt.visual and t == 50:
                #     self._model.debug(visualizer=self._visualizer)
                #     break

                # time.sleep(1)

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)

    def load_model(self, pretrain_model):
        saved_data = torch.load(pretrain_model)
        self._hmr.load_state_dict(saved_data)
        print('load hmr model from {}'.format(pretrain_model))


def main():
    opt = TestOptions().parse()

    imitator = MotionImitator(opt)
    dataset = DatasetFactory.get_by_name(opt.dataset_mode, opt, is_for_train=False)
    videos_info = dataset.video_info

    for v_id, info in enumerate(videos_info):
        # info = {
        #     'images': images_path,
        #     'cams': cams,
        #     'thetas': smpl_data['pose'],
        #     'betas': smpl_data['shape'],
        #     'length': len(images_path)
        # }
        images = info['images']
        image_path = images[0]

        imitator.imitate(image_path, images)


if __name__ == '__main__':
    main()
