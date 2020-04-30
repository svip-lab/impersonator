import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as transform_func
import yaml
import cv2
import numpy as np

from typing import List, Union

from .model import ft_net, ft_net_dense, PCB, PCB_test


class Config(object):
    pass


def create_model(name='PCB', pretrain_path='./model'):
    cfg = Config()

    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, 'model', name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

        # PCB: false
        # batchsize: 32
        # color_jitter: false
        # data_dir:../ Market / pytorch
        # droprate: 0.5
        # erasing_p: 0
        # fp16: true
        # gpu_ids: '0'
        # lr: 0.05
        # name: ft_ResNet50
        # stride: 2
        # train_all: false
        # use_dense: false
        # nclasses: 751

        cfg.PCB = config['PCB']
        cfg.name = config['name']
        cfg.stride = config['stride']
        cfg.use_dense = config['use_dense']
        cfg.nclasses = config['nclasses']

    if cfg.use_dense:
        model_structure = ft_net_dense(cfg.nclasses)
    else:
        model_structure = ft_net(cfg.nclasses, stride=cfg.stride)

    if cfg.PCB:
        model_structure = PCB(cfg.nclasses)

    model_structure.load_state_dict(torch.load(pretrain_path))

    if cfg.PCB:
        model_structure = PCB_test(model_structure)
        cfg.height, cfg.width = 384, 192
    else:
        cfg.height, cfg.width = 256, 128

    return model_structure, cfg


class PCBReIDMetric(nn.Module):

    def __init__(self, name, pretrain_path):
        super(PCBReIDMetric, self).__init__()
        self.model, self.opt = create_model(name, pretrain_path)
        self.model.eval()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, x: List[np.ndarray], bboxs: Union[List[np.ndarray], None] = None, ignore_head: bool = False):
        """

        Args:
            x (list of np.ndarray): [(3, height, width), (3, height, width), ..., (3, height, width)], each item of
                                (3, height, width) is in the range of [0, 1.0], with np.float32.
            bboxs (list of np.ndarray or None): [(4,), (4,), ..., (4,)] with np.int32
            ignore_head (bool):

        Returns:
            feats (torch.tensor): (bs, dim)
        """
        x = self.crop_patch(x, bboxs)
        feats = self.extract_feat(x, bboxs, ignore_head)
        return feats

    def crop_patch(self, images: torch.tensor, bboxs: Union[torch.tensor, None] = None):
        """

        Args:
            images (torch.tensor): [bs, 3, height, width] is in range of [0, 255] with torch.float32
            bboxs (torch.tensor or None): [(4,), (4,), ..., (4,)] with np.int32

        Returns:
            crop (torch.tensor): (bs, 3, resize_height, resize_width)
        """

        if bboxs is None:
            crop_imgs = F.interpolate(
                images, size=(self.opt.height, self.opt.width),
                mode="bilinear", align_corners=True
            )
        else:
            bs = len(images)
            crop_imgs = []
            for i in range(bs):
                x = images[i]
                box = bboxs[i]
                if box is not None:
                    x0, y0, x1, y1 = box
                    crop = x[:, y0:y1, x0:x1]
                else:
                    crop = x
                crop = transform_func.normalize(crop, mean=self.mean, std=self.std)
                crop.unsqueeze_(0)
                crop = F.interpolate(crop, size=(self.opt.height, self.opt.width), mode="bilinear", align_corners=True)
                crop_imgs.append(crop)
            crop_imgs = torch.cat(crop_imgs, dim=0)
        return crop_imgs

    def extract_feat(self, x: torch.tensor,
                     bboxs: Union[List[np.ndarray], None] = None,
                     ignore_head: bool = False):

        with torch.no_grad():
            ff = self.model(x)
            sqrt_num = 6
            # ipdb.set_trace()
            if ignore_head:
                ff = ff[:, :, 1:]
                sqrt_num = 5

            if self.opt.PCB:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(sqrt_num)
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

            return ff

    def load_img(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.opt.width, self.opt.height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0

        image = self.crop_patch(image, bboxs=None)
        return image


if __name__ == '__main__':
    import ipdb

    pReIdMetric = PCBReIDMetric('PCB')
    print(pReIdMetric)

    norm = True

    src_path = './examples/gt.jpg'
    src_img = pReIdMetric.load_img(src_path)
    src_out = pReIdMetric.extract_feat(src_img, norm=norm)

    print(src_img.shape, src_img.max(), src_img.min())
    for img_name in ['source.jpg', 'gt.jpg', 'ours.jpg', 'pG2.jpg', 'SHUP.jpg', 'DSC.jpg']:
        img_path = os.path.join('./examples', img_name)

        test_img = pReIdMetric.load_img(img_path)
        test_out = pReIdMetric.extract_feat(test_img, norm=norm)
        print(img_name, torch.sum((src_out * test_out)))

    ipdb.set_trace()
