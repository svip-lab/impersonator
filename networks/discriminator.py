import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from .networks import NetworkBase


class PatchDiscriminator(NetworkBase):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='batch', use_sigmoid=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchDiscriminator, self).__init__()

        norm_layer = self._get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class GlobalLocalDiscriminator(NetworkBase):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='batch', use_sigmoid=False):
        super(GlobalLocalDiscriminator, self).__init__()
        self.global_model = PatchDiscriminator(4, ndf=ndf, n_layers=n_layers,
                                               norm_type=norm_type, use_sigmoid=use_sigmoid)
        self.local_model = PatchDiscriminator(input_nc, ndf=ndf, n_layers=n_layers,
                                              norm_type=norm_type, use_sigmoid=use_sigmoid)
        # from utils.demo_visualizer import MotionImitationVisualizer
        # self._visualizer = MotionImitationVisualizer('debug', ip='http://10.10.10.100', port=31102)

    def forward(self, global_x, local_x, local_rects):
        glocal_outs = self.global_model(global_x)
        crop_imgs = self.crop_body(local_x, local_rects)
        local_outs = self.local_model(crop_imgs)

        # self._visualizer.vis_named_img('body_imgs', crop_imgs[:, 0:3])

        return torch.cat([glocal_outs, local_outs], dim=0)

    @staticmethod
    def crop_body(imgs, rects):
        """
        :param imgs: (N, C, H, W)
        :return:
        """
        bs, _, ori_h, ori_w = imgs.shape
        head_imgs = []

        for i in range(bs):
            min_x, max_x, min_y, max_y = rects[i].detach()
            head = imgs[i:i+1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
            head = F.interpolate(head, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
            head_imgs.append(head)

        head_imgs = torch.cat(head_imgs, dim=0)

        return head_imgs


class MultiScaleDiscriminator(NetworkBase):

    def __init__(self, input_nc, n_scales=5, ndf=64, n_layers=3, use_sigmoid=False):
        super(MultiScaleDiscriminator, self).__init__()

        # low-res to high-res
        scale_models = nn.ModuleList()
        # scale_n_layers = [1, 1, 1, 1, 1]
        for i in range(n_scales):
            # n_layers = scale_n_layers[i]
            model = PatchDiscriminator(input_nc, ndf, n_layers, use_sigmoid)
            scale_models.append(model)

        self.n_scales = n_scales
        self.scale_models = scale_models

    def forward(self, x, is_detach=False):
        scale_outs = []
        for i in range(0, self.n_scales):
            x_scale = x[i]
            if is_detach:
                x_scale = x_scale.detach()

            outs = self.scale_models[i](x_scale)
            # print(i, x[i].shape, outs.shape)

            scale_outs.append(outs)

        return scale_outs



