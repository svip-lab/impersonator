import torch
import torch.nn as nn
import numpy as np
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



