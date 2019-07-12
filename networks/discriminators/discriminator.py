import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
import functools
from networks.networks import NetworkBase


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


class GlobalLocalPatchDiscriminator(NetworkBase):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='batch', use_sigmoid=False):
        """Construct a  global and local PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(GlobalLocalPatchDiscriminator, self).__init__()

        self.global_model = PatchDiscriminator(input_nc, ndf, n_layers, norm_type, use_sigmoid=use_sigmoid)
        self.local_model = PatchDiscriminator(input_nc, ndf, n_layers, norm_type, use_sigmoid=use_sigmoid)

    def forward(self, inputs, masks):
        """Standard forward."""
        global_scores = self.global_model(inputs)
        local_scores = self.local_model(inputs * masks)
        return global_scores, local_scores


class MultiScaleDiscriminator(NetworkBase):

    def __init__(self, input_nc, n_scales=5, ndf=64, n_layers=3, use_sigmoid=False):
        super(MultiScaleDiscriminator, self).__init__()

        # low-res to high-res
        scale_models = nn.ModuleList()
        # scale_n_layers = [1, 1, 1, 1, 1]
        for i in range(n_scales):
            # n_layers = scale_n_layers[i]
            model = PatchDiscriminator(input_nc, ndf, n_layers, use_sigmoid=use_sigmoid)
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


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SNDiscriminator(NetworkBase):

    def __init__(self, input_nc, ndf, n_layers=4):
        super(SNDiscriminator, self).__init__()

        self.n_layers = n_layers
        kw = 4
        layers = list()

        max_cnum = 256
        c_in = input_nc
        c_out = ndf
        for i in range(self.n_layers):
            if i != self.n_layers - 1:
                layers.append(nn.Sequential(
                    spectral_norm(nn.Conv2d(c_in, c_out, stride=2, kernel_size=kw, padding=0)),
                    nn.LeakyReLU(0.02, inplace=True)
                ))
            else:
                layers.append(nn.Sequential(
                    spectral_norm(nn.Conv2d(c_in, c_out, stride=2, kernel_size=kw, padding=0))
                ))

            c_in = c_out
            c_out = min(max_cnum, c_out * 2)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    snd = SNDiscriminator(input_nc=5, ndf=64, nlayers=4)
    x = torch.rand(2, 5, 256, 256)
    x_outs = snd(x)
    print(x_outs.shape)

