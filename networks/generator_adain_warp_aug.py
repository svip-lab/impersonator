import torch.nn as nn
import torch.nn.functional as F
from .networks import NetworkBase
import torch
import ipdb


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, eps=1e-5):
        super(AdaptiveInstanceNorm, self).__init__()

        self.eps = eps

    def forward(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])

        style_mean, style_std = self.calc_mean_std(style_feat)
        return self.normalize(content_feat, style_feat, style_std)

    def normalize(self, content_feat, style_mean, style_std):
        size = content_feat.size()
        content_mean, content_std = self.calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def calc_mean_std(self, feat):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + self.eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, norm_fun=None):
        super(ResidualBlock, self).__init__()

        if norm_fun is not None:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                norm_fun(dim_out, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                norm_fun(dim_out, affine=True))
        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return x + self.main(x)


class AdaINResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(AdaINResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False))
        self.adain = AdaptiveInstanceNorm()

    def forward(self, content, style):
        style_mean, style_std = self.adain.calc_mean_std(style)

        x = self.main[0](content)
        x = self.main[1](self.adain.normalize(x, style_mean, style_std))
        x = self.main[2](x)

        return content + x


class AdaINBlock(nn.Module):

    def __init__(self, conv):
        super(AdaINBlock, self).__init__()
        self.conv = conv
        self.adain = AdaptiveInstanceNorm()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, style):
        x = self.conv(x)
        x = self.adain(x, style)
        x = self.act(x)
        return x


# class BGGenerator(NetworkBase):
#     """Generator. Encoder-Decoder Architecture."""
#     def __init__(self, conv_dim=64, c_dim=5, repeat_num=9, k_size=4, n_down=2):
#         super(BGGenerator, self).__init__()
#         self._name = 'bg_generator'
#
#         layers = []
#         layers.append(nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
#         layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
#         layers.append(nn.ReLU(inplace=True))
#
#         # Down-Sampling
#         curr_dim = conv_dim
#         for i in range(n_down):
#             layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=k_size, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim * 2
#
#         # Bottleneck
#         for i in range(repeat_num):
#             layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, norm_fun=nn.InstanceNorm2d))
#
#         # Up-Sampling
#         for i in range(n_down):
#             layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=2, padding=1, output_padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
#             layers.append(nn.ReLU(inplace=True))
#
#             layers.append(nn.Conv2d(curr_dim//2, curr_dim//2, kernel_size=k_size, stride=1, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
#             layers.append(nn.ReLU(inplace=True))
#
#             curr_dim = curr_dim // 2
#
#         layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
#         layers.append(nn.Tanh())
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x, c=None):
#         if c is not None:
#             # replicate spatially and concatenate domain information
#             c = c.unsqueeze(2).unsqueeze(3)
#             c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
#             x = torch.cat([x, c], dim=1)
#         return self.model(x)

class BGGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=9, k_size=4, n_down=2):
        super(BGGenerator, self).__init__()
        self._name = 'bg_generator'

        layers = []
        layers.append(nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(n_down):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=k_size, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, norm_fun=nn.InstanceNorm2d))

        # Up-Sampling
        for i in range(n_down):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=2, padding=1, output_padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(curr_dim//2, curr_dim//2, kernel_size=k_size, stride=1, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))

            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x, c=None):
        if c is not None:
            # replicate spatially and concatenate domain information
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)
        return self.model(x)


class ResUnetGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, k_size=4, n_down=2):
        super(ResUnetGenerator, self).__init__()
        self._name = 'resunet_generator'

        self.repeat_num = repeat_num
        self.n_down = n_down

        encoders = []

        encoders.append(nn.Sequential(
            nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True)
        ))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(n_down):
            encoders.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim*2, kernel_size=k_size, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

            curr_dim = curr_dim * 2

        self.encoders = nn.Sequential(*encoders)

        # Bottleneck
        resnets = []
        for i in range(repeat_num):
            resnets.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.resnets = nn.Sequential(*resnets)

        # Up-Sampling
        decoders = []
        skippers = []
        for i in range(n_down):
            decoders.append(nn.Sequential(
                nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=2, padding=1, output_padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

            skippers.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

            curr_dim = curr_dim // 2

        self.decoders = nn.Sequential(*decoders)
        self.skippers = nn.Sequential(*skippers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        self.attetion_reg = nn.Sequential(*layers)

    def inference(self, x):
        # encoder, 0, 1, 2, 3 -> [256, 128, 64, 32]
        encoder_outs = self.encode(x)

        # resnet, 32
        resnet_outs = []
        src_x = encoder_outs[-1]
        for i in range(self.repeat_num):
            src_x = self.resnets[i](src_x)
            resnet_outs.append(src_x)

        return encoder_outs, resnet_outs

    def forward(self, x):

        # encoder, 0, 1, 2, 3 -> [256, 128, 64, 32]
        encoder_outs = self.encode(x)

        # resnet, 32
        resnet_outs = self.resnets(encoder_outs[-1])

        # decoder, 0, 1, 2 -> [64, 128, 256]
        d_out = self.decode(resnet_outs, encoder_outs)

        img_outs, mask_outs = self.regress(d_out)
        return img_outs, mask_outs

    def encode(self, x):
        e_out = self.encoders[0](x)

        encoder_outs = [e_out]
        for i in range(1, self.n_down + 1):
            e_out = self.encoders[i](e_out)
            encoder_outs.append(e_out)
            #print(i, e_out.shape)
        return encoder_outs

    def decode(self, x, encoder_outs):
        d_out = x
        for i in range(self.n_down):
            d_out = self.decoders[i](d_out)  # x * 2
            skip = encoder_outs[self.n_down - 1 - i]
            d_out = torch.cat([skip, d_out], dim=1)
            d_out = self.skippers[i](d_out)
            #print(i, d_out.shape)
        return d_out

    def regress(self, x):
        return self.img_reg(x), self.attetion_reg(x)


class ResUnetAdaINGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, k_size=4, n_down=2):
        super(ResUnetAdaINGenerator, self).__init__()
        self._name = 'resunet_adain_generator'

        self.repeat_num = repeat_num
        self.n_down = n_down

        encoders = []
        encoders.append(nn.Sequential(
            nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True)
        ))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(n_down):
            encoders.append(AdaINBlock(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=k_size, stride=2, padding=1, bias=False)))
            curr_dim = curr_dim * 2

        self.encoders = nn.Sequential(*encoders)

        # Bottleneck
        resnets = []
        for i in range(repeat_num):
            resnets.append(AdaINResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.resnets = nn.Sequential(*resnets)

        # Up-Sampling
        decoders = []
        skippers = []
        for i in range(n_down):
            decoders.append(nn.Sequential(
                nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=2, padding=1, output_padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

            skippers.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            curr_dim = curr_dim // 2

        self.decoders = nn.Sequential(*decoders)
        self.skippers = nn.Sequential(*skippers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        self.attetion_reg = nn.Sequential(*layers)

    def forward(self, x, encode_outs, resnets_outs):
        assert self.n_down + 1 == len(encode_outs)
        assert self.repeat_num == len(resnets_outs)

        # encoder, 0, 1, 2, 3 -> [256, 128, 64, 32]
        encoder_outs = self.encode(x, encode_outs)

        # resnet, 32
        resnet_out = encoder_outs[-1]
        for i in range(self.repeat_num):
            resnet_out = self.resnets(resnet_out, resnets_outs[i])

        # decoder, 0, 1, 2 -> [64, 128, 256]
        d_out = self.decode(resnet_out, encoder_outs)

        img_outs, mask_outs = self.regress(d_out)
        return img_outs, mask_outs

    def encode(self, x, encode_outs):
        e_out = self.encoders[0](x, encode_outs[0])

        encoder_outs = [e_out]
        for i in range(1, self.n_down + 1):
            e_out = self.encoders[i](e_out, encode_outs[i])
            encoder_outs.append(e_out)
            # print(i, e_out.shape)
        return encoder_outs

    def decode(self, x, encoder_outs):
        d_out = x
        for i in range(self.n_down):
            d_out = self.decoders[i](d_out)  # x * 2
            skip = encoder_outs[self.n_down - 1 - i]
            d_out = torch.cat([skip, d_out], dim=1)
            d_out = self.skippers[i](d_out)
            # print(i, d_out.shape)
        return d_out

    def regress(self, x):
        return self.img_reg(x), self.attetion_reg(x)


class ImpersonatorAugGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, bg_dim, src_dim, tsf_dim, conv_dim=64, repeat_num=6):
        super(ImpersonatorAugGenerator, self).__init__()
        self._name = 'impersonator_aug_generator'

        self.n_down = 3
        self.repeat_num = repeat_num
        # background generator
        self.bg_model = BGGenerator(conv_dim=conv_dim, c_dim=bg_dim, repeat_num=1, k_size=3, n_down=5)

        # source generator
        self.src_model = ResUnetGenerator(conv_dim=conv_dim, c_dim=src_dim, repeat_num=repeat_num, k_size=3, n_down=self.n_down)

        # transfer generator
        self.tsf_model = ResUnetAdaINGenerator(conv_dim=conv_dim, c_dim=tsf_dim, repeat_num=repeat_num, k_size=3, n_down=self.n_down)

    def inference(self, src_encoder_outs, src_resnet_outs, tsf_inputs, T):
        # encoder
        src_x = src_encoder_outs[0]
        tsf_x = self.tsf_model.encoders[0](tsf_inputs)

        tsf_encoder_outs = [tsf_x]
        for i in range(1, self.n_down + 1):
            src_x = src_encoder_outs[i]
            warp = self.transform(src_x, T)
            tsf_x = self.tsf_model.encoders[i](tsf_x, src_x) + warp
            tsf_encoder_outs.append(tsf_x)

        # resnets
        T_scale = self.resize_trans(src_x, T)
        for i in range(self.repeat_num):
            src_x = src_resnet_outs[i]
            warp = self.stn(src_x, T_scale)
            tsf_x = self.tsf_model.resnets[i](tsf_x, src_x) + warp

        # decoders
        tsf_img, tsf_mask = self.tsf_model.regress(self.tsf_model.decode(tsf_x, tsf_encoder_outs))

        # print(front_rgb.shape, front_mask.shape)
        return tsf_img, tsf_mask

    def forward(self, bg_inputs, aug_bg_inputs, src_inputs, tsf_inputs, T):

        img_bg = self.bg_model(bg_inputs)
        aug_bg = self.bg_model(aug_bg_inputs)

        # encoder
        src_x = self.src_model.encoders[0](src_inputs)
        tsf_x = self.tsf_model.encoders[0](tsf_inputs)

        src_encoder_outs = [src_x]
        tsf_encoder_outs = [tsf_x]
        for i in range(1, self.n_down + 1):
            src_x = self.src_model.encoders[i](src_x)
            warp = self.transform(src_x, T)
            tsf_x = self.tsf_model.encoders[i](tsf_x, src_x) + warp

            src_encoder_outs.append(src_x)
            tsf_encoder_outs.append(tsf_x)

        # resnets
        T_scale = self.resize_trans(src_x, T)
        for i in range(self.repeat_num):
            src_x = self.src_model.resnets[i](src_x)
            warp = self.stn(src_x, T_scale)
            tsf_x = self.tsf_model.resnets[i](tsf_x, src_x) + warp

        # decoders
        src_img, src_mask = self.src_model.regress(self.src_model.decode(src_x, src_encoder_outs))
        tsf_img, tsf_mask = self.tsf_model.regress(self.tsf_model.decode(tsf_x, tsf_encoder_outs))

        fake_src_imgs = src_mask * img_bg + (1 - src_mask) * src_img
        fake_tsf_imgs = tsf_mask * img_bg + (1 - tsf_mask) * tsf_img

        # print(front_rgb.shape, front_mask.shape)
        return img_bg, aug_bg, fake_src_imgs, src_mask, fake_tsf_imgs, tsf_mask

    def resize_trans(self, x, T):
        _, _, h, w = x.shape

        T_scale = T.permute(0, 3, 1, 2)  # (bs, 2, h, w)
        T_scale = F.interpolate(T_scale, size=(h, w), mode='bilinear', align_corners=True)
        T_scale = T_scale.permute(0, 2, 3, 1)  # (bs, h, w, 2)

        return T_scale

    def stn(self, x, T):
        x_trans = F.grid_sample(x, T)

        return x_trans

    def transform(self, x, T):
        T_scale = self.resize_trans(x, T)
        x_trans = self.stn(x, T_scale)
        return x_trans


