import torch.nn as nn
import torch.nn.functional as F
from .networks import NetworkBase
import torch
import ipdb


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class ResNetGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=9, k_size=4, n_down=2):
        super(ResNetGenerator, self).__init__()
        self._name = 'resnet_generator'

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
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(n_down):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=2, padding=1,
                                             output_padding=1, bias=False))
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
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True)
        ))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(n_down):
            encoders.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim*2, kernel_size=k_size, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim*2, affine=True),
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
                nn.InstanceNorm2d(curr_dim//2, affine=True),
                nn.ReLU(inplace=True)
            ))

            skippers.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim//2, affine=True),
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
            # print(i, d_out.shape)
        return d_out

    def regress(self, x):
        return self.img_reg(x), self.attetion_reg(x)


class ImpersonatorGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, bg_dim, src_dim, tsf_dim, conv_dim=64, repeat_num=6):
        super(ImpersonatorGenerator, self).__init__()
        self._name = 'impersonator_generator'

        self.n_down = 3
        self.repeat_num = repeat_num
        # background generator
        self.bg_model = ResNetGenerator(conv_dim=conv_dim, c_dim=bg_dim, repeat_num=repeat_num, k_size=3, n_down=self.n_down)

        # source generator
        self.src_model = ResUnetGenerator(conv_dim=conv_dim, c_dim=src_dim, repeat_num=repeat_num, k_size=3, n_down=self.n_down)

        # transfer generator
        self.tsf_model = ResUnetGenerator(conv_dim=conv_dim, c_dim=tsf_dim, repeat_num=repeat_num, k_size=3, n_down=self.n_down)

    def forward(self, bg_inputs, src_inputs, tsf_inputs, T):

        img_bg = self.bg_model(bg_inputs)

        src_img, src_mask, tsf_img, tsf_mask = self.infer_front(src_inputs, tsf_inputs, T)

        # print(front_rgb.shape, front_mask.shape)
        return img_bg, src_img, src_mask, tsf_img, tsf_mask

    def encode_src(self, src_inputs):
        return self.src_model.inference(src_inputs)

    def infer_front(self, src_inputs, tsf_inputs, T):
        # encoder
        src_x = self.src_model.encoders[0](src_inputs)
        tsf_x = self.tsf_model.encoders[0](tsf_inputs)

        src_encoder_outs = [src_x]
        tsf_encoder_outs = [tsf_x]
        for i in range(1, self.n_down + 1):
            src_x = self.src_model.encoders[i](src_x)
            warp = self.transform(src_x, T)
            tsf_x = self.tsf_model.encoders[i](tsf_x) + warp

            src_encoder_outs.append(src_x)
            tsf_encoder_outs.append(tsf_x)

        # resnets
        T_scale = self.resize_trans(src_x, T)
        for i in range(self.repeat_num):
            src_x = self.src_model.resnets[i](src_x)
            warp = self.stn(src_x, T_scale)
            tsf_x = self.tsf_model.resnets[i](tsf_x) + warp

        # decoders
        src_img, src_mask = self.src_model.regress(self.src_model.decode(src_x, src_encoder_outs))
        tsf_img, tsf_mask = self.tsf_model.regress(self.tsf_model.decode(tsf_x, tsf_encoder_outs))

        # print(front_rgb.shape, front_mask.shape)
        return src_img, src_mask, tsf_img, tsf_mask

    def swap(self, tsf_inputs, src_encoder_outs12, src_encoder_outs21, src_resnet_outs12, src_resnet_outs21, T12, T21):
        # encoder
        src_x12 = src_encoder_outs12[0]
        src_x21 = src_encoder_outs21[0]
        tsf_x = self.tsf_model.encoders[0](tsf_inputs)

        tsf_encoder_outs = [tsf_x]
        for i in range(1, self.n_down + 1):
            src_x12 = src_encoder_outs12[i]
            src_x21 = src_encoder_outs21[i]
            warp12 = self.transform(src_x12, T12)
            warp21 = self.transform(src_x21, T21)

            tsf_x = self.tsf_model.encoders[i](tsf_x) + warp12 + warp21
            tsf_encoder_outs.append(tsf_x)

        # resnets
        T_scale12 = self.resize_trans(src_x12, T12)
        T_scale21 = self.resize_trans(src_x21, T21)
        for i in range(self.repeat_num):
            src_x12 = src_resnet_outs12[i]
            src_x21 = src_resnet_outs21[i]
            warp12 = self.stn(src_x12, T_scale12)
            warp21 = self.stn(src_x21, T_scale21)
            tsf_x = self.tsf_model.resnets[i](tsf_x) + warp12 + warp21

        # decoders
        tsf_img, tsf_mask = self.tsf_model.regress(self.tsf_model.decode(tsf_x, tsf_encoder_outs))

        # print(front_rgb.shape, front_mask.shape)
        return tsf_img, tsf_mask

    def inference(self, src_encoder_outs, src_resnet_outs, tsf_inputs, T):
        # encoder
        src_x = src_encoder_outs[0]
        tsf_x = self.tsf_model.encoders[0](tsf_inputs)

        tsf_encoder_outs = [tsf_x]
        for i in range(1, self.n_down + 1):
            src_x = src_encoder_outs[i]
            warp = self.transform(src_x, T)

            tsf_x = self.tsf_model.encoders[i](tsf_x) + warp
            tsf_encoder_outs.append(tsf_x)

        # resnets
        T_scale = self.resize_trans(src_x, T)
        for i in range(self.repeat_num):
            src_x = src_resnet_outs[i]
            warp = self.stn(src_x, T_scale)
            tsf_x = self.tsf_model.resnets[i](tsf_x) + warp

        # decoders
        tsf_img, tsf_mask = self.tsf_model.regress(self.tsf_model.decode(tsf_x, tsf_encoder_outs))

        # print(front_rgb.shape, front_mask.shape)
        return tsf_img, tsf_mask

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


if __name__ == '__main__':
    imitator = ImpersonatorGenerator(bg_dim=4, src_dim=6, tsf_dim=6, conv_dim=64, repeat_num=6)

    bg_x = torch.rand(2, 4, 256, 256)
    src_x = torch.rand(2, 6, 256, 256)
    tsf_x = torch.rand(2, 6, 256, 256)
    T = torch.rand(2, 256, 256, 2)

    img_bg, src_img, src_mask, tsf_img, tsf_mask = imitator(bg_x, src_x, tsf_x, T)

    ipdb.set_trace()
