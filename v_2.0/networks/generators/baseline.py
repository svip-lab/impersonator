import torch.nn as nn
import torch.nn.functional as F
from networks.networks import NetworkBase
import torch
import ipdb


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, norm_layer):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class ResNetGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=9, k_size=4, n_down=2, norm_layer=nn.InstanceNorm2d):
        super(ResNetGenerator, self).__init__()
        self._name = 'resnet_generator'

        layers = []
        layers.append(nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(norm_layer(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(n_down):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=k_size, stride=2, padding=1, bias=False))
            layers.append(norm_layer(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, norm_layer=norm_layer))

        # Up-Sampling
        for i in range(n_down):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=2, padding=1, output_padding=1, bias=False))
            layers.append(norm_layer(curr_dim//2, affine=True))
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
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, k_size=4, n_down=2, norm_layer=nn.InstanceNorm2d):
        super(ResUnetGenerator, self).__init__()
        self._name = 'resunet_generator'

        self.repeat_num = repeat_num
        self.n_down = n_down

        encoders = []

        encoders.append(nn.Sequential(
            nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            norm_layer(conv_dim, affine=True),
            nn.ReLU(inplace=True)
        ))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(n_down):
            encoders.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim*2, kernel_size=k_size, stride=2, padding=1, bias=False),
                norm_layer(curr_dim*2, affine=True),
                nn.ReLU(inplace=True)
            ))

            curr_dim = curr_dim * 2

        self.encoders = nn.Sequential(*encoders)

        # Bottleneck
        resnets = []
        for i in range(repeat_num):
            resnets.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, norm_layer=norm_layer))

        self.resnets = nn.Sequential(*resnets)

        # Up-Sampling
        decoders = []
        skippers = []
        for i in range(n_down):
            decoders.append(nn.Sequential(
                nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=2, padding=1, output_padding=1, bias=False),
                norm_layer(curr_dim//2, affine=True),
                nn.ReLU(inplace=True)
            ))

            skippers.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=1, padding=1, bias=False),
                norm_layer(curr_dim//2, affine=True),
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


class ConcatOrTextureWarpingGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, bg_dim, src_dim, tsf_dim, conv_dim=64, repeat_num=6, norm_type='instance'):
        super(ConcatOrTextureWarpingGenerator, self).__init__()
        self._name = 'concat_or_texture_warping_generator'

        norm_layer = self._get_norm_layer(norm_type)

        self.n_down = 3
        self.repeat_num = repeat_num
        # background generator
        self.bg_model = ResNetGenerator(conv_dim=conv_dim, c_dim=bg_dim, repeat_num=repeat_num,
                                        k_size=3, n_down=self.n_down, norm_layer=norm_layer)

        # transfer generator
        self.tsf_model = ResUnetGenerator(conv_dim=conv_dim, c_dim=3 + src_dim + tsf_dim, repeat_num=repeat_num,
                                          k_size=3, n_down=self.n_down, norm_layer=norm_layer)

    def forward(self, bg_inputs, inputs):

        img_bg = self.bg_model(bg_inputs)

        tsf_img, tsf_mask = self.tsf_model(inputs)

        # print(front_rgb.shape, front_mask.shape)
        return img_bg, tsf_img, tsf_mask

    def inference(self, inputs):
        tsf_img, tsf_mask = self.tsf_model(inputs)

        # print(front_rgb.shape, front_mask.shape)
        return tsf_img, tsf_mask


class FeatureWarpingGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, bg_dim, src_dim, tsf_dim, conv_dim=64, repeat_num=6, norm_type='instance'):
        super(FeatureWarpingGenerator, self).__init__()
        self._name = 'feature_warping_generator'

        norm_layer = self._get_norm_layer(norm_type)

        self.n_down = 3
        self.repeat_num = repeat_num
        # background generator
        self.bg_model = ResNetGenerator(conv_dim=conv_dim, c_dim=bg_dim, repeat_num=repeat_num,
                                        k_size=3, n_down=self.n_down, norm_layer=norm_layer)

        self.src_encoders, self.src_residuals, curr_dim = self.enc_res_blocks(c_dim=src_dim, conv_dim=conv_dim,
                                                                              repeat_num=0,
                                                                              n_down=self.n_down,
                                                                              k_size=3, norm_layer=norm_layer)
        self.tsf_encoders, self.tsf_residuals, curr_dim = self.enc_res_blocks(c_dim=3+tsf_dim, conv_dim=conv_dim,
                                                                              repeat_num=repeat_num,
                                                                              n_down=self.n_down,
                                                                              k_size=3, norm_layer=norm_layer)

        self.decoders, self.skippers, self.img_reg, self.attention_reg = self.decoder_blcoks(
            curr_dim=curr_dim, n_down=self.n_down, k_size=3)

    def enc_res_blocks(self, c_dim, conv_dim, repeat_num, n_down, k_size, norm_layer=nn.InstanceNorm2d):
        encoders = list()

        encoders.append(nn.Sequential(
            nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            norm_layer(conv_dim, affine=True),
            nn.ReLU(inplace=True)
        ))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(n_down):
            encoders.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim*2, kernel_size=k_size, stride=2, padding=1, bias=False),
                norm_layer(curr_dim*2, affine=True),
                nn.ReLU(inplace=True)
            ))

            curr_dim = curr_dim * 2

        # Bottleneck
        resnets = list()
        for i in range(repeat_num):
            resnets.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, norm_layer=norm_layer))

        return nn.Sequential(*encoders), nn.Sequential(*resnets), curr_dim

    def decoder_blcoks(self, curr_dim, n_down, k_size, norm_layer=nn.InstanceNorm2d):
        # Up-Sampling
        decoders = []
        skippers = []
        for i in range(n_down):
            decoders.append(nn.Sequential(
                nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=2, padding=1, output_padding=1, bias=False),
                norm_layer(curr_dim//2, affine=True),
                nn.ReLU(inplace=True)
            ))

            skippers.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=1, padding=1, bias=False),
                norm_layer(curr_dim//2, affine=True),
                nn.ReLU(inplace=True)
            ))

            curr_dim = curr_dim // 2

        decoders = nn.Sequential(*decoders)
        skippers = nn.Sequential(*skippers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        img_reg = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        attention_reg = nn.Sequential(*layers)

        return decoders, skippers, img_reg, attention_reg

    def forward(self, bg_inputs, src_inputs, tsf_inputs):

        img_bg = self.bg_model(bg_inputs)

        # encoder
        src_x = self.src_encoders[0](src_inputs)
        tsf_x = self.tsf_encoders[0](tsf_inputs)

        src_encoder_outs = [src_x]
        for i in range(1, self.n_down + 1):
            src_x = self.src_encoders[i](src_x)
            tsf_x = self.tsf_encoders[i](tsf_x)
            src_encoder_outs.append(src_x)

        # resnets
        tsf_x = self.tsf_residuals(tsf_x)

        # decoders
        tsf_img, tsf_mask = self.regress(self.decode(tsf_x, src_encoder_outs))

        # print(front_rgb.shape, front_mask.shape)
        return img_bg, tsf_img, tsf_mask

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
        return self.img_reg(x), self.attention_reg(x)

    def swap(self, tsf_inputs, src_encoder_outs12, src_encoder_outs21, src_resnet_outs12, src_resnet_outs21, T12, T21):
        raise NotImplementedError

    def inference(self, src_encoder_outs, src_resnet_outs, tsf_inputs, T):
        raise NotImplementedError


if __name__ == '__main__':
    # imitator = ConcatOrTextureWarpingGenerator(bg_dim=4, src_dim=3, tsf_dim=3, conv_dim=64, repeat_num=6)
    #
    # bg_x = torch.rand(2, 4, 256, 256)
    # x = torch.rand(2, 9, 256, 256)
    #
    # img_bg, tsf_img, tsf_mask = imitator(bg_x, x)
    #
    # ipdb.set_trace()

    imitator = FeatureWarpingGenerator(bg_dim=4, src_dim=3, tsf_dim=6, conv_dim=64, repeat_num=6)

    bg_x = torch.rand(2, 4, 256, 256)
    src_inputs = torch.rand(2, 3, 256, 256)
    tsf_inputs = torch.rand(2, 6, 256, 256)

    img_bg, tsf_img, tsf_mask = imitator(bg_x, src_inputs, tsf_inputs)
    print(img_bg.shape, tsf_img.shape, tsf_mask.shape)

    ipdb.set_trace()