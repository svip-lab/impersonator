import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


class GatedConv2dWithActivation(nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class GatedDeConv2dWithActivation(nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding,
                                                dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        # print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation, with_attn=False):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)   # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)   # transpose check
        attention = self.softmax(energy)   # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)   # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        if self.with_attn:
            return out, attention
        else:
            return out


class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self, c_dim=5):
        super(InpaintSANet, self).__init__()
        cnum = 32
        self.coarse_net = nn.Sequential(
            # input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv2dWithActivation(c_dim, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample 128
            GatedConv2dWithActivation(cnum, 2*cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample to 64
            GatedConv2dWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # atrous convlution
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # upsample
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            # Self_Attn(2*cnum, 'relu'),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            # Self_Attn(cnum//2, 'relu'),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        self.refine_conv_net = nn.Sequential(
            # input is 5*256*256
            GatedConv2dWithActivation(c_dim, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample
            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample
            GatedConv2dWithActivation(2*cnum, 2*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            # Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )
        self.refine_attn = SelfAttention(4 * cnum, 'relu', with_attn=False)
        self.refine_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            # Self_Attn(cnum, 'relu'),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )

    def forward(self, imgs, masks, only_out=False, only_x=False):
        # Coarse
        masked_imgs = imgs * (1 - masks) + masks
        input_imgs = torch.cat([masked_imgs, masks], dim=1)
        # print(input_imgs.size(), imgs.size(), masks.size())
        x = self.coarse_net(input_imgs)
        x = torch.clamp(x, -1., 1.)
        coarse_x = x
        # Refine
        masked_imgs = imgs * (1 - masks) + coarse_x * masks
        input_imgs = torch.cat([masked_imgs, masks], dim=1)
        x = self.refine_conv_net(input_imgs)
        x = self.refine_attn(x)
        # print(x.size(), attention.size())
        x = self.refine_upsample_net(x)
        x = torch.clamp(x, -1., 1.)

        comp_imgs = x * masks + imgs * (1 - masks)

        if only_out:
            return comp_imgs
        elif only_x:
            return x
        else:
            return coarse_x, x, comp_imgs
