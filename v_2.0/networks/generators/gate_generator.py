import torch.nn as nn
import torch.nn.functional as F
from networks.networks import NetworkBase
import torch
import ipdb


class GateConvBlock(nn.Module):

    def __init__(self, c_in, c_out, ksize, stride=1, rate=1, activation='leaky_relu', use_norm=True):
        super(GateConvBlock, self).__init__()

        self.ksize = ksize
        self.stride = 1
        self.dilation = rate

        # feature convolutions
        feat_convs = list()
        feat_convs.append(nn.Conv2d(c_in, c_out, kernel_size=ksize, stride=stride, dilation=rate, padding=0))
        if use_norm:
            feat_convs.append(nn.InstanceNorm2d(c_out))
        if activation == 'leaky_relu':
            feat_convs.append(nn.LeakyReLU(0.02, inplace=True))
        self.feat_convs = nn.Sequential(*feat_convs)

        # mask convlutions
        self.mask_convs = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ksize, stride=stride, dilation=rate, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.same_pad(x)
        feats = self.feat_convs(x)
        masks = self.mask_convs(x)
        feats = feats * masks
        return feats, masks

    def same_pad(self, x):
        h, w = x.shape[2:]
        padding_rows = self.calculate_pad(h, self.ksize, self.stride, self.dilation)
        padding_cols = self.calculate_pad(w, self.ksize, self.stride, self.dilation)

        rows_odd = (padding_rows % 2 != 0)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            x = F.pad(x, (0, int(cols_odd), 0, int(rows_odd)))
        x = F.pad(x, (padding_rows // 2, padding_rows // 2, padding_cols // 2, padding_cols // 2))
        return x

    @staticmethod
    def calculate_pad(input_x, k_x, stride, dilation):
        effective_k_x = (k_x - 1) * dilation + 1
        out_x = (input_x + stride - 1) // stride
        padding_needed = max(0, (out_x - 1) * stride + effective_k_x - input_x)
        padding_x = max(0, (out_x - 1) * stride +
                           (k_x - 1) * dilation + 1 - input_x)
        return padding_x


class GateDeConvBlock(nn.Module):

    def __init__(self, c_in, c_out, ksize, stride=1, activation='leaky_relu'):
        super(GateDeConvBlock, self).__init__()

        self.ksize = ksize
        self.stride = 1

        # feature convolutions
        feat_convs = list()
        feat_convs.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=ksize, stride=stride,
                                             padding=ksize//2, output_padding=1))
        feat_convs.append(nn.InstanceNorm2d(c_out))
        if activation == 'leaky_relu':
            feat_convs.append(nn.LeakyReLU(0.02, inplace=True))
        self.feat_convs = nn.Sequential(*feat_convs)

        # mask convlutions
        self.mask_convs = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=ksize, stride=stride,
                               padding=ksize//2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.feat_convs(x)
        masks = self.mask_convs(x)
        feats = feats * masks
        return feats, masks


class GateUnetGenerator(NetworkBase):

    def __init__(self, c_dim, cnum=64):
        super(GateUnetGenerator, self).__init__()

        # encoder
        self.gate_conv1 = GateConvBlock(c_in=c_dim, c_out=cnum, ksize=7, stride=2)     # 128 * 128 * cnum
        self.gate_conv2 = GateConvBlock(c_in=cnum, c_out=cnum*2, ksize=5, stride=2)    # 64 * 64 * (2*cnum)
        self.gate_conv3 = GateConvBlock(c_in=cnum*2, c_out=cnum*4, ksize=5, stride=2)  # 32 * 32 * (4*cnum)
        self.gate_conv4 = GateConvBlock(c_in=cnum*4, c_out=cnum*8, ksize=3, stride=2)  # 16 * 16 * (8*cnum)
        self.gate_conv5 = GateConvBlock(c_in=cnum*8, c_out=cnum*8, ksize=3, stride=2)  # 8 * 8 * (8*cnum)
        # self.gate_conv6 = GateConvBlock(c_in=cnum*8, c_out=cnum*8, ksize=3, stride=2)  # 4 * 4 * (8*cnum)
        # self.gate_conv7 = GateConvBlock(c_in=cnum*8, c_out=cnum*8, ksize=3, stride=2)  # 2 * 2 * (8*cnum)

        # dilated conv
        self.dilated_conv = nn.ModuleList([
            GateConvBlock(c_in=cnum*8, c_out=cnum*8, ksize=3, stride=1, rate=2),
            GateConvBlock(c_in=cnum*8, c_out=cnum*8, ksize=3, stride=1, rate=4),
            GateConvBlock(c_in=cnum*8, c_out=cnum*8, ksize=3, stride=1, rate=8),
            GateConvBlock(c_in=cnum*8, c_out=cnum*8, ksize=3, stride=1, rate=16)
        ])

        # decoder
        # ### 2 x 2 x (cnum*8) -> 4 x 4 x (cnum*8)
        # self.gate_deconv7 = nn.ModuleList([
        #     GateDeConvBlock(c_in=cnum * 8, c_out=cnum * 8, ksize=3, stride=2),
        #     GateConvBlock(c_in=cnum * 16, c_out=cnum * 8, ksize=3, stride=1)
        # ])
        #
        # ### 4 x 4 x (cnum*8) -> 8 x 8 x (cnum*8)
        # self.gate_deconv6 = nn.ModuleList([
        #     GateDeConvBlock(c_in=cnum * 8, c_out=cnum * 8, ksize=3, stride=2),
        #     GateConvBlock(c_in=cnum * 16, c_out=cnum * 8, ksize=3, stride=1)
        # ])

        ### 8 x 8 x (cnum*8) -> 16 x 16 x (cnum*8)
        self.gate_deconv5 = nn.ModuleList([
            GateDeConvBlock(c_in=cnum * 8, c_out=cnum * 8, ksize=3, stride=2),
            GateConvBlock(c_in=cnum * 16, c_out=cnum * 8, ksize=3, stride=1)
        ])

        ### 16 x 16 x (cnum*8) -> 32 x 32 x (cnum*4)
        self.gate_deconv4 = nn.ModuleList([
            GateDeConvBlock(c_in=cnum * 8, c_out=cnum * 4, ksize=3, stride=2),
            GateConvBlock(c_in=cnum * 8, c_out=cnum * 4, ksize=3, stride=1)
        ])

        ### 32 x 32 x (cnum*8) -> 64 x 64 x (cnum*4)
        self.gate_deconv3 = nn.ModuleList([
            GateDeConvBlock(c_in=cnum * 4, c_out=cnum * 2, ksize=3, stride=2),
            GateConvBlock(c_in=cnum * 4, c_out=cnum * 2, ksize=3, stride=1)
        ])

        ### 64 x 64 x (cnum*4) -> 128 x 128 x (cnum*2)
        self.gate_deconv2 = nn.ModuleList([
            GateDeConvBlock(c_in=cnum * 2, c_out=cnum * 1, ksize=3, stride=2),
            GateConvBlock(c_in=cnum * 2, c_out=cnum * 1, ksize=3, stride=1)
        ])

        ### 128 x 128 x (cnum*2) -> 256 x 256 x (cnum*1)
        self.gate_deconv1 = nn.ModuleList([
            GateDeConvBlock(c_in=cnum * 1, c_out=3, ksize=3, stride=2),
            GateConvBlock(c_in=c_dim+3, c_out=3, ksize=3, stride=1)
        ])

    def forward(self, x):
        # incom_imgs = images * (1 - masks)

        # encoder
        x1, mask1 = self.gate_conv1(x)      # 128 x 128 x (1*cnum)
        x2, mask2 = self.gate_conv2(x1)     # 64  x 64  x (2*cnum)
        x3, mask3 = self.gate_conv3(x2)     # 32  x 32  x (4*cnum)
        x4, mask4 = self.gate_conv4(x3)     # 16  x 16  x (8*cnum)
        x5, mask5 = self.gate_conv5(x4)     # 8   x 8   x (8*cnum)
        # x6, mask6 = self.gate_conv6(x5)     # 4   x 4   x (8*cnum)
        # x7, mask7 = self.gate_conv7(x6)     # 2   x 2   x (8*cnum)

        # dilated conv
        dilated_x = x5
        for dilate_conv in self.dilated_conv:
            dilated_x, _ = dilate_conv(dilated_x)  # 2   x 2   x (8*cnum)
            # print('dilated', dilated_x7.shape)

        # decoder
        # dx7, _ = self.gate_deconv7[0](dilated_x)  # 4   x 4   x (8*cnum)
        # dx7 = torch.cat([x6, dx7], dim=1)          # 4   x 4   x (16*cnum)
        # dx7, dmask8 = self.gate_deconv7[1](dx7)    # 4   x 4   x (8*cnum)
        #
        # dx6, _ = self.gate_deconv6[0](dx7)         # 8   x 8   x (8*cnum)
        # dx6 = torch.cat([x5, dx6], dim=1)          # 8   x 8   x (16*cnum)
        # dx6, dmask9 = self.gate_deconv6[1](dx6)    # 8   x 8   x (8*cnum)

        dx5, _ = self.gate_deconv5[0](dilated_x)   # 16  x 16  x (8*cnum)
        dx5 = torch.cat([x4, dx5], dim=1)          # 16  x 16  x (16*cnum)
        dx5, dmask10 = self.gate_deconv5[1](dx5)   # 16  x 16  x (8*cnum)

        dx4, _ = self.gate_deconv4[0](dx5)         # 32  x 32  x (4*cnum)
        dx4 = torch.cat([x3, dx4], dim=1)          # 32  x 32  x (8*cnum)
        dx4, dmask11 = self.gate_deconv4[1](dx4)   # 32  x 32  x (4*cnum)

        dx3, _ = self.gate_deconv3[0](dx4)         # 64  x 64  x (2*cnum)
        dx3 = torch.cat([x2, dx3], dim=1)          # 64  x 64  x (4*cnum)
        dx3, dmask12 = self.gate_deconv3[1](dx3)   # 64  x 64  x (2*cnum)

        dx2, _ = self.gate_deconv2[0](dx3)         # 128 x 128 x (1*cnum)
        dx2 = torch.cat([x1, dx2], dim=1)          # 128 x 128 x (2*cnum)
        dx2, dmask13 = self.gate_deconv2[1](dx2)   # 128 x 128 x (1*cnum)

        dx1, _ = self.gate_deconv1[0](dx2)         # 256 x 256 x 3
        dx1 = torch.cat([x, dx1], dim=1)           # 256 x 256 x 6
        dx1, dmask14 = self.gate_deconv1[1](dx1)   # 256 x 256 x 3
        output = torch.tanh(dx1)
        comp_imgs = x[:, 0:3, ...] + output * x[:, 3:, ...]
        return output, comp_imgs


if __name__ == '__main__':
    x = torch.rand(2, 4, 256, 256)
    generator = GateUnetGenerator(c_dim=4)
    out, mask = generator(x)
    print(out.shape, mask.shape)

    # gate_conv = GateConvBlock(c_in=32, c_out=64, ksize=5, stride=2, rate=16, activation='leaky_relu')
    # x = torch.rand(2, 32, 16, 16)
    # feats, masks = gate_conv(x)
    # print(feats.shape)

    # gate_conv = GateDeConvBlock(c_in=32, c_out=64, ksize=7, stride=2, activation='leaky_relu')
    # x = torch.rand(2, 32, 128, 128)
    # feats, masks = gate_conv(x)
    # print(feats.shape)

    ipdb.set_trace()

    # imitator = ImpersonatorGenerator(bg_dim=4, src_dim=6, tsf_dim=6, conv_dim=64, repeat_num=6)
    #
    # bg_x = torch.rand(2, 4, 256, 256)
    # src_x = torch.rand(2, 6, 256, 256)
    # tsf_x = torch.rand(2, 6, 256, 256)
    # T = torch.rand(2, 256, 256, 2)
    #
    # img_bg, src_img, src_mask, tsf_img, tsf_mask = imitator(bg_x, src_x, tsf_x, T)
    #
    # ipdb.set_trace()