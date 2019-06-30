import torch
import torch.nn as nn
import torchvision.models as models


class Vgg19(torch.nn.Module):
    """
    Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (1): ReLU(inplace)
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace)
          (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (6): ReLU(inplace)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU(inplace)
          (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (11): ReLU(inplace)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU(inplace)
          (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU(inplace)
          (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (17): ReLU(inplace)
          (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (20): ReLU(inplace)
          (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU(inplace)
          (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (24): ReLU(inplace)
          (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (26): ReLU(inplace)
          (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (29): ReLU(inplace)
          xxxx(30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(31): ReLU(inplace)
          xxxx(32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(33): ReLU(inplace)
          xxxx(34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(35): ReLU(inplace)
          xxxx(36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    """

    def __init__(self, requires_grad=False, before_relu=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        print('loading vgg19 ...')

        if before_relu:
            slice_ids = [1, 6, 11, 20, 29]
        else:
            slice_ids = [2, 7, 12, 21, 30]

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(slice_ids[0]):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[0], slice_ids[1]):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[1], slice_ids[2]):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[2], slice_ids[3]):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[3], slice_ids[4]):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_out1 = self.slice1(X)
        h_out2 = self.slice2(h_out1)
        h_out3 = self.slice3(h_out2)
        h_out4 = self.slice4(h_out3)
        h_out5 = self.slice5(h_out4)
        out = [X, h_out1, h_out2, h_out3, h_out4, h_out5]
        # out = [h_out1, h_out2, h_out3, h_out4, h_out5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, before_relu=False):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(before_relu=before_relu).cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
