# -*- coding: utf-8 -*-
# @Time    : 2018/10/29 8:20 PM
# @Author  : Zhixin Piao
# @Email   : piaozhx@shanghaitech.edu.cn

'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import h5py

from .batch_smpl import SMPL, batch_orth_proj_idrot


def subsample(inputs, factor):
    """Subsamples the input along the spatial dimensions.

    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.

    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return F.max_pool2d(inputs, [1, 1], stride=factor)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.stride = stride

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        # tf implementation there needs bias
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=True)

        self.stride = stride

        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(
        #         # tf implementation there needs bias
        #         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)
        #     )
        if in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                # tf implementation there needs bias
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x, layer_id='layer1', block_id='0', results_outs=OrderedDict()):
        # out = F.relu(self.bn1(x))
        # shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # out = self.conv1(out)
        # out = self.conv2(F.relu(self.bn2(out)))
        # out = self.conv3(F.relu(self.bn3(out)))
        # out += shortcut

        # layer1.0.bn1
        out_name = '{}.{}'.format(layer_id, block_id)

        preact = F.relu(self.bn1(x))
        shortcut_out = self.shortcut(preact) if hasattr(self, 'shortcut') else subsample(x, factor=self.stride)
        conv1_out = F.relu(self.bn2(self.conv1(preact)))
        conv2_out = F.relu(self.bn3(self.conv2(conv1_out)))
        conv3_out = self.conv3(conv2_out)
        conv3_out_add = conv3_out + shortcut_out

        results_outs[out_name + '.shortcut.0'] = shortcut_out
        results_outs[out_name + '.conv1'] = conv1_out
        results_outs[out_name + '.conv2'] = conv2_out
        results_outs[out_name + '.conv3'] = conv3_out
        results_outs[out_name] = conv3_out_add

        return conv3_out_add


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.post_bn = nn.BatchNorm2d(2048)

    def _make_layer(self, block, planes, num_blocks, stride):
        # layers = []
        # layers.append(block(self.in_planes, planes, stride))
        #
        # self.in_planes = planes * block.expansion
        # for i in range(1, num_blocks):
        #     layers.append(block(self.in_planes, planes))
        # return nn.Sequential(*layers)

        layers = []
        layers.append(block(self.in_planes, planes, 1))

        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            s = stride if i == num_blocks - 1 else 1
            layers.append(block(self.in_planes, planes, stride=s))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # here need padding =1
        # out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = F.max_pool2d(out, kernel_size=3, stride=2, ceil_mode=True)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.post_bn(out))

        # need global avg_pooling
        out = F.avg_pool2d(out, 7)

        out = out.view(out.size(0), -1)
        return out


def preActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])


def preActResNet34():
    return PreActResNet(PreActBlock, [3, 4, 6, 3])


def preActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])


def preActResNet101():
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3])


def preActResNet152():
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3])


def load_mean_theta(smpl_mean_theta_path, total_theta_count=85):
    mean = np.zeros(total_theta_count, dtype=np.float)

    if smpl_mean_theta_path:
        mean_values = h5py.File(smpl_mean_theta_path, 'r')
        mean_pose = mean_values['pose'][...]
        # Ignore the global rotation.
        mean_pose[:3] = 0
        mean_shape = mean_values['shape'][...]

        # This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi

        # Initialize scale at 0.9
        mean[0] = 0.9
        mean[3:75] = mean_pose[:]
        mean[75:] = mean_shape[:]

        mean_values.close()
    else:
        mean[0] = 0.9

    return mean


class ThetaRegressor(nn.Module):
    def __init__(self, input_dim, out_dim, iterations=3):
        super(ThetaRegressor, self).__init__()
        self.iterations = iterations

        # register mean theta
        self.register_buffer('mean_theta', torch.rand(out_dim, dtype=torch.float32))

        fc_blocks = OrderedDict()

        fc_blocks['fc1'] = nn.Linear(input_dim, 1024, bias=True)
        fc_blocks['relu1'] = nn.ReLU()
        fc_blocks['dropout1'] = nn.Dropout(p=0.5)

        fc_blocks['fc2'] = nn.Linear(1024, 1024, bias=True)
        fc_blocks['relu2'] = nn.ReLU()

        fc_blocks['dropout2'] = nn.Dropout(p=0.5)

        fc_blocks['fc3'] = nn.Linear(1024, out_dim, bias=True)
        # small_xavier initialization
        nn.init.xavier_normal_(fc_blocks['fc3'].weight, gain=0.1)
        nn.init.zeros_(fc_blocks['fc3'].bias)

        self.fc_blocks = nn.Sequential(fc_blocks)

    def forward(self, x):
        """
        :param x: the output of encoder, 2048 dim
        :return: a list contains [[theta1, theta1, ..., theta1],
                                 [theta2, theta2, ..., theta2], ... , ],
                shape is iterations X N X 85(or other theta count)
        """
        batch_size = x.shape[0]
        theta = self.mean_theta.repeat(batch_size, 1)
        for _ in range(self.iterations):
            total_inputs = torch.cat([x, theta], dim=1)
            theta = theta + self.fc_blocks(total_inputs)

        return theta


class HumanModelRecovery(nn.Module):
    """
        regressor can predict betas(include beta and theta which needed by SMPL) from coder
        extracted from encoder in a iteration way
    """

    def __init__(self, smpl_pkl_path, feature_dim=2048, theta_dim=85, iterations=3):
        super(HumanModelRecovery, self).__init__()

        # define resnet50_v2
        self.resnet = preActResNet50()

        # define smpl
        self.smpl = SMPL(pkl_path=smpl_pkl_path)

        self.feature_dim = feature_dim
        self.theta_dim = theta_dim

        self.regressor = ThetaRegressor(feature_dim + theta_dim, theta_dim, iterations)
        self.iterations = iterations

    def forward(self, inputs):
        out = self.resnet.conv1(inputs)

        # here need padding =1
        # out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = F.max_pool2d(out, kernel_size=3, stride=2, ceil_mode=True)

        out = self.resnet.layer1(out)

        out = self.resnet.layer2(out)

        out = self.resnet.layer3(out)

        out = self.resnet.layer4(out)

        out = F.relu(self.resnet.post_bn(out))
        # need global avg_pooling
        out = F.avg_pool2d(out, 7)

        features = out.view(out.size(0), -1)

        # regressor
        thetas = self.regressor(features)

        return thetas

    def get_details(self, theta):
        """
            purpose:
                calc verts, joint2d, joint3d, Rotation matrix

            inputs:
                theta: N X (3 + 72 + 10)

            return:
                thetas, verts, j2d, j3d, Rs
        """

        cam = theta[:, 0:3].contiguous()
        pose = theta[:, 3:75].contiguous()
        shape = theta[:, 75:].contiguous()
        verts, j3d, rs = self.smpl(beta=shape, theta=pose, get_skin=True)
        j2d = batch_orth_proj_idrot(j3d, cam)

        detail_info = {
            'theta': theta,
            'cam': cam,
            'pose': pose,
            'shape': shape,
            'verts': verts,
            'j2d': j2d,
            'j3d': j3d
        }

        return detail_info
