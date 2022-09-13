################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-03-2022                                                             #
# Author: Florian Mies                                                         #
# Website: https://github.com/travela                                          #
################################################################################

"""
File to place any kind of generative models
and their respective helper functions.
"""

from abc import abstractmethod
from matplotlib import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.functional import relu, avg_pool2d
from avalanche.models.utils import MLP, Flatten
from avalanche.models.base_model import BaseModel


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetEncoder(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3, nclasses=10):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.nclasses = nclasses
        self.resblocks = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1),
            self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2),
            self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2),
            self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2),
        )

        self.features = nn.Sequential(nn.Linear(512, z_dim),nn.ReLU(inplace=True))
        self.classifier = nn.Linear(z_dim, nclasses)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resblocks(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def encode(self,x):
        x = self.resblocks(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return x



__all__ = ["ResNetEncoder"]