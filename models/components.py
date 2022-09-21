import math

import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from vendor.mtlpt import SABlock


class SxTAM(nn.Module):
    """Spatial cross-Task Attention Module"""
    def __init__(self, in_ch, s, use_alpha):
        super().__init__()

        ## Projection layers
        self.conv_b = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(),
                                    nn.Flatten(2))

        self.conv_c = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(),
                                    nn.Flatten(2))

        self.conv_d = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

        ## Channel-wise weights
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1, in_ch, 1, 1))

        self.down = nn.MaxPool2d(s, s)

        self.d = nn.Upsample(scale_factor=1/s)
        self.u = nn.Upsample(scale_factor=s)


    def forward(self, x, y):
        # downscale and flatten spatial dimension
        x_ = self.d(x)
        B = self.conv_b(x_).transpose(1,2)
        C = self.conv_c(self.down(y))
        D = self.conv_d(self.down(y))

        # compute correlation matrix
        # (b, hw_x, c) @ (b, c, hw_y) = (b, hw_x, hw_y) -T-> (b, hw_y, hw_x)
        coeff = math.sqrt(B.size(2))
        corr = self.softmax(B @ C / coeff).transpose(1,2)

        # (b, c, hw_y) @ (b, hw_y, hw_x) = (b, c, hw_x) -view-> (b, c, h_x, w_x)
        out = self.u((D.flatten(2) @ corr).view_as(x_))

        if self.use_alpha:
            out *= self.alpha

        return out

class Project(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU())

class Extractor(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU())