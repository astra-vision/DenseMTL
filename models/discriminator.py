import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchvision.transforms import RandomHorizontalFlip


class SpectralConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x):
        return self.f(x)

# As described in Wang et al., Semi-supervised Multi-task Learning for Semantics and Depth
class Discriminator(nn.Sequential):
    def __init__(self, in_ch, out_ch=1, ndf=64):
        super().__init__(
            RandomHorizontalFlip(), # this wasn't on master!
            SpectralConv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SpectralConv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SpectralConv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SpectralConv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SpectralConv2d(ndf * 8, out_ch, kernel_size=4, stride=2, padding=1))