# Adapted from code by Hoyer et al.
# https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth/blob/master/models/depth_decoder.py
#
# Original version by Niantic Labs - Monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py
#
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from vendor.three_ways.model_parts import ASPP
from .layers import ConvBlock, Conv3x3, upsample
from utils.logs import cpprint


class DecoderHead(nn.Module):
    def __init__(self, num_ch_enc=None, num_out_ch=None, active_scales=None,
                 use_skips=True, n_upconv=4, num_ch_dec=[16, 32, 64, 128, 256],
                 is_regression=False, **kwargs):
        super(DecoderHead, self).__init__()
        assert num_ch_enc is not None and num_out_ch is not None

        self.num_out_ch = num_out_ch
        self.use_skips = use_skips
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array(num_ch_dec)
        self.n_upconv = n_upconv
        self.active_scales = range(n_upconv) if active_scales is None else active_scales
        self.is_regression = is_regression

        self._init_model(**kwargs)

        cpprint(f'DecoderHead init: '
                f'\n is_regression (i.e. sigmoid applied on output level)={is_regression}'
                f'\n convs keys={self.convs.keys()}'
                f'\n active scales {self.active_scales}', c='cyan')

    def _init_model(self, aspp_pooling=True, intermediate_aspp=False, aspp_rates=[6, 12, 18],
                    batch_norm=False, dropout=0.0, n_project_skip_ch=-1):
        self.convs = OrderedDict()
        for i in range(self.n_upconv, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.n_upconv else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            if i == self.n_upconv and intermediate_aspp:
                self.convs['upconv', i, 0] = ASPP(num_ch_in, aspp_rates, aspp_pooling, num_ch_out)
            else:
                self.convs['upconv', i, 0] = ConvBlock(num_ch_in, num_ch_out, bn=batch_norm, dropout=dropout)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                if n_project_skip_ch == -1:
                    num_ch_in += self.num_ch_enc[i - 1]
                    self.convs['skip_proj', i] = nn.Identity()
                else:
                    num_ch_in += n_project_skip_ch
                    self.convs['skip_proj', i] = nn.Sequential(
                        nn.Conv2d(self.num_ch_enc[i - 1], n_project_skip_ch, kernel_size=1),
                        nn.BatchNorm2d(n_project_skip_ch),
                        nn.ReLU(inplace=True))

            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 1] = ConvBlock(num_ch_in, num_ch_out, bn=batch_norm, dropout=dropout)

        for i in self.active_scales:
            if self.is_regression:
                self.convs['side', i] = Conv3x3(self.num_ch_dec[i], self.num_out_ch)
            else:
                self.convs['side', i] = nn.Conv2d(self.num_ch_dec[i], self.num_out_ch, 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, enc_outs, x=None, exec_layer=None):
        out = {}

        if x is None:
            x = enc_outs[-1]
        if exec_layer is None:
            exec_layer = 'all'

        # cpprint(f'>calling decoder section with exec_layer={exec_layer}')
        for i in range(self.n_upconv, -1, -1):
            if exec_layer != 'all' and i not in exec_layer:
                # cpprint(f'>skip {i}')
                continue

            x = self.convs['upconv', i, 0](x)

            if x.shape[-1] < enc_outs[i - 1].shape[-1] or i == 0:
                x = [upsample(x)]
            else:
                x = [x]

            if self.use_skips and i > 0:
                projected_features = self.convs['skip_proj', i](enc_outs[i - 1])
                x += [projected_features]

            x = torch.cat(x, 1)

            x = self.convs['upconv', i, 1](x)
            out['upconv', i] = x

            if i in self.active_scales:
                # cpprint(f' outputting at scale={i}', c='red')
                if self.is_regression:
                    out[i] = torch.sigmoid(self.convs['side', i](x))
                else:
                    out[i] = self.convs['side', i](x)

        return out