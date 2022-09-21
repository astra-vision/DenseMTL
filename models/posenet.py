# Adapted from code by Niantic Labs - Monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/pose_decoder.py
#
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn

from vendor.utils import remove_resnet_tail_
from vendor.monodepth2.pose_decoder import PoseDecoder
from vendor.monodepth2.resnet_encoder import ResnetEncoder
from vendor.monodepth2.layers import transformation_from_parameters
from utils.logs import cpprint


class PoseNetwork(nn.Module):
    def __init__(self, pretrain, frame_ids, pose_model_input, provide_uncropped_for_pose, freeze=False):
        super().__init__()

        num_input_frames = len(frame_ids)
        self.num_pose_frames = 2 if pose_model_input == 'pairs' else num_input_frames

        self.encoder = ResnetEncoder(num_layers=18, num_input_images=self.num_pose_frames,
            pretrained=pretrain)
        remove_resnet_tail_(self.encoder)

        self.decoder = PoseDecoder(self.encoder.num_ch_enc, num_input_features=1,
            num_frames_to_predict_for=2)

        self.frame_ids = frame_ids
        self.provide_uncropped_for_pose = provide_uncropped_for_pose

        if freeze:
            print('Freezing posenet!')
            self.requires_grad_(False)

    def forward(self, x):
        out = {}
        img_type = 'color_full_aug' if self.provide_uncropped_for_pose else 'color_aug'

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select which maps the pose network takes as input
            pose_feats = {f_i: x[img_type, f_i, 0] for f_i in self.frame_ids}

            for f_i in self.frame_ids[1:]:
                if f_i != 's':
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.encoder(torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.decoder(pose_inputs)
                    # out['axisangle', 0, f_i] = axisangle
                    # out['translation', 0, f_i] = translation

                    # Invert the matrix if the frame id is negative
                    out['cam_T_cam', 0, f_i] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs = torch.cat(
                [x[img_type, i, 0] for i in self.frame_ids if i != 's'], 1)

            pose_inputs = [self.encoder(pose_inputs)]
            axisangle, translation = self.decoder(pose_inputs)

            for i, f_i in enumerate(self.frame_ids[1:]):
                if f_i != 's':
                    # out['axisangle', 0, f_i] = axisangle
                    # out['translation', 0, f_i] = translation
                    out['cam_T_cam', 0, f_i] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return out