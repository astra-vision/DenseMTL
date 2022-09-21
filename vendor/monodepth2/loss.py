# Adapted from Monodepth2 - Niantic Labs
# https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
#
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import disp_to_depth, get_smooth_loss, SSIM, BackprojectDepth, Project3D


class MonodepthLoss:
    def __init__(self, scales, batch_size, frame_ids, height, width,
                 min_depth, max_depth, test_min_depth, test_max_depth,
                 disparity_smoothness, no_ssim, avg_reprojection, disable_automasking,
                 crop_h=None, crop_w=None):
        self.scales = scales
        self.height = height if crop_h is None else crop_h
        self.width = width if crop_w is None else crop_w
        self.frame_ids = frame_ids
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.test_min_depth = test_min_depth
        self.test_max_depth = test_max_depth
        self.disparity_smoothness = disparity_smoothness
        self.no_ssim = no_ssim
        self.avg_reprojection = avg_reprojection
        self.disable_automasking = disable_automasking
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.inter = nn.Upsample(size=(self.height, self.width), mode='bilinear', align_corners=False)

        if not self.no_ssim:
            self.ssim = SSIM().to(self.device)

        self.backproject = BackprojectDepth(batch_size, self.height, self.width).to(self.device)
        self.project_3d = Project3D(batch_size, self.height, self.width).to(self.device)


    def generate_depth_test_pred(self, outputs):
        res = {}

        for scale in self.scales:
            key = ('depth', 0, scale)
            pred = outputs['disp', scale]
            res[key] = disp_to_depth(self.inter(pred), self.test_min_depth, self.test_max_depth)
        return res


    def generate_images_pred(self, inputs, outputs): # should be rather (self, depth, rgb, inv_K, K)
        """
            Generate the warped (reprojected) color images for a minibatch.
            Returns a dict with the multiscale depth maps and warps, and the the point cloud.
        """
        res = {}

        for scale in self.scales:
            key = ('depth', 0, scale)
            if key in outputs:
                res[key] = outputs[key]
            else:
                pred = outputs['disp', scale]
                res[key] = disp_to_depth(self.inter(pred), self.min_depth, self.max_depth)

            cam_points = self.backproject(depth=res[key], inv_K=inputs['inv_K', 0])
            if scale == 0:
                res['cam_points'] = cam_points[:, :3].reshape(-1, 3, self.height, self.width)

            for frame_id in self.frame_ids[1:]:
                T = inputs['stereo_T'] if frame_id == 's' else outputs['cam_T_cam', 0, frame_id]

                pix_coords = self.project_3d(cam_points, inputs['K', 0], T)
                res['color', frame_id, scale] = F.grid_sample(
                    input=inputs['color', frame_id, 0],
                    grid=pix_coords,
                    padding_mode='border', align_corners=True)

        return res


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images."""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch."""
        res = dict(mono_loss=0)

        for scale in self.scales:
            loss = 0

            disp = outputs['disp', scale]
            color = inputs['color', 0, scale]
            target = inputs['color', 0, 0]

            ## Reprojection
            reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = outputs['color', frame_id, scale]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)
            if self.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            ## Identity reprojection
            if not self.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.frame_ids[1:]:
                    pred = inputs['color', frame_id, 0]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            if not self.disable_automasking:
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device)*1e-5 # break ties
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.size(1) == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.disable_automasking and scale == 0:
                res['automask'] = (idxs > reprojection_loss.size(1) - 1).unsqueeze(1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            res['mono_loss', scale] = loss
            res['mono_loss'] += loss

        res['mono_loss'] /= len(self.scales)
        return res