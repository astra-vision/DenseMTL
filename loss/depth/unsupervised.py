import torch

from vendor.monodepth2 import PhotometricLoss
from loss.setup_loss import SetupLoss
import metrics
from utils.logs import cpprint


def inv2meters(d):
    """Convert inverse normalized depth values to depth in meters."""
    return 655.36 / d - 1

def meters2inv(d):
    """Convert depth meter values to inverse normalized depth."""
    return 655.36 / (d+1)

MIN_DEPTH, MAX_DEPTH = 1e-3, 80

class UnsupervisedDepthLoss(SetupLoss):
    """Loss function for the Monodepth setup."""
    def __init__(self, scales, batch_size, crop_h=None, crop_w=None, weight=1, **kwargs):
        bounds = (kwargs['test_min_depth'], kwargs['test_max_depth'])
        super().__init__(scales, weight, depth=metrics.Depth(median_scaling=True))

        self.bounds = bounds
        self.loss = PhotometricLoss(scales, batch_size, crop_h=crop_h, crop_w=crop_w, **kwargs)
        self.val_loss = PhotometricLoss(scales, batch_size, **kwargs)

        cpprint(f'SSDE init'
                f'\n Scales: {self.scales} with weight={weight}'
                f'\n Evaluating depth on scale s=0', c='yellow')

    def main_metric(self):
        return 'RMSE', self.metrics.depth.rmse, 1

    def evaluate(self, pred, gt, mask):
        self.metrics.depth.measure(pred, gt, mask)

    def scale(self, pred, gt, mask):
        scales = [g[m].median()/p[m].median() for p, g, m in zip(pred, gt, mask)]
        scales = torch.stack(scales).view(-1, 1, 1, 1)
        return scales

    def forward(self, x, y, viz, train=True):
        maps = {}
        loss = self._init_losses('monodepth', 'feature_dist')
        fn = self.loss if train else self.val_loss

        y.update(fn.generate_images_pred(x, y))
        y.update(fn.compute_losses(x, y))

        gt, mask_gt = x['gt_depth'], x['depth_mask']
        gt_m = inv2meters(gt)
        gt = meters2inv(gt_m.clamp(MIN_DEPTH, MAX_DEPTH))

        for s in self.scales:
            loss[f'mono_{s}'] = y['mono_loss', s]

            pred_m = y['depth', 0, s]

            # Visualization with median scaling applied
            scale = self.scale(pred_m, gt_m, mask_gt)
            loss[f'median{s}'] = scale.mean()

            pred_scaled = meters2inv((scale*pred_m).clamp(MIN_DEPTH, MAX_DEPTH))
            maps[f'pred_scaled_{s}'] = viz.depth(pred_scaled, mask_gt)
            # pred_unscaled = meters2inv(pred_m.clamp(MIN_DEPTH, MAX_DEPTH))
            # maps[f'pred_unscaled{s}'] = viz.depth(pred_unscaled, mask_gt)

        # if 'imnet_features' in y:
        #     loss.feature_dist = torch.dist(y['encoder_features'], y['imnet_features'])
        # loss.monodepth = y['mono_loss']
        # loss.total = loss.monodepth + loss.feature_dist
        # loss.total = y['mono_loss']
        loss.total = self.mean(loss[f'mono_{s}'] for s in self.scales)

        if not train:
            pred_m = y['depth', 0, 0]

            # Add out of bound points in mask
            mask_gt *= (MIN_DEPTH < gt_m)*(gt_m < MAX_DEPTH)

            ## Evaluation in meters
            self.evaluate(pred_m, gt_m, mask_gt)

        maps.update(rgb_0=x['color',  0, 0],
                    mask_gt=mask_gt.float(),
                    # rgb_p=x['color', -1, 0],
                    # rgb_n=x['color',  1, 0],
                    automask=y['automask'].float(),
                    # repr_p=y['color', -1, 0],
                    # repr_n=y['color',  1, 0],
                    # pred_scaled=viz.depth(pred_scaled, mask_gt),
                    # pred_unscaled=viz.depth(pred_unscaled, mask_gt),
                    gt=viz.depth(gt, mask_gt))

        return loss, maps