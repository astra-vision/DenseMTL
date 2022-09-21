import torch
import torch.nn.functional as F

import metrics
from utils.logs import cpprint
from utils.train import resize_input
from loss.setup_loss import SetupLoss
from models.normals import cosine_loss, unproject, d2normals, remove_borders, dilate_mask


MIN_DEPTH, MAX_DEPTH = 1e-3, 80
def depth2disp(d):
    return 65536 / (d+1)

def disp2meters(d):
    return 655.36 / d - 1


class NormalsLoss(SetupLoss):
    """
        Multi-scale semantic segmentation loss. This is the base class for semantic segmentation
        supervision. To use a different function extend it and implement a custom `loss_fn`
    """
    def __init__(self, scales, weight=1):
        meters = {f'normals_{s}': metrics.Normals() for s in scales}
        super().__init__(scales, weight, **meters)

        self.maps = meters.keys()
        self.eval_meter = meters[f'normals_{scales[-1]}']
        cpprint(f'NormalsLoss init'
                f'\n MAKE SURE DEPTH GT IS INVERSE NORMALIZED'
                f'\n Scales {list((s, w.item()) for s, w in zip(self.scales, self.weight))}'
                f'\n Evaluating normals on scale {scales[-1]}', c='yellow')

    def main_metric(self):
        return 'mErr', self.eval_meter.mean, 1

    def loss_fn(self, pred, target, mask):
        # normalize option? for supervision
        return cosine_loss(pred, target)[mask].mean()

    def evaluate(self, pred, target, mask, scale):
        self.metrics[f'normals_{scale}'].measure(pred, target, mask)

    def view_d(self, depth_m, mask, viz):
        clipped_depth = depth_m.clamp(MIN_DEPTH, MAX_DEPTH)*100 # clamp meters and convert to cm
        disp_cm = depth2disp(clipped_depth)                     # convert to 1/cm
        cm = viz.depth(disp_cm, mask)                           # convert to colormap
        return cm

    def get_mask(self, mask, depth, normals, train):
        if not train:
            mask *= (MIN_DEPTH < depth)*(depth < MAX_DEPTH)

        # get initial mask from invalid values in depth map
        dmask = dilate_mask(remove_borders(mask.float()), ksize=3)

        # get invalid values from normals computation
        nmask = torch.isnan(normals).logical_not().all(1, True)

        # combine both masks
        normals_mask = nmask*dmask

        return dmask, normals_mask

    def forward(self, x, y, viz, train=True):
        # get GT depth map and retrieve the normals from it
        depth_gt, mask_gt = x['gt_depth'], x['depth_mask']
        gt_m = disp2meters(depth_gt)

        pointcloud = unproject(depth_gt, x['inv_K', 0])
        normals_gt = d2normals(pointcloud, n_bases=4)

        depth_mask, normals_mask = self.get_mask(mask_gt, gt_m, normals_gt, train)

        maps = dict(normals_gt=viz.normals(normals_gt, normals_mask))
        loss = self._init_losses('total')

        for s in self.scales:
            pred = resize_input(y[f'normals', s], normals_gt)

            loss[f'normals_{s}'] = self.loss_fn(pred, normals_gt, normals_mask)
            maps[f'normals_{s}'] = viz.normals(pred, normals_mask, normalize=True)

            if not train:
                self.evaluate(pred, normals_gt, normals_mask, s)

        loss.total = self.mean(loss[f'normals_{s}'] for s in self.scales)

        return loss, maps