from statistics import mean

import wandb
import torch
import numpy as np
import torch.nn.functional as F

import metrics
from utils.train import resize_input
from utils.logs import cpprint
from loss.setup_loss import SetupLoss
from vendor.three_ways.loss import berhu


MIN_DEPTH, MAX_DEPTH = 1e-3, 80 # in meters
def depth2disp(d):
    return 65536 / (d+1)
def disp2meters(d):
    return 655.36 / d - 1


class BerHu(SetupLoss):
    ## Using normalized inverse depth values

    def __init__(self, scales, weight=1, median_scaling=True, **kwargs):
        meters = {f'depth_{s}': metrics.Depth(median_scaling=median_scaling) for s in scales}
        super().__init__(scales, weight, **meters)

        self.maps = meters.keys()
        self.median_scaling = median_scaling
        self.eval_meter = meters[f'depth_{scales[-1]}']

        cpprint(f'berHu init'
                f'\n Scales {list((s, w.item()) for s, w in zip(self.scales, self.weight))}'
                f'\n Evaluating with{"" if self.median_scaling else "out"} median scaling'
                f'\n Evaluating depth on scale {scales[-1]}', c='yellow')

    def main_metric(self):
        return 'RMSE', self.eval_meter.rmse, 1

    def evaluate(self, pred, gt, mask, scale):
        self.metrics[f'depth_{scale}'].measure(pred, gt, mask)

    def loss_fn(self, pred, gt, mask):
        return berhu(pred, gt, mask)

    def view(self, inv_depth, mask, viz, p):                       # `inv_depth` is inv-normalize d
        depth = disp2meters(inv_depth)                          # depth in meters
        clipped_depth = depth.clamp(MIN_DEPTH, MAX_DEPTH)*100   # clamp meters and convert to cm
        disp_cm = depth2disp(clipped_depth)                     # convert to 1/cm
        cm = viz.depth(disp_cm, mask, p)                           # convert to colormap
        return cm

    def forward(self, x, y, viz, train=True):
        loss = self._init_losses('total')
        gt, mask_gt = x['gt_depth'], x['depth_mask']
        maps = dict()#rgb=x['color', 0, 0])

        gt_m = disp2meters(gt)
        if not train:
            mask_gt *= (MIN_DEPTH < gt_m)*(gt_m < MAX_DEPTH)
        maps['depth_gt'] = self.view(gt, mask_gt, viz, p=98)

        for s in self.scales:
            out = resize_input(y['depth', s], gt)
            loss[f'depth_{s}'] = self.loss_fn(out, gt, mask_gt)
            maps[f'depth_{s}'] = self.view(out, mask_gt, viz, p=98)

            pred_m = disp2meters(out)
            # maps[f'err_{s}'] = viz.depth((pred_m.clamp_(MIN_DEPTH, MAX_DEPTH)-gt_m).abs(), mask_gt)

            ## Validation
            if not train:
                self.evaluate(pred_m, gt_m, mask_gt, s)

                if self.median_scaling: # median scaling for viz
                    ## Compute error map with median scale + clamped prediction
                    ## medians are determined on masked depth maps ofc
                    scales = [g[m].median()/p[m].median() for g, p, m in zip(gt, pred_m, mask_gt)]
                    mscale = torch.stack(scales)[:,None,None,None]
                    loss[f'median_{s}'] = mscale.mean() # log the median scale on validation batches

                    viz_pred = mscale*pred_m                                    # per-image median scaling
                    clipped_depth = viz_pred.clamp_(MIN_DEPTH, MAX_DEPTH)*100   # clamp and convert to cm
                    disp_cm = depth2disp(clipped_depth)                         # convert to 1/depth

                    maps[f'depth_{s}'] = viz.depth(disp_cm, mask_gt)

        loss.total = self.mean(loss[k] for k in self.maps)
        return loss, maps
