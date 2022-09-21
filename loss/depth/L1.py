from statistics import mean

import wandb
import torch
import numpy as np
import torch.nn.functional as F

import metrics
from utils.train import resize_input
from utils.logs import cpprint
from loss.setup_loss import SetupLoss
import matplotlib.pyplot as plt


MIN_DEPTH, MAX_DEPTH = 1e-3, 80 # in meters
def depth2disp(d):
    return 65536 / (d+1) #(d.clamp(min=1e-3, max=80) + 1)

class L1(SetupLoss):
    ## Using depth values, not normalized inverse depth values

    def __init__(self, scales, weight=1, median_scaling=True, **kwargs):
        super().__init__(scales, weight, depth=metrics.Depth(median_scaling=median_scaling))

        self.median_scaling = median_scaling
        self.maps = [f'depth_{s}' for s in self.scales]
        cpprint(f'Evaluating depth on scales={set(self.maps)}', c='yellow')


    def main_metric(self):
        return 'RMSE', self.metrics.depth.rmse, 1

    def evaluate(self, pred, gt, mask):
        self.metrics.depth.measure(pred, gt, mask)

    def loss_fn(self, pred, gt, mask):
        return F.l1_loss(pred[mask], gt[mask])

    def view(self, depth, mask, viz):
        clipped_depth = depth.clamp(MIN_DEPTH, MAX_DEPTH)*100 # preview is 1/cm
        # clipped_depth = depth.clamp(MIN_DEPTH*100, MAX_DEPTH*100) # !CM preview is 1/cm
        return viz.depth(depth2disp(clipped_depth), mask)

    def rgb_mask(self, x, m, color):
        # x and m are both is a single-channel map: err map and mask respectively
        # color is the three channel pixel color used to visually mask the rgb vis
        rgb_x = x.transpose(0,1).repeat(3,1,1,1)
        rgb_x[:, ~m[:,0]] = color
        rgb_x = rgb_x.transpose(0,1)
        return rgb_x

    def forward(self, x, y, viz, train=True):
        loss = self._init_losses('total')

        # all datasets output depth values in meters
        # This L1 loss supervises the output of depth values: m
        gt, mask_gt = x['gt_depth'], x['depth_mask']

        maps = {}
        upsampled = {}
        for s in self.scales:
            upsampled[s] = torch.relu(resize_input(y['depth', s], gt))
            loss[f'depth_{s}'] = self.loss_fn(upsampled[s], gt, mask_gt)
            maps[f'depth_{s}'] = self.view(upsampled[s], mask_gt, viz)

        # Use of highest scale for 'advanced' visualizations
        pred = upsampled[max(self.scales)]

        RED = torch.tensor([1.,0.,0.], dtype=pred.dtype, device=pred.device)[:,None]

        # look at the depth distribution with a histogram
        # fig = plt.hist(gt[0,0].cpu().numpy(), density=True, bins=50); plt.savefig('src_raw.png'); plt.clf()
        # fig = plt.hist(gt[0,0][mask_gt[0,0]].cpu().numpy(), density=True, bins=50); plt.savefig('tgt_raw.png'); plt.clf()

        if train: ## Training visualization
            err = torch.abs(pred - gt)*mask_gt
            vmax = np.percentile(err[mask_gt].detach().cpu().numpy(), q=75)
            err_q75 = err.clamp(max=vmax) / vmax
            maps['err_q75'] = self.rgb_mask(err_q75, mask_gt, RED)

        if not train: ## Validation visualization
            # visualize discretized depth with semseg color decoder
            # d_entr = x['gt_ent'].argmax(1)
            # maps['gt_ent'] = viz.seg(d_entr)

            # we take the loaded mask but also remove the out-of-range values
            mask_gt *= (MIN_DEPTH < gt)*(gt < MAX_DEPTH)

            if self.median_scaling:
                ## Compute error map with median scale + clamped prediction
                # scales = [gt[i][mask_gt[i]].median()/pred[i][mask_gt[i]].median() for i in range(len(gt))]
                # mscale = torch.stack(scales)[:,None,None,None]
                # pred *= mscale
                pred = torch.stack([g[m].median()/p[m].median()*p
                    for p, g, m in zip(pred, gt, mask_gt)])
                pred.clamp_(MIN_DEPTH, MAX_DEPTH)

            # overwrite with scaled version and proper mask
            maps[f'depth_{s}'] = self.view(pred, mask_gt, viz)

            if False: ## Visualize error map
                err = torch.abs(pred - gt)

                vmax = np.percentile(err[mask_gt].cpu().numpy(), q=75)
                err_q75 = err.clamp(max=vmax) / vmax
                maps['err_q75'] = self.rgb_mask(err_q75, mask_gt, RED)

            # Evaluation is done in meters
            self.evaluate(upsampled[s], gt, mask_gt)

        maps.update(rgb=x['color_aug', 0, 0],
                    gt=self.view(gt, mask_gt, viz))

        loss.total = self.mean(loss[k] for k in self.maps)

        return loss, maps
