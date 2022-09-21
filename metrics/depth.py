import math

import torch

from .average_meters import AverageMetrics


MIN_DEPTH, MAX_DEPTH = 1e-3, 80

class Depth(AverageMetrics):
    def __init__(self, median_scaling=True):
        super().__init__(['Abs. Rel error', 'Sqr. Rel error', 'RMSE', 'Acc. < 1.25', \
            'Acc. < 1.25**2', 'Acc. < 1.25**3'])
        self.median_scaling = median_scaling

    def _metrics(self, pred, gt, mask):
        # Median scaling see Monodepth-2:
        # github.com/nianticlabs/monodepth2/blob/ab2a1bf7d45ae53b1ae5a858f4451199f9c213b3/trainer.py#L517
        if self.median_scaling:
            # compute per-image median scale factor between masked prediction and masked gt
            pred = torch.cat([p[m] * d[m].median() / p[m].median()
                for d, p, m in zip(gt, pred, mask)])
        else:
            pred = pred[mask]
        pred.clamp_(MIN_DEPTH, MAX_DEPTH)

        gt = gt[mask] # GT is masked as is

        ## gt and pred are flattened
        mse = (pred - gt).pow(2).mean()
        abs_rel = ((pred - gt).abs() / gt).mean()
        sqr_rel = ((pred - gt).pow(2) / gt).mean()

        acc = torch.max((gt / pred), (pred / gt))
        acc1 = (acc < 1.25   ).float().mean()
        acc2 = (acc < 1.25**2).float().mean()
        acc3 = (acc < 1.25**3).float().mean()

        return abs_rel, sqr_rel, mse, acc1, acc2, acc3

    def get_scores(self):
        d = super().get_scores()

        # Wait to accumulate mse from all validation batches to compute the sqrt:
        d['RMSE'] = math.sqrt(d['RMSE'])
        self.rmse = d['RMSE']

        return d