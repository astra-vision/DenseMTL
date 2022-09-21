import math

import torch
import torch.nn.functional as F

from .average_meters import AverageMetrics


class Normals(AverageMetrics):
    def __init__(self):
        super().__init__(['Mean', 'Median', 'RMSE', 'error < 11.25°', \
            'error < 22.5°', 'error < 30°'])

    def _metrics(self, pred, gt, mask):
        error = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))

        mse = error.pow(2).mean()
        thr1 = (error < 11.25).float().mean()
        thr2 = (error < 22.5 ).float().mean()
        thr3 = (error < 30   ).float().mean()

        return error.mean(), error.median(), mse, thr1, thr2, thr3

    def get_scores(self):
        d = super().get_scores()
        self.mean = d['Mean']
        d['RMSE'] = math.sqrt(d['RMSE'])
        return d