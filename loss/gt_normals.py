import torch
import torch.nn.functional as F

import metrics
from loss.setup_loss import SetupLoss
from models.normals import cosine_loss
from utils.logs import cpprint
from utils.train import resize_input


class NormalsGTLoss(SetupLoss):
    """
        Multi-scale semantic segmentation loss. This is the base class for semantic segmentation
        supervision. To use a different function extend it and implement a custom `loss_fn`
    """
    def __init__(self, scales, weight=1, loss='cos'):
        meters = {f'normals_{s}': metrics.Normals() for s in scales}
        super().__init__(scales, weight, **meters)

        self.maps = meters.keys()
        self.eval_meter = meters[f'normals_{scales[-1]}']

        if loss == 'cos':
            self.loss = cosine_loss
        elif loss == 'l1':
            self.loss = F.l1_loss

        cpprint(f'NormalsLoss init'
                f'\n Scales {list((s, w.item()) for s, w in zip(self.scales, self.weight))}'
                f'\n Evaluating normals on scale {scales[-1]}', c='yellow')

    def main_metric(self):
        return 'mErr', self.eval_meter.mean, 1

    def loss_fn(self, pred, target):
        pred = F.normalize(pred)
        target = F.normalize(target)
        return self.loss(pred, target).mean()

    def evaluate(self, pred, target, scale):
        b, _, h, w = pred.shape
        mask = torch.ones(b,1,h,w, device=pred.device, dtype=bool)
        pred = F.normalize(pred)
        self.metrics[f'normals_{scale}'].measure(pred, target, mask)

    def forward(self, x, y, viz, train=True):
        normals_gt = F.normalize(x['gt_normals'])

        maps = dict(normals_gt=viz.gt_normals(normals_gt))
        loss = self._init_losses('total')

        for s in self.scales:
            pred = resize_input(y[f'normals', s], normals_gt)

            loss[f'normals_{s}'] = self.loss_fn(pred, normals_gt)
            maps[f'normals_{s}'] = viz.gt_normals(pred, normalize=True)

            if not train:
                self.evaluate(pred, normals_gt, s)

        loss.total = self.mean(loss[f'normals_{s}'] for s in self.scales)

        return loss, maps