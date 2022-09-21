import torch
import torch.nn.functional as F

import metrics
from loss.setup_loss import SetupLoss
from utils.logs import cpprint
from utils.train import resize_input


class EdgesLoss(SetupLoss):
    """
        Multi-scale semantic segmentation loss. This is the base class for semantic segmentation
        supervision. To use a different function extend it and implement a custom `loss_fn`
    """
    def __init__(self, scales, weight=1, pos_weight=.95):
        meters = {f'edges_{s}': metrics.Edges() for s in scales}
        super().__init__(scales, weight, **meters)

        self.maps = meters.keys()
        self.eval_meter = meters[f'edges_{scales[-1]}']

        self.class_weight = torch.tensor([1. - pos_weight, pos_weight]).cuda()

        cpprint(f'EdgesLoss init'
                f'\n Scales {list((s, w.item()) for s, w in zip(self.scales, self.weight))}'
                f'\n Evaluating edges on scale {scales[-1]}', c='yellow')

    def main_metric(self):
        return 'F1', self.eval_meter.f1, 0

    def loss_fn(self, pred, target):
        return F.cross_entropy(pred, target, weight=self.class_weight)

    def evaluate(self, pred, target, scale):
        self.metrics[f'edges_{scale}'].measure(label_trues=target, label_preds=pred)

    def forward(self, x, y, viz, train=True):
        loss = self._init_losses('total')
        y_true = x['gt_edges']
        maps = dict(edges_gt=viz.edges(y_true))

        for s in self.scales:
            y_pred = y[f'edges', s]
            upsampled = resize_input(y_pred, y_true, align_corners=True)
            maps[f'edges_{s}'] = viz.edges(upsampled.argmax(1))
            loss[f'edges_{s}'] = self.loss_fn(upsampled, y_true)

            if not train:
                self.evaluate(upsampled.argmax(1), y_true, s)

        loss.total = self.mean(loss[k] for k in self.maps)
        return loss, maps