import torch.nn.functional as F

import metrics
from utils.train import resize_input
from loss.setup_loss import SetupLoss
from utils.logs import cpprint

class SemsegLoss(SetupLoss):
    """
        Multi-scale semantic segmentation loss. This is the base class for semantic segmentation
        supervision. To use a different function extend it and implement a custom `loss_fn`
    """
    def __init__(self, scales, n_classes, weight=1):
        meters = {f'semseg_{s}': metrics.Semseg(n_classes) for s in scales}
        super().__init__(scales, weight, **meters)

        self.maps = meters.keys()
        self.eval_meter = meters[f'semseg_{scales[-1]}']
        cpprint(f'SemsegLoss init'
                f'\n Scales {list((s, w.item()) for s, w in zip(self.scales, self.weight))}'
                f'\n Evaluating semseg on scale {scales[-1]}', c='yellow')

    def main_metric(self):
        return 'mIoU', self.eval_meter.mIoU, 0

    def loss_fn(self, pred, target):
        return F.cross_entropy(pred, target, ignore_index=250)

    def evaluate(self, pred, target, scale):
        self.metrics[f'semseg_{scale}'].measure(label_trues=target, label_preds=pred)

    def forward(self, x, y, viz, train=True):
        loss = self._init_losses('total')
        y_true = x['lbl']
        maps = dict(rgb_gt=x['color_aug', 0, 0],
                    semseg_gt=viz.seg(y_true))

        for s in self.scales:
            y_pred = y[f'semseg', s]
            upsampled = resize_input(y_pred, y_true, align_corners=True)
            maps[f'semseg_{s}'] = viz.seg(upsampled.argmax(1))
            loss[f'semseg_{s}'] = self.loss_fn(upsampled, y_true)

            if not train:
                self.evaluate(upsampled.argmax(1), y_true, s)

        loss.total = self.mean(loss[k] for k in self.maps)
        return loss, maps