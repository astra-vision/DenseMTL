import torch
import torch.nn as nn
from easydict import EasyDict

from metrics import MTLRelativePerf

from .setup_loss import SetupLoss


class MultiLoss(SetupLoss):
    """Combine multiple SetupLoss instances into a single build."""
    def __init__(self, tasks, baselines=None, gammas=None):
        metrics = {k: v for m in tasks.values() for k, v in m.metrics.items()}
        super().__init__([], [], **metrics)

        # mtl metric is handled outside of SetupLoss.metrics
        self.tasks = tasks
        if len(tasks) > 1:
            self.mtl_metric = MTLRelativePerf(baselines, gammas)


    def forward(self, x, y, viz, train=True):
        losses, maps, totals = self._init_losses('total'), {}, []
        for name, loss_fn in self.tasks.items():
            loss, m = loss_fn(x, y, viz, train)
            totals.append(loss.pop('total'))
            losses.update(**loss, **{name: totals[-1]})
            maps.update(m)
        losses.total = sum(totals)
        return losses, maps

    def main_metric(self):
        if len(self.tasks) == 1:
            return next(iter(self.tasks.values())).main_metric()

        # If multi-tasking use relative performance to baseline as main metric
        metric_values = {t: self.tasks[t].main_metric()[1] for t in self.tasks}
        rel_perf = self.mtl_metric.get_scores(metric_values)
        return 'relPerf', rel_perf, 0