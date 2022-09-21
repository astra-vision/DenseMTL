import numpy as np
import torch


class AverageMetrics(object):
    def __init__(self, metrics):
        self.names = metrics
        self.reset()

    def reset(self):
        # keep last row for num of updates for averaging
        self.metrics = np.zeros(len(self.names) + 1)

    def _metrics(self):
        raise NotImplementedError

    def measure(self, *batch, **kwargs):
        for arguments in zip(*[x.detach() for x in batch]):
            self.metrics += tuple(m.item() for m in self._metrics(*arguments, **kwargs)) + (1,)

    def get_scores(self):
        metrics = (self.metrics / self.metrics[-1])[:-1] # averaged by total count
        return dict(zip(self.names, metrics))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.detach()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = torch.true_divide(self.sum, self.count)


class AverageMeterDict(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avgs = {}
        self.sums = {}
        self.counts = {}

    def update(self, vals):
        for k, v in vals.items():
            if torch.is_tensor(v):
                v = v.detach().cpu()
            if k not in self.sums:
                self.sums[k] = 0
                self.counts[k] = 0
            self.sums[k] += v
            self.counts[k] += 1
            self.avgs[k] = torch.true_divide(self.sums[k], self.counts[k])