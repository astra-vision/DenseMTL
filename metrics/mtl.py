import math

import torch
import torch.nn.functional as F

from .average_meters import AverageMetrics


class MTLRelativePerf():
    def __init__(self, baselines, gammas):
        """Gamma values are set as: 0 if higher is better, and 1 if lower is better."""
        assert len(baselines) == len(gammas)
        self.b = baselines
        self.g = gammas

    def get_scores(self, metrics): ##! could remove gamma from MTLRelativePerf intializer
        b, g, m = self._ordered_zip(self.b, self.g, metrics)
        return ((-1)**g*(m - b)/b).mean()*100

    def _ordered_zip(self, b, g, m):
        # Make sure the ordering in all three dicts (b, g, and m) is the same:
        return torch.tensor(list(zip(*[(b[k], g[k], m[k]) for k in m])))