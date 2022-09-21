from typing import List, Tuple

import torch
import torch.nn as nn
from easydict import EasyDict
from utils.visualize import Visualize

class SetupLoss(nn.Module):
    """Base class for multi-scale weighted loss."""
    def __init__(self, scales, weight, **metrics):
        super().__init__()
        self.metrics = EasyDict(metrics)

        self.scales = scales
        w_ = (weight,)*len(scales) if type(weight) is float or type(weight) is int else weight
        assert len(self.scales) == len(w_)
        self.weight = torch.tensor(w_, dtype=float)

    def main_metric(self) -> Tuple[str, int, int]:
        """Returns name of main metric, value, and 0 (higher = better) or 1 (lower = better)"""
        raise NotImplementedError

    def mean(self, losses) -> torch.Tensor:
        L = torch.stack(list(losses))
        w = self.weight.type(L.dtype).to(L.device)
        return (w*L).mean()

    def forward(self, x: dict, y: dict, viz: Visualize, train=True) -> dict:
        raise NotImplementedError

    def _init_losses(self, *names) -> EasyDict:
        d = dict(zip(names, torch.zeros(len(names))))
        return EasyDict(d)

    def _init_maps(self, batch_size: int, outputs: List[str]) -> EasyDict:
        d = dict(zip(outputs, ([None]*batch_size,)*len(outputs)))
        return EasyDict(d)