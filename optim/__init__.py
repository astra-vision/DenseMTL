from torch.optim import Optimizer

from . import param_groups
from optim.utils import optimizer, scheduler


def get_optimizer(cfg, model) -> Optimizer:
    pgroups = param_groups.get(cfg, model)
    return optimizer(cfg.optim.name, pgroups, cfg.optim.get('kwargs', {}))


def get_scheduler(cfg, optimizer):
    return scheduler(cfg.scheduler.name, optimizer, cfg.scheduler.get('kwargs', {}))