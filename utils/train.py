import contextlib, random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector as p2v


def current_val_interval(cfg, step):
    v_intervals = [(int(k), int(v)) for k, v in cfg["training"]["val_interval"].items()]
    for k, v in sorted(v_intervals, reverse=True):
        if step > k:
            return v

def resize_input(input, target, align_corners=True, mode='bilinear'):
    """Resize input to match last two dimensions of target (generally, spatial dimensions h, w)."""
    h, w = input.shape[-2:]
    ht, wt = target.shape[-2:]

    if h != ht and w != wt:
        return F.interpolate(input, size=(ht, wt), mode=mode, align_corners=align_corners)
    return input

def count(module):
    """Count total number of parameters in nn.Module."""
    return p2v(module.parameters()).numel()

def setup_seeds(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@contextlib.contextmanager
def np_local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_trainer(cfg):
    if 'da' in cfg.training:
        from training.da_train import Trainer
    else:
        from training.train import Trainer
    return Trainer(cfg)