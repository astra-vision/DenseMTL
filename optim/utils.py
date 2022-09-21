from itertools import chain, tee
from functools import reduce

import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import parameters_to_vector as p2v
# from torch.nn.modules.module import ModuleAttributeError

from utils.logs import cpprint


def optimizer(name, param_groups, optim_kwargs):
    cpprint(f'Calling {name} with arguments:', optim_kwargs, c='green')
    if name == 'SGD':
        return SGD(param_groups, **optim_kwargs)
    elif name == 'Adam':
        return Adam(param_groups, **optim_kwargs)
    elif name == 'AdamW':
        return AdamW(param_groups, **optim_kwargs)

    raise NotImplementedError

def scheduler(name, optimizer, scheduler_kwargs):
    if name == 'step':
        return StepLR(optimizer, **scheduler_kwargs)
    raise NotImplementedError


def assemble_param_groups(model, submodules, default, groups):
    """groups: dict(k=lr)"""
    param_groups = []
    for key, lr in groups.items():
        try:
            module = get_submodule(model, key) # use nn.Module.get_submodule(key)
        except:
            raise AttributeError(f'No submodule found with key: {key}')

        if key not in submodules:
            raise KeyError(f'Error in param_group submodules def, missing key: {key} '
                            'found in config and nn.Module')
        submodules.remove(key)
        param_groups.append(_verbose_group(key, _get_params(module), lr))

    if submodules:
        cpprint(f'submodules {submodules} assigned to default group', c='magenta')
        params = chain(*map(lambda k: _get_params(model, k), submodules))
        param_groups.append(_verbose_group('default', params, default))
    return param_groups


def _get_params(model, name=None):
    submodule = model if name is None else get_submodule(model, name)
    if isinstance(submodule, nn.Parameter):
        return iter([submodule])
    return submodule.parameters()

def _verbose_group(name, param_itr, lr):
    itr1, itr2 = tee(param_itr)
    return dict(name=name, params=itr1, numel=f'{p2v(itr2).numel():,}', lr=lr, initial_lr=lr)

def get_submodule(object, keys):
    """Recursive getattr() on module using dot-separated key"""
    return reduce(lambda o, k: getattr(o, k), keys.split('.'), object)