
from utils.logs import cpprint
from .utils import assemble_param_groups

def get(cfg, model):
    lr = cfg.setup.lr
    default = lr.pop('default')

    if 'baseline' in cfg.setup.name:
        submodules = ['backbone'] + [f'heads.{t}' for t in cfg.setup.model.tasks]
    elif cfg.setup.name in ['padnet', 'padnet+'] or '3ways' in cfg.setup.name:
        submodules = ['backbone'] + [f'params.{t}' for t in cfg.setup.model.tasks]
    elif 'ours-cxatt' in cfg.setup.name:
        submodules = ['backbone'] + [f'params.{t}' for t in cfg.setup.model.tasks]

    # plugin monodepth requires wrapping the model group keys with 'main.' prefix
    if 'monodepth' in cfg.setup.name:
        submodules = [f'main.{m}' for m in submodules] + ['monodepth.pose_net']

    if not submodules:
        raise NotImplementedError

    # cpprint(cfg.setup.name, submodules, c='red')
    groups = assemble_param_groups(model, submodules, default, lr)
    cpprint(f'Parameter groups:', groups, c='green')

    return groups
