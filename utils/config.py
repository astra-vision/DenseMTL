
import functools, os

import yaml
from easydict import EasyDict

from .utils import get_checkpoint, now

def construct_cfg(args, base, setup_cfg, env_vars):
    """Combines the two configuration files with the CLI options."""
    cfg = EasyDict({**base, **setup_cfg}) # 1st level merge

    ## Fill missing config keys
    config_name = args.config.rsplit('/', 1)[1].split('.')[0]
    if 'da' in cfg.training:
        ds_setup_name = f'{cfg.data.source.dataset}2{cfg.data.target.dataset}'
    else:
        ds_setup_name = cfg.data.dataset

    root = cfg.training.logdir
    base_dir = os.path.join(root, ds_setup_name)
    eval_dir = '' if args.evaluate is None else 'eval'

    cfg.data.dataset_seed = int(args.seed)
    cfg.training.seed = int(args.seed)

    cfg.update(DEBUG=args.debug,
               PROJECT=args.project,
               WANDB=args.disable_wandb,
               SWEEP=False,
               NAME=f's{args.seed}_{ds_setup_name}_{config_name}',
               base_dir=os.path.join(base_dir, config_name, f's{args.seed}'),
               logdir=os.path.join(base_dir, eval_dir, config_name, f's{args.seed}', now()))

    ## Resolve environment variables
    resolve_cfg_vars_(cfg, env_vars)

    if args.resume:
        cfg.training.resume = get_checkpoint(cfg.logdir)

    return cfg

def resolve_cfg_vars_(cfg, env_vars):
    for k, v in cfg.items():
        if isinstance(v, dict):
            resolve_cfg_vars_(v, env_vars)
        elif isinstance(v, str):
            cfg[k] = os.path.expandvars(v)
            if '$ENV:' in v:
                var, *_ = v.replace('$ENV:', '').split(os.sep)
                cfg[k] = v.replace('$ENV:'+var, env_vars[var])

def read_yaml(fp):
    assert os.path.isfile(fp)
    with open(fp) as f:
        return yaml.safe_load(f)

def write_yaml(fp, cfg):
    os.makedirs(fp, exist_ok=True)
    with open(fp + '/cfg.yml', 'w') as f:
        yaml.dump(cfg, f)

def load_env(fp):
    assert os.path.isfile(fp)
    with open(fp) as f:
        env_vars = yaml.safe_load(f)
        env_vars = {k: os.path.expandvars(v) for k, v in env_vars.items()}
        os.environ.update(env_vars)
    return env_vars