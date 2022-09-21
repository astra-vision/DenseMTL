import os, argparse, glob
from utils.logs import cpprint

from utils.config import construct_cfg, read_yaml, write_yaml, load_env
from utils.train import get_trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-config', type=str, default='configs/env_config.yml')
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--project', default='DenseMTL')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate', type=str, default=None)
    parser.add_argument('-s', '--seed', default=42)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-w', '--disable-wandb', action='store_false')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ## Load CLI arguments
    args = get_args()

    ## Load environment variables
    env_vars = load_env(args.env_config)

    ## Load base config
    base = read_yaml(args.base) if args.base else {}

    ## Load main config
    main_cfg = read_yaml(args.config)

    ## Assemble the final config object
    cfg = construct_cfg(args, base, main_cfg, env_vars)

    ## Dump config in project directory
    if not args.debug:
        write_yaml(cfg.logdir, cfg)

    trainer = get_trainer(cfg)

    cpprint(f'-> {"Debugging" if args.debug else "Running"}: {cfg.logdir}', c='red')
    if args.evaluate is None:
        cpprint('-> Loaded trainer for training', c='red')
        trainer.train()
    else:
        cpprint('-> Loaded trainer for evaluation', c='red')
        trainer.evaluate(args.evaluate)