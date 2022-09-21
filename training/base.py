import os, time
from datetime import timedelta

import wandb
import torch
from tqdm import tqdm
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm_

from dataset import build_dataset
from metrics import AverageMeter, AverageMeterDict
from utils.logs import cpprint
from utils.visualize import Visualize
from utils.train import current_val_interval, setup_seeds


class TrainerBase():
    def __init__(self, cfg: EasyDict):
        self.cfg = cfg
        self._set_defaults()

        if self.cfg.WANDB and not self.cfg.SWEEP:
            wandb.init(project=self.cfg.PROJECT,
                       name=self.cfg.NAME,
                       config=self.cfg)

        # Plant a seed for the planet
        setup_seeds(self.cfg.training.seed)

        # Training mode: benchmark and debugging
        torch.backends.cudnn.benchmark = self.cfg.training.benchmark
        torch.autograd.set_detect_anomaly(self.cfg.training.detect_anomaly)

        # AMP and device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Run from child class
        self.setup_dls()
        self.setup_viz()


    def setup_viz(self):
        self.viz_train = Visualize(dir='train',
                                   max=self.cfg.data.n_train_logged,
                                   segmap2color=self.train_ds.encoder.segmap2color,
                                   max_percentile=self.cfg.data.max_percentile)
        self.viz_valid = Visualize(dir='valid',
                                   max=self.cfg.data.n_valid_logged,
                                   segmap2color=self.valid_ds.encoder.segmap2color,
                                   max_percentile=self.cfg.data.max_percentile)

    def _set_defaults(self):
        self.cfg.data.max_percentile = self.cfg.data.get('max_percentile', 100)
        self.use_amp = self.cfg.training.use_amp
        self.logdir = self.cfg.logdir
        self.early_stopping = None

    def log(self, d):
        no_viz = {k: v for k, v in d.items() if type(v).__module__ != 'wandb.sdk.data_types'}
        cpprint(no_viz)
        if self.cfg.WANDB:
            wandb.log({**d, 'step': self.step})

    def _init_losses(self, *names):
        return EasyDict(dict(zip(names, torch.zeros(len(names)))))

    def model_step_(self):
        if self.cfg.optim.clip_grad_norm:
            clip_grad_norm_(self.model.parameters(), self.cfg.optim.clip_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

    def evaluate(self, f):
        self.step = 1
        self.model.load_setup(f)
        self.validate()

    def debug_step(self, avg_meter, timer):
        self.train_logging(avg_meter, timer)
        self.validate(limit=True)

        name, curr, is_better = self.is_better()
        cpprint('[METRIC]', name, curr, is_better, c='red')
        if is_better:
            self.best_metric = curr

    def train(self):
        if self.cfg.training.resume is not None:
            self.load_resume_()
        else:
            self.step, self.best_metric = 0, None

        avg_meter = AverageMeterDict()
        timer = AverageMeter()
        flag = True
        all_time = time.time()

        while self.step <= self.cfg.training.iterations and flag:
            # self.log_corr()

            for inputs in self.train_itr():
                ## Training
                if True:
                    self.step += 1

                    start = time.time()

                    self.train_step(inputs, avg_meter)

                    timer.update(time.time() - start)

                if self.cfg.DEBUG:
                    return self.debug_step(avg_meter, timer)

                if self.needs_printing():
                    self.train_logging(avg_meter, timer)
                    timer.reset()
                    avg_meter.reset()

                ## Validation
                if self.needs_validating():
                    # reset the metrics before evaluation
                    for m in self.loss.metrics.values():
                        m.reset()

                    self.validate()

                    name, curr, is_better = self.is_better()
                    if is_better:
                        self.best_metric = curr
                        if self.cfg.WANDB:
                            wandb.run.summary[name] = curr

                        if self.cfg.training.save_model and not self.cfg.SWEEP:
                            out_fp = self.save_resume()
                            cpprint(f'saved checkpoint in {out_fp}', color='magenta')

                    if self.early_stopping is not None:
                        if not self.early_stopping.step(self.best_metric):
                            flag = False
                            break

                # save model
                if self.step in [29_000, 30_000, 31_000, 32_000, 33_000, 34_000, 35_000, 36_000, 37_000, 38_000, \
                                 39_000, 40_000, 41_000, 42_000, 43_000, 44_000, 45_000, 46_000, 47_000, 48_000, \
                                 49_000, 50_000, 51_000, 52_000, 53_000, 54_000, 55_000, 56_000, 57_000, 58_000, \
                                 59_000, 60_000, 61_000, 62_000, 63_000, 64_000, 65_000, 66_000, 67_000]:
                    cpprint(f'Saving model to {self.logdir}', c='red')
                    self.save_model()

                # Checkpoint
                if self.step in [20_000, 30_000, 35_000, 40_000, 42_000, 45_000, 50_000, \
                    55_000, 60_000, 65_000, 70_000, 75_000, 80_000, 85_000, 90_000, \
                    100_000, 105_000, 110_000, 115_000, 120_000]:
                    out_fp = self.save_resume()
                    cpprint(f'saved checkpoint in {out_fp} step={self.step}', c='magenta')

                if self.needs_end():
                    flag = False
                    all_time = time.time() - all_time
                    break

        if self.logdir and self.cfg.training.save_finished_model:
            name, curr, is_better = self.is_better()
            cpprint(self.logdir, f'Finished training in {str(timedelta(seconds=all_time)):0>8}: {name}={curr}', c='red')
            self.save_model()

    def save_model(self):
        state = self.model.state_dict()
        f = os.path.join(self.logdir, f'model_{self.step}.pkl')
        torch.save(state, f)

    def is_better(self):
        name, curr, lower_is_better = self.loss.main_metric()
        curr_best = self.best_metric or curr
        return name, curr, curr <= curr_best if lower_is_better else curr >= curr_best

    def get_lrs(self):
        return {f'lr/{g["name"]}': g['lr'] for g in self.optimizer.param_groups}

    @torch.no_grad()
    def validate(self, limit=False):
        limit = self.cfg.data.n_valid_logged if limit else len(self.valid_dl)

        avg_meter = AverageMeterDict()
        self.model.eval()

        for i, inputs in tqdm(enumerate(self.valid_dl), total=limit):
            if i >= limit:
                break

            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.device, non_blocking=True)

            outputs = self.model(inputs)

            viz = self.validate_step(inputs, outputs, avg_meter)
            self.viz_valid.add(viz)

        self.validation_logging(avg_meter)

    def train_logging(self, avg_meter, timer):
        loss = avg_meter.avgs['total']
        print(f'[{self.step + 1:_}/{self.cfg.training.iterations:_}] Loss: {loss:.4f}')

        imgs = self.viz_train.log_dict()
        extra = dict(time_per_batch=timer.avg, time_per_img=timer.avg / self.cfg.data.train_bs)

        model = self.model.main if hasattr(self.model, 'monodepth') else self.model
        depth_k = 'disp' if hasattr(self.model, 'monodepth') else 'depth'
        if False and hasattr(model, 'm') and model.use_SxTAM:
            d2s = {f'alpha_d2s_s{s}': wandb.Histogram(
                    model.m['semseg'][str(s)][f'SxTAM_{depth_k}'].alpha.tolist())
                        for s in model.stages}

            s2d = {f'alpha_s2d_s{s}': wandb.Histogram(
                    model.m[depth_k][str(s)][f'SxTAM_semseg'].alpha.tolist())
                        for s in model.stages}

            self.log({**d2s, **s2d})

        if False and hasattr(self.model, 'm') and model.use_CxTAM:
            d2s = {f'beta_d2s_s{s}': model.m['semseg'][str(s)][f'CxTAM_{depth_k}'].beta.item()
                    for s in model.stages}

            s2d = {f'beta_s2d_s{s}': model.m[depth_k][str(s)]['CxTAM_semseg'].beta.item()
                    for s in model.stages}

            self.log({**d2s, **s2d})

        self.log({**extra, **imgs, **self.get_lrs(),
                  **{f'train/{k}': v for k, v in avg_meter.avgs.items()}})

    def validation_logging(self, avg_meter):
        imgs = self.viz_valid.log_dict()

        logs = {}
        for k, v in avg_meter.avgs.items():
            logs[f'valid/{k}'] = v

        for name, metric in self.loss.metrics.items():
            for k, v in metric.get_scores().items():
                logs[f'metrics_{name}/{k}'] = v

        self.log({**logs, **imgs})

    def inference(self, inputs):
        self.model.train()

        for k, v in inputs.items():
            if torch.is_tensor(v) and v.device != self.device:
                inputs[k] = v.to(self.device, non_blocking=True)

        outputs = self.model(inputs)

        return outputs

    def needs_printing(self):
        return (self.step + 1) % self.cfg.training.print_interval == 0

    def needs_validating(self):
        return (self.step + 1) % current_val_interval(self.cfg, self.step + 1) == 0 \
            or self.needs_end()

    def needs_end(self):
        return (self.step + 1) == self.cfg.training.iterations

    def save_resume(self):
        modules = {k: getattr(self, k).state_dict() for k in self.saved_modules()}
        to_save = dict(**modules,
                       step=self.step + 1,
                       best=self.best_metric)

        save_path = os.path.join(self.logdir, f'resume.pkl')
        torch.save(to_save, save_path)
        return save_path

    def load_resume_(self):
        resume = self.cfg.training.resume
        assert os.path.isfile(resume), 'no checkpoint found'

        checkpoint = torch.load(resume)
        for module in self.saved_modules():
            if module == 'optimizer' or module == 'scheduler':
                continue
            print(module)
            getattr(self, module).load_state_dict(checkpoint[module], False) #)

        self.step, self.best_metric = checkpoint['step'], checkpoint['best']
        cpprint(f'Loaded checkpoint {resume} (step={self.step})', c='green')