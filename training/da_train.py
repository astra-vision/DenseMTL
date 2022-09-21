import math

import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F

from utils.logs import cpprint
from dataset import build_dataset
from training.base import TrainerBase
from training import get_setup
from loss import get_loss
from optim import get_optimizer, get_scheduler
from models.discriminator import Discriminator
from vendor.ctrl_uda.cross_task_relation import DepthProb
from vendor.advent.discriminator import get_fc_discriminator
from vendor.advent.func import bce_loss, lr_poly, prob_2_entropy
from utils.utils import colorize

SRC_LABEL = 0
TGT_LABEL = 1


class Trainer(TrainerBase):
    def __init__(self, config):
        super().__init__(config)
        self._da_defaults()

        ## Main model
        self.model = get_setup(self.cfg, self.device)
        self.loss = get_loss(self.cfg, self.device)
        self.optimizer = get_optimizer(self.cfg, self.model)

        ## Task-pair interactions
        self.tasks = {k: v.kwargs.num_out_ch for k, v in self.cfg.setup.model.tasks.items()}
        self.pairs = {(s, t) for s in self.tasks for t in self.tasks if s != t}

        ## Upsampler
        d = self.cfg.data
        h, w = d.get('crop_h') or d.height, d.get('crop_w') or d.width
        self.upsample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)

        ## Adversarial modules
        self.setup_adv_modules()

    def setup_adv_modules(self):
        ch = dict(semseg=self.train_ds.n_classes,
                  depth=1,
                  normals=3)

        discr, self.discr_optim = {}, {}

        # scales = [f'ent_{x}' for x in self.da.ent_align]
        ent_disc_in_ch = sum(ch[t] for t in self.tasks)
        for s in self.da.ent_align:
            for t in self.tasks:
                k = f'{t}_{s}'
                discr[k] = Discriminator(in_ch=ch[t]).to(self.device)
                self.discr_optim[k] = Adam(discr[k].parameters(), lr=self.da.lr_D, betas=(0.9, 0.99))

        self.discr = nn.ModuleDict(discr)
        self.discr.train()

    def _da_defaults(self):
        self.da = self.cfg.training.da

        # self.da.bin_normals = self.da.get('bin_normals', False)
        self.da.doda = self.da.get('doda', True)

        cpprint(f'Setup setup', self.da, c='green')

    def train_itr(self):
        return zip(self.source_dl, self.target_dl)

    def saved_modules(self):
        modules = {'model', 'optimizer', 'discr'} # we are missing the optimizers
        return modules

    def adjust_lr_(self):
        max_iter = self.cfg.training.get('max_iterations', self.cfg.training.iterations)

        # for g in self.optimizer.param_groups:
        #     g['lr'] = lr_poly(base_lr=g['initial_lr'],
        #                       iter=self.step, max_iter=max_iter, power=0.9)

        for discr in self.discr_optim.values():
            for g in discr.param_groups:
                g['lr'] = lr_poly(base_lr=self.da.lr_D,
                                  iter=self.step, max_iter=max_iter, power=0.9)

    def epsilon_map(self, x, t, s, detach=False, src=None):
        pred, D = self.upsample(x[t, int(s)]), self.discr[f'{t}_{s}']

        if t == 'depth':
            d = torch.relu(pred)
            dmin, dmax = src['gt_depth'].min(), src['gt_depth'].max()
            e = (d-dmin)/(dmax-dmin)

        elif t == 'semseg':
            e = prob_2_entropy(pred.softmax(1))

        elif t == 'normals':
            e = F.normalize(pred, dim=-3)

        dout = D(e.detach() if detach else e)

        return dout

    def zero_grad_(self):
        # generator optimizer
        self.optimizer.zero_grad()

        # discriminators optimizers
        for optim in self.discr_optim.values():
            optim.zero_grad()

    def step_(self):
        self.optimizer.step()
        for optim in self.discr_optim.values():
            optim.step()

    def discr_require_grad_(self, freeze):
        # freeze/unfreeze discriminator optimizers
        for discr in self.discr.values():
            discr.requires_grad_(freeze)

    def train_step(self, inputs, avg_meter):
        loss = self._init_losses()
        src_x, tgt_x = inputs

        # doda config parameter can be set to an integer designating the step
        # from which adversarial training will be started off.
        DODA = self.da.doda and self.step >= self.da.doda

        # Adjust the learning rates of G'optim and Ds' optims
        self.adjust_lr_()

        # Clear gradient buffers of G and D.
        self.zero_grad_()

        ## (I) Train MTL network: disable gradient accumulation on discriminator
        self.discr_require_grad_(False)

        K = [f'{t}_{s}' for s in self.da.ent_align for t in self.tasks]

        # (I.1) train on source with task supervision
        if True:
            src_x['is_src'] = True
            src_pred = self.inference(src_x)

            task_losses, viz = self.loss(src_x, src_pred, self.viz_train)
            loss.src_task = task_losses['total']
            loss.update(task_losses)
            loss.src_task.backward()

        # (I.2) Adversarial training on target and fool the discriminator
        if DODA:
            tgt_x['is_src'] = False
            tgt_pred = self.inference(tgt_x)

            for s in self.da.ent_align:
                for t in self.tasks:
                    dmap = self.epsilon_map(tgt_pred, t, s, src=src_x, detach=False)
                    loss[f'adv_{t}_{s}'] = self.da.lambda_adv_ent*bce_loss(dmap, SRC_LABEL)

            adv_loss = torch.stack([loss[f'adv_{k}'] for k in K]).mean()
            adv_loss.backward()


        ## (II) Train discriminator: enable gradient accumulation discriminator
        # No need to recompute src and tgt predictions since G's outputs are detached for D step
        self.discr_require_grad_(True)

        # (II.1) Inference on SRC
        if DODA:
            for s in self.da.ent_align:
                for t in self.tasks:
                    dmap = self.epsilon_map(src_pred, t, s, src=src_x, detach=True)
                    loss[f'dis_{t}_{s}'] = bce_loss(dmap, SRC_LABEL)

            disc_loss = torch.stack([loss[f'dis_{k}'] for k in K]).mean()
            disc_loss.backward()

        # (II.2) Inference on TGT
        if DODA:
            for s in self.da.ent_align:
                for t in self.tasks:
                    dmap = self.epsilon_map(tgt_pred, t, s, src=src_x, detach=True)
                    loss[f'dis_{t}_{s}'] = bce_loss(dmap, TGT_LABEL)

            disc_loss = torch.stack([loss[f'dis_{k}'] for k in K]).mean()
            disc_loss.backward()

        # update the parameters of the discriminators
        self.step_()

        avg_meter.update(loss)
        return viz

    def validate_step(self, inputs, outputs, avg_meter):
        loss, viz = self.loss(inputs, outputs, self.viz_valid, train=False)
        avg_meter.update(loss)
        return viz

    def setup_dls(self):
        ## Dataset initialization
        data = self.cfg.data

        src_config = {**data, **data.source}
        tgt_config = {**data, **data.target}

        common = dict(load_sequence=data.load_sequences,
                      load_labels=data.load_semantic_gt)

        self.source_ds = self.train_ds = \
            build_dataset(split=data.src_split, cfg=src_config, **common)
        self.target_ds = build_dataset(split=data.tgt_split, cfg=tgt_config, **common)
        self.valid_ds = build_dataset(split=data.val_split, cfg=tgt_config, **common)

        ## Dataloader initialization
        source_dl_cfg = dict(batch_size=self.cfg.data.train_bs,
                             num_workers=self.cfg.training.n_workers,
                             shuffle=self.cfg.training.shuffle_trainset,
                             pin_memory=True, drop_last=True)

        self.source_dl = DataLoader(self.source_ds, **source_dl_cfg)
        self.target_dl = DataLoader(self.target_ds, **source_dl_cfg)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=self.cfg.data.valid_bs,
            num_workers=True, pin_memory=True, drop_last=True)

        if True:
            x = self.source_ds[0]
            gt = x['lbl']
            cpprint(f'Source dataset {len(self.source_dl):_} batches of size {len(gt)}', gt.shape, gt.unique(), c='red')
            # gt, mask = x['gt_depth'], x['depth_mask']
            # cpprint('range', gt[mask].min(), gt[mask].max(), 'mask density', mask.mean())

            x = self.target_ds[0]
            gt = x['lbl']
            cpprint(f'Target dataset {len(self.target_dl):_} batches of size {len(gt)}', gt.shape, gt.unique(), c='red')
            # gt, mask = x['gt_depth'], x['depth_mask']
            # cpprint('range', gt[mask].min(), gt[mask].max(), 'mask density', mask.mean())

            x = self.valid_ds[0]
            gt = x['lbl']
            cpprint(f'Validation dataset {len(self.valid_dl):_} batches of size {len(gt)}', gt.shape, gt.unique(), c='red')
            # gt, mask = x['gt_depth'], x['depth_mask']
            # cpprint('range', gt[mask].min(), gt[mask].max(), 'mask density', mask.mean())

    @torch.no_grad()
    def log_corr(self):
        return True

        hist64x64 = torch.zeros(1, 64, 64)
        hist8x64 = torch.zeros(1, 8, 64)
        hist128x128 = torch.zeros(1, 128, 128)
        hist8x128 = torch.zeros(1, 8, 128)
        hist8 = torch.zeros(1, 8 + 1)

        ones = torch.ones(1, 320*640).float()
        # for i, x in tqdm(enumerate(self.source_dl), total=100):
        for x in tqdm(self.target_dl):

            gt = x['lbl'].flatten(1)
            gt[gt == 250] = 8
            hist8.scatter_add_(1, gt, src=ones)

            y = self.inference(x)['f']
            ## D_feats ##############################################
            dq = y['depth', 0, 'feats'].flatten(-2).softmax(1)

            #  x S_feats
            sq = y['semseg', 0, 'feats'].flatten(-2).softmax(1)

            ch_hist = torch.einsum('bdS,bsS->bsd', dq, sq).cpu()
            hist64x64 += ch_hist / ch_hist.sum((1,2))

            # x S_logits
            sq = y['semseg', 0, 'logits'].flatten(-2).softmax(1)
            ch_hist = torch.einsum('bdS,bsS->bsd', dq, sq).cpu()
            hist8x64 += ch_hist / ch_hist.sum((1,2))

            ## D_feats ##############################################
            dq = y['depth', 2, 'feats'].flatten(-2).softmax(1)

            # x S_feats
            sq = y['semseg', 2, 'feats'].flatten(-2).softmax(1)
            ch_hist = torch.einsum('bdS,bsS->bsd', dq, sq).cpu()
            hist128x128 += ch_hist / ch_hist.sum((1,2))

            # x S_logits
            sq = y['semseg', 2, 'logits'].flatten(-2).softmax(1)
            ch_hist = torch.einsum('bdS,bsS->bsd', dq, sq).cpu()
            hist8x128 += ch_hist / ch_hist.sum((1,2))


        co = lambda x: wandb.Image(colorize(torch.log(1+x), cmap='plasma'))
        # wandb.log({'hist8': co(hist8[None].expand(-1,2,-1))})
        # print(hist8)

        wandb.log({'hist64x64': co(hist64x64),
                   'hist8x64': co(hist8x64),
                   'hist128x128': co(hist128x128),
                   'hist8x128': co(hist8x128)})