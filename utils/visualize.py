import os

import torch
import wandb
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from utils.utils import colorize


class Visualize():
    def __init__(self, dir, max, segmap2color, max_percentile, wandb=True):
        self.dir = dir
        self.visualized = []
        self.max = max
        self.first = True
        self.segmap2color = segmap2color
        self.max_percentile = max_percentile
        self.topil = ToPILImage()
        self.wandb = wandb

    def add(self, batch):
        """Unwrap batch elements: {k: t[]} -> {k: t}[] and append to buffer."""
        for instance in zip(*batch.values()):
            if len(self) < self.max:
                self.visualized.append(
                    {k: v.detach().cpu() if torch.is_tensor(v) else v
                        for k, v in zip(batch.keys(), instance) if v is not None })

    def log_dict(self):
        if self.wandb:
            return self.wandb_log()

    def wandb_log(self):
        imgs = {f'{self.dir}_{i}_{k}': wandb.Image(v) for i, d in enumerate(self.visualized)
            for k, v in d.items() if self._needs_logging(k)}
        self.first = False
        self.visualized = []
        return imgs

    def seg(self, segmaps):
        if isinstance(segmaps, torch.Tensor):
            segmaps = segmaps.detach().cpu().numpy()
        if type(segmaps) is list:
            return segmaps
        return [self.segmap2color(m).permute(2,0,1) for m in segmaps]

    def gt_normals(self, normals, normalize=False):
        if normalize:
            normals = F.normalize(normals, dim=-3)
            normals[:, -1].abs_() # flip outward facing normals on z-axis

        encoded = (normals + 1)/2
        res = encoded.split(1)
        return res

    def normals(self, normals, mask, normalize=False):
        if normalize:
            normals = F.normalize(normals, dim=-3)
            normals[:, -1].abs_() # flip outward facing normals on z-axis

        # normals are mapped from xyz: [-1,1]x[-1,1]x[0,1] to [0,1]x[0,1]x[.5,1]
        encoded = (normals + 1)/2

        res = []
        white = torch.ones_like(normals[0])
        for x, m in zip(normals, mask):
            res.append(torch.where(m.expand(3,-1,-1), x, white))
        return res

    def edges(self, edges):
        res = [e.expand(3,*e.shape[1:]) for e in edges.float().split(1)]
        return res

    def depth(self, depth, mask=None, p=100):
        if type(depth) is list:
            return depth
        if mask is None or type(mask) is list:
            mask = torch.ones_like(depth)
            # mask = torch.ones(len(depth))
        mask = mask.cpu()

        res = []
        WHITE = torch.tensor([1.,1.,1.])[:,None].double()
        for x, m in zip(depth, mask):
            dmap = torch.tensor(colorize(x, max_percentile=p)).permute(2,0,1)
            dmap[:, (m==0)[0]] = WHITE
            res.append(dmap)
        return res

    def __len__(self):
        return len(self.visualized)

    def _needs_logging(self, key):
        return self.dir == 'train' or ('rgb' not in key) or self.first
