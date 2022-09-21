import torch.nn as nn

from training.setup import Setup


class MultiTask(Setup):
    def __init__(self, backbone: nn.Module, heads: nn.ModuleDict, scale=4):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.inter = nn.Upsample(scale_factor=2**scale, mode='bilinear')

    def forward(self, x):
        rgb = x['color_aug', 0, 0]
        features = self.backbone(rgb)

        heads = self.heads.named_children()
        # construct returned dict containing predictions and feature maps
        out = {(task, 0): self.inter(head(features)) for task, head in heads}
        out['f'] = features[-1] # save bottleneck for domain alignment

        return out