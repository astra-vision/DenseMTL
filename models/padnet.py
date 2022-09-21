import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from training.setup import Setup
from utils.train import resize_input
from vendor.mtlpt import InitialTaskPredictionModule, MultiTaskDistillationModule


class Padnet(Setup):
    def __init__(self, backbone, num_ch, tasks, inter_num_ch=256):
        super().__init__()
        self.backbone = backbone

        self.tasks = tasks.keys()
        main_tasks_ch = {t: p.kwargs.num_out_ch for t, p in tasks.items()
            if not getattr(p, 'auxiliary', False)}
        aux_tasks_ch = {t: p.kwargs.num_out_ch for t, p in tasks.items()}

        self.first_block = InitialTaskPredictionModule(aux_tasks_ch, self.tasks, num_ch)

        self.dist_module = MultiTaskDistillationModule(main_tasks_ch.keys(), self.tasks, inter_num_ch)

        self.second_block = nn.ModuleDict({
            t: nn.Sequential(Bottleneck(inter_num_ch, inter_num_ch//4),
                             Bottleneck(inter_num_ch, inter_num_ch//4),
                             nn.Conv2d(inter_num_ch, num_out_ch, 1))
            for t, num_out_ch in main_tasks_ch.items()})

        # sort parameters by task for supervision
        modules = (self.first_block.layers, self.dist_module.self_attention, self.second_block)
        self.params = nn.ModuleDict({
            t: nn.ModuleList([getattr(m, t, nn.Identity()) for m in modules]) for t in tasks})

    def forward(self, x):
        rgb = x['color_aug', 0, 0]
        bottleneck = self.backbone(rgb)[-1]

        # Intermediate predictions
        x = self.first_block(bottleneck)
        inter_out = {(t, 2): x[t] for t in self.tasks}

        # Feature refinement with cross-task distillation
        x = self.dist_module(x)

        # Final prediction section
        x = {t: self.second_block[t](x[t]) for t in self.tasks}
        final_out  = {(t, 0): resize_input(x[t], rgb, False) for t in self.tasks}

        return {**inter_out, **final_out}