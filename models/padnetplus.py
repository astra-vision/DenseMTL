import torch.nn as nn

from training.setup import Setup
from vendor.mtlpt import MultiTaskDistillationModule

class PadnetPlus(Setup):
    def __init__(self, backbone, heads, enc_1st_layers, enc_2nd_layers):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.enc_1st_layers = enc_1st_layers
        self.enc_2nd_layers = enc_2nd_layers

        tasks = self.heads.keys()
        self.dist_module = MultiTaskDistillationModule(tasks, tasks, 128)

        modules = (self.dist_module.self_attention, self.heads)
        self.params = nn.ModuleDict({
            t: nn.ModuleList([getattr(m, t, nn.Identity()) for m in modules]) for t in tasks})

    def forward(self, x):
        # Pass on shared encoder
        rgb = x['color_aug', 0, 0]
        features = self.backbone(rgb)

        # Decoder first section with intermediate outputs
        dec_1st = {t: head(features, exec_layer=self.enc_1st_layers)
            for t, head in self.heads.named_children()}

        # Refinement of features with distillation module
        if len(self.heads) > 1:
            inter_features = {f'features_{t}': f['upconv', 2] for t, f in dec_1st.items()}
            distilled = self.dist_module(inter_features)
        else:
            t = list(self.heads.keys())[0]
            distilled = {t: dec_1st[t]['upconv', 2]}

        # Decoder second section with final outputs
        dec_2nd = {t: head(features, x=distilled[t], exec_layer=self.enc_2nd_layers)
            for t, head in self.heads.named_children()}

        # Sort out all outputs for supervision
        out = {(t, s): f for o in (dec_1st, dec_2nd) for t, d in o.items()
            for s, f in d.items() if isinstance(s, int)}

        out['features'] = features # for monodepth
        return out