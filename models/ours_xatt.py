import torch
import torch.nn as nn

from training.setup import Setup
from vendor.mtlpt import SABlock
from .components import Extractor, Project, SxTAM
from utils.logs import cpprint
from utils.train import count


DEBUG = False
class OursAtt(Setup):
    stage_nb = {4: (256, 1), 3: (256, 1), 2: (128, 2), 1: (128, 4), 0: (64, 8)}

    def __init__(self, backbone, heads, tasks, ablation, enc_1st_layers=[], enc_2nd_layers=[], enc_3rd_layers=[],
                 enc_layers=None, alignment=None, k=None, att_ent_align=False, att_feat_align=False, stages=None,
                 interaction_hist=False, cxtam_align=False, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.heads = heads

        self.ablation = ablation
        self.stages = stages

        if enc_layers is None:
            self.enc_layers = [enc_1st_layers, enc_2nd_layers, enc_3rd_layers]
        else:
            self.enc_layers = enc_layers

        assert sum(self.enc_layers, []) == list(range(4, -1, -1))

        self.use_SA = 'SA' in self.ablation
        self.use_SxTAM = 'SxTAM' in self.ablation
        self.use_CxTAM = 'CxTAM' in self.ablation
        self.no_distil = not (self.use_SA or self.use_SxTAM or self.use_CxTAM)
        self.split = 'split' in self.ablation
        self.alpha = 'no_alpha' not in self.ablation
        self.sa_out_ch = 2**('full_sa' not in self.ablation)

        assert 'mult' not in self.ablation # should be prod not mult
        self.base_fusion = 'prod' in self.ablation or 'add' in self.ablation
        self.include_base = not self.base_fusion

        self.pairs = {(s, t) for s in tasks for t in tasks if s != t}
        self.tasks = set(self.heads)
        self._init_distil_modules(tasks, **kwargs)

        cpprint(f'Ours init:'
                f'\n Ablation parameters={self.ablation}'
                f'\n Task pairs: {self.pairs}'
                f'\n Decoder sections {self.enc_layers}'
                f'\n Stages of distillation blocks: {self.stages}'
                f'\n Heads={count(self.heads):_} + xBlock={0 if self.no_distil else count(self.m):_}'
                f'\n include_base={self.include_base}, base_fusion={self.base_fusion}',
                c='magenta')

    def get(self, name, scale, target, source=None):
        k = name if source is None else f'{name}_{source}'
        module = self.m[target][str(scale)][k]
        return module

    def _init_distil_modules(self, tasks, freeze_batchnorms=False):
        proj_mult = (self.use_SxTAM + self.use_CxTAM + self.use_SA/self.sa_out_ch)
        scales = {s: self.stage_nb[s] for s in self.stages}
        num_out_ch = {t: tasks[t].kwargs.num_out_ch for t in tasks}
        others = lambda t: {s for s in tasks if s != t}
        post_proj_mult = len(self.heads) if self.include_base else len(self.heads) - 1
        div = 2**self.split

        self.m =  nn.ModuleDict(
            {t: nn.Identity() if self.no_distil else nn.ModuleDict({
                **{str(scale): nn.ModuleDict({
                    **{f'extract_{s}': Extractor(ch//div, ch//div) for s in others(t)},

                    **{f'SA_{s}': SABlock(ch//div, ch//self.sa_out_ch)
                        if self.use_SA else nn.Identity() for s in others(t)},

                    **{f'SxTAM_{s}': SxTAM(ch//div, ds, self.alpha)
                        if self.use_SxTAM else nn.Identity() for s in others(t)},

                    **{f'CxTAM_{s}': CxTAM()
                        if self.use_CxTAM else nn.Identity() for s in others(t)},

                    **{f'proj_{s}': Project(int(proj_mult*ch//div), ch//div)
                        for s in others(t)},

                    'post': Project(post_proj_mult*ch//div, ch),
                })

                for scale, (ch, ds) in scales.items()}})
            for t in tasks})

        # provide easy access to parameters per-task, .parameters() won't return duplicates.
        self.params = nn.ModuleDict({
            t: nn.ModuleList([getattr(m, t) for m in (self.heads, self.m)]) for t in tasks})

    def tensor_split(self, x):
        # torch.Tensor.tensor_split is available @ v1.8.0
        if self.split:
            return x.split(x.size(1)//2, dim=1)
        return [x]

    def distillation_block(self, code, stage):
        if self.no_distil:
            cpprint('early return which bypassed cross distillation') if DEBUG else None
            return code
        out = {}
        for s, t in self.pairs:
            cpprint(f'handling pair {s}->{t}') if DEBUG else None
            # base code is just the features output
            f_base = code[t]

            # extract relevant features for target task from source task
            extractor = self.get('extract', stage, t, s)
            f_dir = extractor(code[s])

            f = []
            if self.use_SA:
                cpprint(f'computing sa') if DEBUG else None
                sa_fn = self.get('SA', stage, t, s)
                sa = sa_fn(f_dir)
                f.append(sa)

            if self.use_SxTAM:
                cpprint(f'computing SxTAM') if DEBUG else None
                sxtam_fn = self.get('SxTAM', stage, t, s)
                s_corr = sxtam_fn(f_base, f_dir)
                f.append(s_corr)

            if self.use_CxTAM:
                cpprint(f'computing CxTAM') if DEBUG else None
                cxtam_fn = self.get('CxTAM', stage, t, s)
                c_corr = cxtam_fn(f_base, f_dir)
                f.append(c_corr)

            # fused and projected together
            cpprint(f'fusing codes: {len(f)}') if DEBUG else None
            project = self.get('proj', stage, t, s)
            out[t, s] = project(torch.cat(f, dim=1))

        ## Aggregate and fuse with base code
        neg = lambda t: {s for s in self.tasks if s != t}
        cpprint(f'fuse back step: T-semseg={neg("semseg")}; T-depth={neg("depth")}; T-normals={neg("normals")}') if DEBUG else None

        distilled = {}
        for t in self.tasks:
            fuse = self.get('post', stage, t)

            if self.include_base:
                distilled[t] = fuse(torch.cat([code[t]] + [out[t, s] for s in neg(t)], 1))

            elif 'prod' in self.ablation:
                cpprint(f'using [prod] for base fusion') if DEBUG else None
                distilled[t] = code[t] * fuse(torch.cat([out[t, s] for s in neg(t)], 1))

            elif 'add' in self.ablation:
                cpprint(f'using [add] for base fusion') if DEBUG else None
                distilled[t] = code[t] + fuse(torch.cat([out[t, s] for s in neg(t)], 1))

        return distilled

    def forward(self, x):
        # Inference on shared encoder
        rgb = x['color_aug', 0, 0]
        features = self.backbone(rgb)

        x, dec_res = {t: None for t in self.heads}, []
        for j, section in enumerate(self.enc_layers): # j=section number
            last = section[-1]

            res = {t: head(enc_outs=features, x=x[t], exec_layer=section)
                for t, head in self.heads.named_children()}
            dec_res.append(res)

            x = {t: res[t]['upconv', last] for t in self.heads}

            if len(self.stages) > j:
                cpprint('> going through attention block') if DEBUG else None
                assert self.stages[j] == last, 'block must be placed at the end of decoder section'
                x = self.distillation_block(x, stage=last)

        out = {(t, scale): f for o in dec_res for t, d in o.items()
            for scale, f in d.items() if isinstance(scale, int)}

        return out