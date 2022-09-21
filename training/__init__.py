from torch import nn

from optim import *
from models.posenet import PoseNetwork
from models.padnetplus import PadnetPlus
from models.parts import get_backbone, get_head
from training.multi_setup import MultiSetup


def get_setup(cfg, device):
    m = cfg.setup.model
    setups = {}
    backbone = None

    if 'baseline' in cfg.setup.name:
        from models.multitask import MultiTask

        backbone, num_ch_enc = get_backbone(m.backbone, 'imnet', m.use_dilation)
        heads = nn.ModuleDict({name: get_head(task.head, num_ch_enc, **task.kwargs)
                    for name, task in m.tasks.items()})

        mtl = MultiTask(backbone, heads).to(device)
        setups.update(main=mtl)

    elif cfg.setup.name == 'padnet':
        from models.padnet import Padnet

        backbone, num_ch_enc = get_backbone(m.backbone, 'imnet', m.use_dilation)
        padnet = Padnet(backbone, num_ch_enc[-1], m.tasks).to(device)
        setups.update(main=padnet)

    elif cfg.setup.name == 'padnet+' or '3ways' in cfg.setup.name:
        from models.padnetplus import PadnetPlus

        backbone, num_ch_enc = get_backbone(m.backbone, 'imnet', m.use_dilation)
        heads = nn.ModuleDict({name: get_head(task.head, num_ch_enc, **task.kwargs)
                    for name, task in m.tasks.items()})

        padnetplus = PadnetPlus(backbone, heads, enc_1st_layers=[4,3,2], enc_2nd_layers=[1,0]).to(device)
        setups.update(main=padnetplus)

    elif cfg.setup.name in ['ours', 'ours-cxatt'] or 'ours-cxatt' in cfg.setup.name:
        from models.ours_xatt import OursAtt as Ours

        backbone, num_ch_enc = get_backbone(m.backbone, 'imnet', m.use_dilation)
        heads = nn.ModuleDict({name: get_head(task.head, num_ch_enc, **task.kwargs)
                    for name, task in m.tasks.items()})

        ours = Ours(backbone, heads, m.tasks, **cfg.setup.get('kwargs', {})).to(device)
        setups.update(main=ours)


    # Monodepth module setup as a plugin
    if 'monodepth' in cfg.setup.name:
        from models.monodepth import Monodepth

        # if backbone is None:
        #     backbone, num_ch_enc = get_backbone(m.backbone, m.pretrain, m.use_dilation)

        imnet_encoder = None
        if m.feature_dist:
            imnet_encoder, _ = get_backbone(m.backbone, 'imnet', use_dilation=m.use_dilation)
            imnet_encoder.requires_grad_(False).eval()

        posenet = PoseNetwork(m.pretrain, **m.pose_kwargs)
        monodepth = Monodepth(posenet, imnet_encoder).to(device)
        setups.update(monodepth=monodepth)

        # freeze option for backbone, decoder etc... (m.freeze_backbone)
        if m.get('freeze_backbone', False):
            print(f'freezing backbone!')
            setups['main'].backbone.requires_grad_(False)

    if not setups:
        raise NotImplementedError

    return setups['main'] if len(setups) == 1 else MultiSetup(setups)