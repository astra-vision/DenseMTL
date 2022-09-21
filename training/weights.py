import os

import torch

from vendor.three_ways.utils import download_model
from utils.logs import cpprint


def _get_path(weights, part):
    return os.path.join(os.environ['DOWNLOAD_DIR'], weights, part)

def _load_dict(path, model, device):
    state = torch.load(path, map_location=device)
    diffs = model.load_state_dict(state, False)
    return diffs

def _load_to(module, weights, part, device, name='?'):
    path = _get_path(weights, part)
    cpprint(f' Loading weights onto {name} ({type(module).__name__}) from {path}')
    diffs = _load_dict(path, module, device)
    cpprint(f' {diffs}', c='red')

def load_weights_(m, cfg, device):
    weights = cfg.setup.model.get('pretrain', None)
    if weights is None or weights == 'imnet':
        return None

    if cfg.setup.name in ['3ways_monodepth', 'ours-cxatt_monodepth', 'baseline_monodepth']: # separate into two...
        cpprint(f'[{cfg.setup.name}] Loading weights from {weights}', c='green')
        download_model(weights)

        _load_to(m.main.backbone, weights, 'encoder.pth', device, 'Backbone')
        _load_to(m.monodepth.pose_net.encoder, weights, 'pose_encoder.pth', device, 'Pose encoder')
        _load_to(m.monodepth.pose_net.decoder, weights, 'pose.pth', device, 'Pose decoder')

        for name, k in m.main.heads.named_children():
            _load_to(k, weights, 'depth.pth', device, f'{name} decoder')

    elif cfg.setup.name == '3ways':
        cpprint(f'[{cfg.setup.name}] Loading weights from {weights}', c='green')
        download_model(weights)

        if weights is not None:
            _load_to(m.backbone, weights, 'encoder.pth', device)
            for k in m.heads.children():
                _load_to(k, weights, 'depth.pth', device)