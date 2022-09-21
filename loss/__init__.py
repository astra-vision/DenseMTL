
from .depth import L1, BerHu, UnsupervisedDepthLoss
from .semseg import SemsegLoss
from .normals import NormalsLoss
from .gt_normals import NormalsGTLoss
from .edges import EdgesLoss

loss_fn = dict(L1_depth=L1, # uses non-inverted depth values
            berhu=BerHu, # expects inverse-normalized depth values
            ssde=UnsupervisedDepthLoss,
            normals=NormalsLoss,
            gt_normals=NormalsGTLoss,
            edges=EdgesLoss,
            cross_entropy=SemsegLoss)

def get_loss(cfg, device):
    p = cfg.setup.loss

    if cfg.setup.name in ['baseline', 'padnet', 'padnet+', 'cross_stitch', 'ours', 'ours-cxatt', \
        '3ways_monodepth', '3ways', 'ours-cxatt_monodepth', 'baseline_monodepth']: # is this list necessary?
        from .multi_loss import MultiLoss
        losses = {name: loss_fn[task.loss](**task.kwargs) for name, task in p.tasks.items()}
        return MultiLoss(losses, **_kwargs(p)).to(device)

    # elif cfg.setup.name == 'monodepth':
    #     from .depth.unsupervised import UnsupervisedDepthLoss
    #     return UnsupervisedDepthLoss(batch_size=cfg.data.train_bs, **p).to(device)

    raise NotImplementedError

def _kwargs(d):
    return d.get('kwargs', {}) or {}
