from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop



def get_optimizer(cfg):
    optimizers = dict(sgd=SGD, adam=Adam, asgd=ASGD, adamax=Adamax, adadelta=Adadelta, \
        adagrad=Adagrad, rmsprop=RMSprop)

    if cfg['training']['optimizer'] is None:
        name = 'sgd'
    else:
        name = cfg['training']['optimizer']['name']

    if name not in optimizers:
        raise NotImplementedError(f'Optimizer {name} not implemented')

    print(f'Using {name} optimizer')
    return optimizers[name]

def setup_optimizer(cfg, optimizer, lr, backbone_lr, pose_lr, semantic_lr, grad_clip):
    cfg['training']['optimizer'] = dict(name=optimizer, lr=lr, backbone_lr=backbone_lr)

    if pose_lr is not None:
        cfg['training']['optimizer']['pose_lr'] = pose_lr
    if semantic_lr is not None:
        cfg['training']['optimizer']['segmentation_lr'] = semantic_lr
    if optimizer == 'sgd':
        cfg['training']['optimizer'].update(momentum=0.9, weight_decay=0.0005)

    cfg['CLIP_GRAD_NORM'] = grad_clip
    return cfg