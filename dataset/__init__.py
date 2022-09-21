from .cityscapes import Cityscapes
from .vkitti2 import VKitti2
from .synthia import Synthia
from .nyudv2 import NYUDv2

from utils.logs import cpprint

def get_dataset(name):
    return {'vkitti2':      VKitti2,
            'synthia':      Synthia,
            'nyudv2':       NYUDv2,
            'cityscapes':   Cityscapes}[name]

def build_dataset(cfg, split, load_labels=True, load_sequence=True):
    dataset = get_dataset(cfg['dataset'])

    common = dict(
        root=cfg['path'],
        n_classes=cfg['n_classes'],
        split=split,
        img_size=(cfg['height'], cfg['width']),
        crop_h=cfg.get('crop_h', cfg['height']),
        crop_w=cfg.get('crop_w', cfg['width']),
        frame_idxs=cfg['frame_ids'],
        num_scales=cfg['num_scales'],
        load_labels=load_labels,
        load_sequence=load_sequence,
        inverse_depth=cfg.get('inverse_depth', False),
        load_gt_depth=True,
        load_onehot=cfg.get('load_onehot', False),
        color_full_scale=cfg['color_full_scale'],
        crop_center=cfg.get('crop_center', False))


    if split in ['train', 'all']:
        split_kargs = dict(
            downsample_gt=True,
            augmentations=cfg['augmentations'],
            dataset_seed=cfg['dataset_seed'],
            restrict_dict=cfg['restrict_to_subset'],
            load_labeled=cfg.get('load_labeled', True),
            load_unlabeled=cfg.get('load_unlabeled', False),
            only_sequences_with_segmentation=cfg['only_sequences_with_segmentation'])

    elif split == 'val':
        split_kargs = dict(
            downsample_gt=cfg['val_downsample_gt'],
            augmentations={},
            num_val_samples=cfg.get('num_val_samples', None),
            only_sequences_with_segmentation=cfg.get('val_only_sequences_with_segmentation', True))

    merge_kwargs = {**common, **split_kargs}
    # cpprint(dataset, split, merge_kwargs)

    return dataset(**merge_kwargs)