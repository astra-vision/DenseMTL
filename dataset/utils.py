from copy import deepcopy
from functools import lru_cache
from PIL import Image

import numpy as np
import imageio

from utils.train import np_local_seed


def get_size(orig_img, width, height):
    size = [width, height]
    if size[0] == -1: size[0] = orig_img.width
    if size[1] == -1: size[1] = orig_img.height
    return size


# 3Gb / 300k = 10000 (per worker)
@lru_cache(maxsize=5000)
def _load_lru_cache(*args, **kwargs):
    return _load(*args, **kwargs)


def resample(ftype):
    if ftype == 'semseg':
        return Image.NEAREST
    if ftype == 'depth':
        return Image.BILINEAR #NEAREST
    if ftype == 'rgb':
        return Image.ANTIALIAS
    return NotImplementedError

def _load(_path, is_segmentation, is_depth, width, height, is_unsigned):
    ftype = 'semseg' if is_segmentation else ('depth' if is_depth else 'rgb')
    sampling = resample(ftype)

    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(_path, 'rb') as f:
        if is_unsigned: # only way I could find to open a bloody ushort
            img = Image.fromarray(np.array(imageio.imread(_path, format='PNG-FI'))[...,0])
            return img.resize(get_size(img, width, height), Image.NEAREST)

        with Image.open(f) as img:
            mode = 'RGB' if ftype == 'rgb' else None
            img = img.convert(mode)
            return img.resize(get_size(img, width, height), sampling)


def pil_loader(path, std_width, std_height, is_segmentation=False, is_depth=False, lru_cache=False, is_unsigned=False):
    if lru_cache:
        load_fn = _load_lru_cache
    else:
        load_fn = _load
    return load_fn(path, is_segmentation, is_depth, std_width, std_height, is_unsigned)


def restrict_to_subset(files, mode, n_subset, seed, load_labeled, load_unlabeled, subset=None):
    assert mode == "fixed" or subset is None
    # print(f'Restrict subset from {len(files)} to {n_subset} images ...')

    if mode == "random":
        with np_local_seed(seed):
            indices = np.random.permutation(len(files))
        indices = indices[:n_subset]
    elif mode == "fixed":
        assert len(subset) == n_subset and subset is not None
        indices = subset
    else:
        raise NotImplementedError(mode)

    indices.sort()
    labeled_files = [f for i, f in enumerate(files) if i in indices]
    unlabeled_files = [f for i, f in enumerate(files) if i not in indices]

    for unlabeled in unlabeled_files:
        unlabeled['labeled'] = False

    assert len(labeled_files) == n_subset
    assert len(unlabeled_files) == len(files) - n_subset

    if load_labeled and load_unlabeled:
        concat_files = deepcopy(labeled_files)
        concat_files.extend(unlabeled_files)
        files = concat_files
    elif load_labeled:
        files = labeled_files
    elif load_unlabeled:
        files = unlabeled_files
    else:
        raise ValueError("Neither unlabeled or labeled data is specified to be loaded.")
    # print("Keep %d images" % (len(files)))

    return files

def infinite_iterator(generator):
    while True:
        for data in generator:
            yield data