# Based on https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py

import os
import random
import torch
import numpy as np

from dataset.base import BaseDataset
from utils.utils import recursive_glob
from .semseg import VKitti2Encoder

class VKitti2(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = VKitti2Encoder(self.n_classes)

        # See http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz
        self.full_res_shape = (1242, 375)
        self.fx = 725.0087
        self.fy = 725.0087
        self.u0 = 620.5
        self.v0 = 187

    def prepare_filenames(self):
        all_files = [p.replace(os.sep, '/') for p in sorted(recursive_glob(rootdir=self.root))]
        # files = [p for p in all_files if ('frames/rgb/Camera_0' in p and 'clone' not in p)]
        files = [p for p in all_files if '15-deg-left/frames/rgb/Camera_0' in p]
        random.shuffle(files)

        if self.split == 'all':
            return files

        split = int(len(files)*.8)
        files = files[:split] if self.split == 'train' else files[split:]
        return files

    def get_image_path(self, index, offset=0):
        """
            Get VKitti2 image instance from index and sequence offset.
            VKitti2's dataset structure in fs is as follow:
            Scene<xx>/<setting>/frames/<rgb|classSegmentation|depth>/Camera_<0|1>/<rgb|classgt|depth>_xxxxx.jpg|png
        """
        img_path = self.files[index]['name']
        if offset == 0:
            return img_path
        prefix, img_name = img_path.rsplit('_', 1)
        frame_number, ext = img_name.split('.')
        return f'{prefix}_{int(frame_number) + offset:05d}.{ext}'

    def get_segmentation_path(self, index):
        return self.files[index]['name'] \
            .replace('rgb/Camera_0/rgb_', 'classSegmentation/Camera_0/classgt_') \
            .replace('.jpg', '.png')

    def get_depth_path(self, index):
        return self.files[index]['name'] \
            .replace('rgb', 'depth') \
            .replace('.jpg', '.png')

    def get_depth(self, index, do_flip):
        import cv2
        path = self.get_depth_path(index)
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)

        if self.downsample_gt:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if do_flip:
            img = np.flip(img, -1) # h,w -> w gets flipped

        if self.inverse_depth:
            depth = 2**16 / (img + 1) # convert to normalized inverse depth
        else:
            depth = img / 100 # convert to m

        mask = np.ones_like(depth, dtype=bool)
        return depth[None], mask[None]

    def preprocess_depth(self, x):
        return x