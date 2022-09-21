import os, random
from PIL import Image

import torch
import numpy as np

from dataset.utils import pil_loader
from utils.utils import recursive_glob
from .base import BaseDataset
from utils.logs import cpprint
from .semseg import SynthiaEncoder


class Synthia(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SynthiaEncoder(self.n_classes)

        # See SYNTHIA_VIDEO_SEQUENCES subset https://synthia-dataset.net/downloads/
        self.full_res_shape = (1280, 760)
        self.fx = 532.740352
        self.fy = 532.740352
        self.u0 = 640
        self.v0 = 380

    def prepare_filenames(self):
        files = [p.replace(os.sep, '/') for p in recursive_glob(rootdir=self.root) if 'RGB' in p]
        random.shuffle(files)

        if self.split == 'all':
            return files
        split = int(len(files)*.8)
        files = files[:split] if self.split == 'train' else files[split:]
        return files

    def get_image_path(self, index, offset=0):
        img_path = self.files[index]['name']
        if offset == 0:
            return img_path
        prefix, img_name = img_path.rsplit('_', 1)
        frame_number, ext = img_name.split('.')
        return f'{prefix}_{int(frame_number) + offset:05d}.{ext}'

    ## Segmentation
    def get_segmentation_path(self, index):
        return self.files[index]['name'].replace('RGB', 'GT/LABELS').replace('_PIT', '')

    def get_segmentation(self, index, do_flip):
        depth_path = self.get_segmentation_path(index).replace('/', os.sep)
        w, h = (self.width, self.height) if self.downsample_gt else (-1, -1)
        lbl = pil_loader(depth_path, w, h, is_segmentation=True, is_unsigned=True)
        lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT) if do_flip else lbl
        return lbl

    def preprocess_semseg(self, semseg):
        lbl = np.array(semseg, dtype=np.int16)
        lbl = self.encode_segmap(lbl)
        return torch.from_numpy(lbl).long()

    ## Depth
    def get_depth_path(self, index):
        return self.files[index]['name'].replace('RGB', 'Depth/Depth').replace('_PIT', '')

    def get_depth(self, index, do_flip):
        import cv2
        path = self.get_depth_path(index).replace('/', os.sep)
        img = cv2.imread(path, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)

        if self.downsample_gt:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if do_flip:
            img = np.flip(img, -1) # b,c,h,w -> w gets flipped

        img = img[None] *.01 # to meters

        if self.inverse_depth:
            img = img * 100 # convert to cm
            img = 65536.0 / (img + 1) # convert to normalized invert depth

        mask = np.ones_like(img, dtype=bool)
        return img, mask

    def preprocess_depth(self, x):
        return x
