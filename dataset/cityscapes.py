import os
from PIL import Image

import json
import torch
import numpy as np

from dataset.base import BaseDataset
from utils.utils import recursive_glob
from .semseg import CityscapesEncoder


class Cityscapes(BaseDataset):
    def __init__(self, **kwargs):
        super(Cityscapes, self).__init__(**kwargs)
        self.encoder = CityscapesEncoder(self.n_classes)

        # See https://www.cityscapes-dataset.com/file-handling/?packageID=8
        self.full_res_shape = (2048, 1024)
        self.fx = 2262.52
        self.fy = 2265.3017905988554
        self.u0 = 1096.98
        self.v0 = 513.137

    def prepare_filenames(self):
        if self.img_size == (512, 1024) or self.img_size == (320, 640):
            mono_root, seq_root = 'leftImg8bit_small', 'leftImg8bit_sequence_small'
        elif self.img_size == (256, 512):
            mono_root, seq_root = 'leftImg8bit_small', 'leftImg8bit_sequence_small'
        else:
            raise NotImplementedError(f"Unexpected image size {self.img_size}")

        self.images_base = str(self.root / mono_root / self.split)
        self.sequence_base = str(self.root / seq_root / self.split)
        self.annotations_base = str(self.root / 'gtFine' / self.split)
        self.depth_base = str(self.root / 'disparity' / self.split)

        path = self.images_base if self.only_sequences_with_segmentation else self.sequence_base
        files = sorted(recursive_glob(rootdir=path))

        return files

    def get_image_path(self, index, offset=0):
        img_path = self.files[index]["name"].rstrip()
        if offset != 0:
            img_path = img_path.replace(self.images_base, self.sequence_base)
            prefix, frame_number, suffix = img_path.rsplit('_', 2)
            img_path = f'{prefix}_{int(frame_number) + offset:06d}_{suffix}'
        return img_path

    def get_segmentation_path(self, index):
        img_path = self.files[index]["name"].rstrip()
        return os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

    def get_depth_path(self, index):
        img_path = self.files[index]["name"].rstrip()
        depth_path = os.path.join(
            self.depth_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + 'disparity.png')
        return depth_path

    def get_cam_params(self, d_path):
        c_path = d_path.replace('disparity', 'camera').replace('png', 'json')
        with open(c_path) as file:
            calib = json.load(file)
        baseline = calib['extrinsic']['baseline']
        fx = calib['intrinsic']['fx']
        return baseline, fx

    def get_depth(self, index, do_flip):
        ## Read image file
        d_path = self.get_depth_path(index)

        img = Image.open(d_path)
        img = img.resize((self.width, self.height), Image.NEAREST)
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = np.asarray(img, np.float32)[None]

        ## Preprocessing
        # CTRL-UDA https://github.com/susaha/ctrl-uda/blob/234c0ed807252fcf771da58ba3dc5b18396ae8a4/ctrl/dataset/base_dataset.py#L129
        baseline, fx = self.get_cam_params(d_path)

        img = img - 1
        with np.errstate(invalid='ignore'):
            mask = img > 0.0
        img = img / 256

        # find maximum depth value in map
        disparity_min = np.amin(img[mask])
        depth_max = (baseline * fx) / disparity_min # convert from disparity to depth values

        # can avoid the warning w/ a mask assignment
        with np.errstate(invalid='ignore'):
            img = (baseline * fx) / img  # computing the depth
        img[~mask] = depth_max

        # img = np.where(mask, (baseline * fx) / img, depth_max)

        if self.inverse_depth:
            img *= 100 # convert to cm depth values
            img = 65536.0 / (img + 1) # convert to normalized inverse depth

        return img, mask

    def preprocess_depth(self, x):
        return x