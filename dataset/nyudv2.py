# nyudv2_dataset

# Based on https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py

import os, random
from PIL import Image

import torch
import numpy as np
from torchvision.transforms import functional as tf

from dataset.base import BaseDataset
from dataset.utils import pil_loader
from utils.utils import recursive_glob
from .semseg import NYUDv2Encoder


class NYUDv2(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = NYUDv2Encoder(self.n_classes)
        self.load_gt_normals = True
        self.load_gt_edges = True

        self.full_res_shape = (1242, 375)
        self.fx = 725.0087
        self.fy = 725.0087
        self.u0 = 620.5
        self.v0 = 187

    def prepare_filenames(self):
        split = str(self.root / f'gt_sets/{self.split}.txt')
        with open(split, 'r') as file:
            paths = file.readlines()
        files = [str(self.root / 'images' / p.rstrip('\n')) + '.png' for p in paths]
        return files

    def get_image_path(self, index, offset=0):
        img_path = self.files[index]['name']
        return img_path

    def get_segmentation_path(self, index):
        return self.files[index]['name'] \
            .replace('images', 'segmentation')

    def get_depth_path(self, index):
        return self.files[index]['name'] \
            .replace('images', 'depth') \
            .replace('.png', '.npy')

    def get_normal_path(self, index):
        return self.files[index]['name'] \
            .replace('images', 'normals')

    def get_edge_path(self, index):
        return self.files[index]['name'] \
            .replace('images', 'edge')

    def get_segmentation(self, index, do_flip):
        lbl = Image.open(self.get_segmentation_path(index))
        lbl = np.array(lbl, dtype=np.float32, copy=False) - 1
        lbl[lbl == -1] = 250

        if self.downsample_gt:
            import cv2
            lbl = cv2.resize(lbl, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if do_flip:
            lbl = np.fliplr(lbl)

        return lbl

    def get_depth(self, index, do_flip):
        path = self.get_depth_path(index)
        lbl = np.load(path).astype(np.float32)*100 # in cm

        if self.downsample_gt:
            import cv2
            lbl = cv2.resize(lbl, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if do_flip:
            lbl = np.fliplr(lbl)

        if self.inverse_depth:
            depth = 2**16 / (lbl + 1) # convert to normalized inverse depth
        else:
            depth = lbl / 100 # convert to meters

        mask = np.ones_like(depth, dtype=bool)
        return depth[None], mask[None]

    def get_normals(self, index, do_flip):
        lbl_path = self.get_normal_path(index)
        if self.downsample_gt:
            lbl = pil_loader(lbl_path, self.width, self.height, is_segmentation=True)
        else:
            lbl = pil_loader(lbl_path, -1, -1, is_segmentation=True)

        lbl = 2*self.to_tensor(lbl)-1 # remap to [-1,1]

        if do_flip:
            lbl = tf.hflip(lbl)
            lbl[0] *= -1
        return lbl

    def get_edges(self, index, do_flip):
        path = self.get_edge_path(index)
        lbl = Image.open(path)
        lbl = np.expand_dims(np.array(lbl, dtype=np.float32, copy=False), axis=2) / 255.

        if self.downsample_gt:
            import cv2
            lbl = cv2.resize(lbl, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            lbl[lbl > 0] = 1
        if do_flip:
            lbl = np.fliplr(lbl)
        return lbl

    def preprocess_semseg(self, semseg):
        x=self.to_tensor(semseg.copy()).long()
        return x[0]

    def preprocess_normals(self, normals):
        return normals

    def preprocess_depth(self, depth):
        return depth

    def preprocess_edges(self, edges):
        return self.to_tensor(edges.copy()).long().squeeze(0)