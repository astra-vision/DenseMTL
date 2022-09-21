import os, random, sys, math
from pathlib import Path
from PIL import Image

import torch
import numpy as np
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as tf

from dataset.utils import pil_loader, restrict_to_subset
from utils.logs import cpprint


BGR_MEAN = (104.00698793, 116.66876762, 122.67891434)
class BaseDataset(data.Dataset):
    def __init__(
            self,
            root,
            split="train",
            n_classes=0,
            img_size=(512, 1024),
            crop_h=None,
            crop_w=None,
            augmentations=None,
            downsample_gt=True,
            frame_idxs=None,
            num_scales=None,
            color_full_scale=0,
            restrict_dict=None,
            dataset_seed=42,
            load_labeled=True,
            load_unlabeled=False,
            load_onehot=False,
            num_val_samples=None,
            only_sequences_with_segmentation=True,
            load_labels=True,  # rename this to clarify (semantic labels)
            load_sequence=True,
            load_gt_depth=False,
            load_gt_normals=False,
            load_gt_edges=False,
            inverse_depth=False,
            crop_center=False
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = None
        self.label_colors = None
        self.class_map = None
        self.void_classes = None
        self.valid_classes = None
        self.full_res_shape = None
        self.fy = None
        self.fx = None
        self.u0 = None
        self.v0 = None
        self.images_base = None
        self.sequence_base = None
        self.annotations_base = None #? rename this idem
        if augmentations is None:
            augmentations = {}
        self.root = Path(root)
        self.split = split
        self.is_train = split != 'val'
        self.augmentations = augmentations
        self.downsample_gt = downsample_gt
        assert downsample_gt #?!
        self.seed = dataset_seed
        self.restrict_dict = restrict_dict
        self.load_labeled = load_labeled
        self.load_unlabeled = load_unlabeled
        self.load_onehot = load_onehot #? rename to more specific (semantic ohe)
        self.num_val_samples = num_val_samples
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.height = self.img_size[0]
        self.width = self.img_size[1]
        self.num_scales = num_scales
        self.frame_idxs = frame_idxs
        self.only_sequences_with_segmentation = only_sequences_with_segmentation
        self.load_labels = load_labels
        self.load_sequence = load_sequence
        self.load_gt_depth = load_gt_depth
        self.load_gt_normals = load_gt_normals
        self.load_gt_edges = load_gt_edges
        self.inverse_depth = inverse_depth
        self.crop_center = crop_center

        # convert mean from [1-255]-BGR to [0,1]-RGB
        self.mean = np.array(BGR_MEAN)[::-1, None, None] / 255

        if not self.load_sequence:
            self.frame_idxs = [0]
            self.num_scales = 1

        no_crop = split == 'val' and not crop_center
        self.crop_h = self.height if no_crop else crop_h
        self.crop_w = self.width if no_crop else crop_w

        assert self.width >= self.height
        assert self.crop_w >= self.crop_h

        self.enable_color_aug = self.augmentations.get("color_aug", False)
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.setup_transforms(color_full_scale)

        assert os.path.isdir(self.root), self.root

        self.files = self.prepare_filenames()
        self.files = [dict(idx=i, name=x.rstrip(), labeled=True) for i, x in enumerate(self.files)]
        if load_sequence:
            self.filter_seqs()

        if self.split == 'train' and self.restrict_dict is not None:
            self.files = restrict_to_subset(self.files, **self.restrict_dict, seed=self.seed,
                load_labeled=self.load_labeled, load_unlabeled=self.load_unlabeled)

        elif self.split == 'val' and self.num_val_samples is not None:
            self.files = self.files[:self.num_val_samples]

        if not self.files or len(self.files) == 0:
            raise Exception(f'No files for split={self.split} found in {self.images_base}')

        cpprint(f'Dataset init: [{type(self).__name__}]',
                f' loaded as {self.height}x{self.width}, cropped to {self.crop_h}x{self.crop_w}',
                f' split={self.split}) inverse_depth={inverse_depth} => #D={len(self.files)}',
            c='blue')

    def setup_transforms(self, color_full_scale):
        self.resize = {}
        for i in range(self.num_scales):
            s = 2**i
            self.resize[i] = transforms.Resize((self.crop_h // s, self.crop_w // s),
                                               interpolation=Image.ANTIALIAS)
        s = 2**color_full_scale
        self.resize_full = transforms.Resize((self.height // s, self.width // s),
                                             interpolation=Image.ANTIALIAS)
        self.to_tensor = transforms.ToTensor()


    def filter_seqs(self):
        """Filter file list, so that all frame_idxs are available."""
        filtered_files = []
        for i, path in enumerate(tqdm(self.files)):
            valid = all(os.path.isfile(self.get_image_path(i, j)) for j in self.frame_idxs)
            if valid:
                filtered_files.append(path)
        self.files = filtered_files

    def get_color(self, index, offset, do_flip):
        img_path = self.get_image_path(index, offset).replace('/', os.sep)
        img = pil_loader(img_path, self.width, self.height)
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def get_segmentation(self, index, do_flip):
        lbl_path = self.get_segmentation_path(index).replace('/', os.sep)
        if self.downsample_gt:
            lbl = pil_loader(lbl_path, self.width, self.height, is_segmentation=True)
        else:
            lbl = pil_loader(lbl_path, -1, -1, is_segmentation=True)
        if do_flip:
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)

        return lbl

    def get_depth(self, index, do_flip):
        depth_path = self.get_depth_path(index).replace('/', os.sep)
        if self.downsample_gt:
            lbl = pil_loader(depth_path, self.width, self.height, is_depth=True)
        else:
            lbl = pil_loader(depth_path, -1, -1, is_depth=True)
        if do_flip:
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)

        return lbl

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
            Returns a single training item from the dataset as a dictionary.

            Values correspond to torch tensors.
            Keys in the dictionary are either strings or tuples:

                ("color", <frame_id>, <scale>)          for raw colour images,
                ("color_aug", <frame_id>, <scale>)      for augmented colour images,
                ("K", scale) or ("inv_K", scale)        for camera intrinsics.

            <frame_id> is either:
                an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index'.

            <scale> is an integer representing the scale of the image relative to the fullsize image:
                -1      images at native resolution as loaded from disk
                 i      images resized to (self.width // 2**s, self.height // 2**s)
        """
        is_labeled = self.files[index]['labeled']
        inputs = dict(idx=self.files[index]['idx'],
                      filename=self.get_image_path(index),
                      is_labeled=is_labeled)

        do_color_aug = self.is_train and random.random() > 0.5 and self.enable_color_aug
        do_flip = self.is_train and 'random_horizontal_flip' in self.augmentations and \
                  random.random() < self.augmentations['random_horizontal_flip']

        for i in self.frame_idxs:
            inputs[('color', i, -1)] = self.get_color(index, i, do_flip)

        if self.load_labels:
            inputs['lbl'] = self.get_segmentation(index, do_flip)

        if self.load_gt_depth:
            inputs['gt_depth'] = self.get_depth(index, do_flip)

        if self.load_gt_normals:
            inputs['gt_normals'] = self.get_normals(index, do_flip)

        if self.load_gt_edges:
            inputs['gt_edges'] = self.get_edges(index, do_flip)

        inputs = self.crop(inputs, do_flip)

        self.preprocess(inputs, do_color_aug)

        for i in self.frame_idxs:
            del inputs['color', i, -1]

        return inputs

    def crop(self, inputs, do_flip):
        w, h = inputs['color', 0, -1].size
        th, tw = self.crop_h, self.crop_w
        assert h <= w and th <= tw
        if w < tw or h < th:
            raise NotImplementedError

        if self.crop_center: # regardless of the mode {train, valid}
            # x1, y1 = w - tw, h - th # <- this bottom right fixed crop
            x1, y1 = math.floor((w - tw)/2), math.floor((h - th)/2)
        elif self.is_train:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
        else: # eval mode
            x1, y1 = 0, 0

        crop_region = (x1, y1, x1 + tw, y1 + th)

        for i in self.frame_idxs:
            img = inputs[("color", i, -1)]
            # inputs[("color_full", i, -1)] = img
            if w != tw or h != th:
                inputs[("color", i, -1)] = img.crop(crop_region)

        if (w != tw or h != th) and "lbl" in inputs:
            inputs["lbl"] = inputs["lbl"].crop(crop_region)

        if (w != tw or h != th) and "pseudo_depth" in inputs:
            inputs["pseudo_depth"] = inputs["pseudo_depth"].crop(crop_region)

        if (w != tw or h != th) and 'gt_depth' in inputs:
            assert isinstance(inputs['gt_depth'], tuple), 'dataset get_depth() should output depth and mask as a tuple'
            inputs['gt_depth'] = tuple(self.single_crop(x, crop_region) for x in inputs['gt_depth'])

        if (w != tw or h != th) and 'gt_normals' in inputs:
            inputs['gt_normals'] = tf.crop_region(inputs['gt_normals'], crop_region)

        if (w != tw or h != th) and 'gt_edges' in inputs:
            inputs['gt_edges'] = tf.crop_region(inputs['gt_edges'], crop_region)

        # adjusting intrinsics to match each scale in the pyramid
        if True or self.load_sequence:
            for scale in range(self.num_scales):
                if scale != 0:
                    continue
                K = self.get_K(x1, y1, do_flip)

                K[0, :] /= (2 ** scale)
                K[1, :] /= (2 ** scale)

                inv_K = np.linalg.pinv(K)

                inputs[("K", scale)] = torch.from_numpy(K)
                inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        return inputs

    def single_crop(self, x, crop_region):
        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            wmin, hmin, wmax, hmax = crop_region
            x = x[..., hmin:hmax, wmin:wmax]
        else:
            assert False
            x = x.crop(crop_region)
        return x

    def preprocess(self, inputs, do_color_aug): ##! how does this work? n, im, i = k
        """
            Resize colour images to the required scales and augment if required
            We create the color_aug object in advance and apply the same augmentation to all
            images in this item. This ensures that all images input to the pose network receive the
            same augmentation.
        """
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        for k in list(inputs):
            if len(k) != 3:
                continue
            n, im, i = k
            if n == "color":
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if n == "color_full":
                inputs[(n, im, 0)] = self.resize_full(inputs[(n, im, -1)])

        for k in list(inputs):
            f = inputs[k]
            if len(k) != 3:
                continue
            n, im, i = k
            if "color" in n:
                inputs[n, im, i] = self.to_tensor(f)
                if i == 0:
                    inputs[n + '_aug', im, i] = self.to_tensor(color_aug(f))
                    # aug -= self.mean
                    # inputs[n + '_aug', im, i] = aug.float()

        if "lbl" in inputs:
            inputs['lbl'] = self.preprocess_semseg(inputs['lbl'])

        if "gt_depth" in inputs: ##! can do better
            d = self.preprocess_depth(inputs['gt_depth'])
            if isinstance(d, tuple):
                inputs['gt_depth'], inputs['depth_mask'] = d
            else:
                assert False
                inputs['gt_depth'] = d

        if 'gt_normals' in inputs:
            inputs['gt_normals'] = self.preprocess_normals(inputs['gt_normals'])

        if 'gt_edges' in inputs:
            inputs['gt_edges'] = self.preprocess_edges(inputs['gt_edges'])

    def preprocess_semseg(self, semseg):
        lbl = np.asarray(semseg)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        return torch.from_numpy(lbl).long()

    def preprocess_depth(self, depth):
        return self.to_tensor(np.array(depth)).float()

    def get_K(self, u_offset, v_offset, do_flip):
        u0 = self.u0
        v0 = self.v0
        if do_flip:
            u0 = self.full_res_shape[0] - u0
            v0 = self.full_res_shape[1] - v0

        return np.array([[self.fx,        0,  u0 - u_offset,  0],
                         [      0,  self.fy,  v0 - v_offset,  0],
                         [      0,        0,              1,  0],
                         [      0,        0,              0,  1]], dtype=np.float32)

    def prepare_filenames(self):
        raise NotImplementedError

    def get_image_path(self, index, offset=0):
        raise NotImplementedError

    def get_segmentation_path(self, index):
        raise NotImplementedError

    def get_depth_path(self, index):
        raise NotImplementedError

    def segmap2color(self, x):
        raise self.encoder.segmap2color(x)

    def encode_segmap(self, x):
        return self.encoder.encode_segmap(x)
