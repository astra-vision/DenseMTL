import torch
import numpy as np

from .labels import Cityscapes, VKitti2, Synthia, NYUDv2
from .mappings import Mappings

class Encoder:
    ignore_index = 250

    def __init__(self, n_classes, klasses, colors, decode=None):
        subset = f'cls{n_classes:02d}'
        if subset not in klasses:
            raise NotImplementedError

        # dataset-index -> train-id mapping
        self.map = klasses[subset]
        assert len(set(self.map.values())) == n_classes

        # train-id -> color mapping
        self.colors = {i: np.array(colors[i]) / 255.0 for i in self.map}

        # for datasets which have GTs saved as RGB maps, we need the mapping containing
        # all colors to be able to convert from RGB values to dataset indices.
        self.decode = decode

    def segmap2color(self, dense):
        rgb = np.zeros(dense.shape + (3,))
        for v in self.map.keys():
            rgb[dense == self.map[v]] = self.colors[v]
        return torch.tensor(rgb)

    def encode_segmap(self, x):
        """
            Will either call _rgb2lbl or _ind2lbl, depending if the input is dense or rgb (some
            datasets provide the semseg GT as a RGB map only...
        """
        if x.shape[-1] == 3: # RGB
            return self._rgb2lbl(x)
        return self._ind2lbl(x)

    def _rgb2lbl(self, rgb):
        """Encodes dataset indices to train labels, used when the GT is rgb format."""
        rgb_f = rgb.reshape(-1, 3)
        d = np.full((rgb_f[:,0].size,), Encoder.ignore_index)
        for v in self.map.keys():
            d[(rgb_f == np.array(self.decode[v])).all(1)] = self.map[v]
        return d.reshape(rgb.shape[:-1])

    def _ind2lbl(self, dense):
        """Encodes dataset indices to train labels, used when the GT is dense format."""
        d = np.full_like(dense, Encoder.ignore_index)
        for v in self.map.keys():
            d[dense == v] = self.map[v]
        return d


VKCS_N_CLASSES = 8

class CityscapesEncoder(Encoder):
    def __init__(self, n_classes):
        *_, cs_colors, vkcs_colors = zip(*Cityscapes.labels)
        colors = vkcs_colors if n_classes == VKCS_N_CLASSES else cs_colors
        super().__init__(n_classes, klasses=Mappings.Cityscapes, colors=colors)

class KITTI360Encoder(Encoder):
    def __init__(self, n_classes):
        *_, cs_colors, _ = zip(*Cityscapes.labels)
        super().__init__(n_classes, klasses=Mappings.Cityscapes, colors=cs_colors)


class VKitti2Encoder(Encoder):
    def __init__(self, n_classes):
        *_, vk_colors, vkcs_colors = zip(*VKitti2.labels)
        colors = vkcs_colors if n_classes == VKCS_N_CLASSES else vk_colors
        super().__init__(n_classes, klasses=Mappings.VKitti2, colors=colors, decode=vk_colors)

class SynthiaEncoder(Encoder):
    def __init__(self, n_classes):
        _, sy_colors = zip(*Synthia.labels)
        super().__init__(n_classes, klasses=Mappings.Synthia, colors=sy_colors)

class NYUDv2Encoder(Encoder):
    def __init__(self, n_classes):
        _, rand_colors = zip(*NYUDv2.labels)
        super().__init__(n_classes, klasses={'cls40':{k:k for k in range(40)}}, colors=rand_colors)
