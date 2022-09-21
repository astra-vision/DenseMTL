import torch

from training.setup import Setup
from utils.logs import cpprint


class Monodepth(Setup):
    def __init__(self, pose_net, imnet_encoder):
        super().__init__()
        self.pose_net = pose_net
        self.imnet_encoder = imnet_encoder

    def forward(self, x, out):
        rgb = x['color_aug', 0, 0]

        if self.imnet_encoder:
            out['encoder_features'] = out['features'][-1]
            with torch.no_grad():
                out['imnet_features'] = self.imnet_encoder(rgb)[-1]

        if self.pose_net:
            out.update(self.pose_net(x))

        return out