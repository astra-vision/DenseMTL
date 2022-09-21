import re

import torch.nn as nn


def get_resnet_num_layers(name):
    match = re.match(r'([a-z]+)([0-9]+)', name, re.I)
    if not match: raise ValueError
    num_layers = int(match.groups()[-1])
    return num_layers

def remove_resnet_tail_(model):
    model.encoder.avgpool = nn.Identity()
    model.encoder.fc = nn.Identity()