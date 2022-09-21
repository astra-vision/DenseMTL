# https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth

import numpy
import torch
import torch.nn.functional as F


def berhu(input, target, mask, apply_log=False, threshold=.2):
    if apply_log:
        input, target = torch.log(1 + input), torch.log(1 + target)

    absdiff = torch.abs(target - input) * mask
    C = threshold * torch.max(absdiff).item()
    loss = torch.mean(torch.where(absdiff <= C,
                                  absdiff,
                                  (absdiff * absdiff + C * C) / (2 * C)))
    return loss

def pixel_wise_entropy(logits, normalize=False):
    assert logits.dim() == 4
    p = F.softmax(logits, dim=1)
    N, C, H, W = p.shape
    pw_entropy = -torch.sum(p * torch.log2(p + 1e-30), dim=1) / numpy.log2(C)
    if normalize:
        pw_entropy = (pw_entropy - torch.min(pw_entropy)) / (torch.max(pw_entropy) - torch.min(pw_entropy))
    return pw_entropy