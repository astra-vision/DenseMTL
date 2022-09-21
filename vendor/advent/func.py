import torch
import numpy as np
import torch.nn.functional as F

def ce_loss(y_pred, y_label):
    y_truth = torch.full_like(y_pred, y_label, device=y_pred.device)
    return F.cross_entropy(y_pred, y_truth)


def bce_loss(y_pred, y_label):
    y_truth = torch.ones_like(y_pred, device=y_pred.device)*y_label
    return F.binary_cross_entropy_with_logits(y_pred, y_truth)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    div = 1 if c == 1 else np.log2(c) # quick fix for depth which only as one channel
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / div

def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)