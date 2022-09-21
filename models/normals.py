import torch
import torch.nn.functional as F
import numpy as np


def color_sim(diff, alpha=.1):
    """RGB similarity metric, as defined in Yang et al. 2017."""
    return torch.exp(-alpha*diff.norm(dim=-3))

def remove_borders(x):
    """Remove 1 pixel along all 4 borders."""
    x[..., (0,-1), :] = False
    x[..., (0,-1)] = False
    return x

def dilate_mask(m, ksize):
    padding = {3: 1, 5: 2}[ksize]
    m = F.conv2d(1 - m, weight=torch.ones(1, 1, ksize, ksize, device=m.device), padding=padding)
    m[m > 0] = 1
    return (1 - m).bool()

def normalize(x):
    x = x*2 - 1
    x = x / x.norm(dim=-3, keepdim=True) # x.div_()?
    return x

def d2normals(points, guide=None, n_bases=1):
    """
        Bases are defined as:
        . → (w0, h0) - basis vector is defined with [(w1, h1), (w0, h0)]
        ↓
        (w1, h1)

        Up to four bases can be used, which sums up to a total of 8 neighbors:
        | (i)  ↑     | (ii)       | (iii)    ↗    | (iv) ↖
        |      . →   |       ← .  |         .     |        .
        |            |         ↓  |          ↘    |      ↙
    """
    # wrapper lambdas for pts and guide
    w = lambda w, h: guide.roll(-w, -1).roll(-h, -2) - guide
    v = lambda w, h: points.roll(-w, -1).roll(-h, -2) - points

    bases = [[( 1,  0), ( 0, -1)],
             [(-1,  0), ( 0,  1)],
             [( 1,  1), ( 1, -1)],
             [(-1, -1), (-1,  1)]]

    normals = []
    for (w1, h1), (w0, h0) in bases[:n_bases]:
        n = torch.cross(v(w1, h1), v(w0, h0))
        w = color_sim(w(w1, h1))*color_sim(w(w0, h0)) if guide else 1
        normals.append(w*n)

    # stack over estimations, normalize vectors, channel axis
    normals = torch.stack(normals).sum(0)
    normals = F.normalize(normals, dim=-3)

    # flip normals pointing outwards on z-axis.
    # tensor is shaped (b,c,h,w) and c is ordered as RGB: flip B
    normals[:, -1].abs_()

    return normals

def unproject(depth, inv_K):
    h, w = depth.shape[-2:]

    meshgrid = torch.meshgrid(torch.arange(h), torch.arange(w))#, indexing='ij')
    pix_coords = torch.stack(meshgrid).flip(0).unsqueeze(0).flatten(2).float().to(depth.device)
    ones = torch.ones(1, 1, h*w, device=depth.device)
    pix_coords = torch.cat((pix_coords, ones), dim=1)

    # forward
    cam_points = depth.flatten(2) * (inv_K[:, :3, :3] @ pix_coords)
    return cam_points.reshape(-1, 3, h, w).contiguous()

def cosine_loss(x, y):
    assert x.shape == y.shape
    sim = F.cosine_similarity(x, y.detach(), dim=-3).unsqueeze(-3)
    return 1 - sim