import math
import torch
from torch import nn

from util.misc import NestedTensor


def position_encoding_image(size, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
    H, W = size
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi

    x_embed = torch.arange(H) # [H]
    y_embed = torch.arange(W) # [W]

    if normalize:
        x_embed = x_embed / H * scale # [H]
        y_embed = y_embed / W * scale # [W]

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32) # [D]
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats) # [D]

    pos_x = x_embed[..., None] / dim_t[None] # [H, D]
    pos_y = y_embed[..., None] / dim_t[None] # [W, D]

    pos_x[..., 0::2] = pos_x[..., 0::2].sin()
    pos_x[..., 1::2] = pos_x[..., 1::2].cos() # [H, D]

    pos_y[..., 0::2] = pos_y[..., 0::2].sin()
    pos_y[..., 1::2] = pos_y[..., 1::2].cos() # [W, D]

    pos = torch.cat([pos_x.unsqueeze(1), pos_y.unsqueeze(0)], dim=2) # [H, W, 2D]
    return pos

