import math
import torch
from torch import nn
import matplotlib.pyplot as plt

def position_encoding_image(size, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
    H, W = size
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi

    x_embed = torch.arange(H) # [H]
    y_embed = torch.arange(W) # [W]

    if normalize: # normalize to [0, 2pi]
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

    pos_x = pos_x.unsqueeze(1).expand(-1, W, -1) # [H, W, D]
    pos_y = pos_y.unsqueeze(0).expand(H, -1, -1)  # [H, W, D]
    pos = torch.cat([pos_x, pos_y], dim=2) # [H, W, 2D]
    pos = pos.flatten(0, 1).unsqueeze(0) # [HxW, 2D]

    return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, size, num_pos_feats=32):
        super().__init__()
        self.size = size
        self.row_embed = nn.Embedding(size[0], num_pos_feats)
        self.col_embed = nn.Embedding(size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self):
        H, W = self.size
        pos = torch.cat([
            self.row_embed.weight.unsqueeze(1).repeat(1, W, 1),
            self.col_embed.weight.unsqueeze(0).repeat(H, 1, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(0)
        return pos