from models.base_classes import Decoder

import torch
from torch import nn
import torch.nn.functional as F

class UorfDecoder(Decoder):
    def __init__(self, opt):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__(opt)
        self.n_freq = opt.n_freq
        self.locality = False
        self.locality_ratio = 4/7
        self.fixed_locality = opt.fixed_locality
        self.out_ch = 4
        input_dim = 33 + 64
        z_dim = 64
        n_layers = 3

        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim + input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        self.f_before = nn.Sequential(*before_skip)
        self.f_after = nn.Sequential(*after_skip)
        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim // 4, 3))
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim + input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        after_skip.append(nn.Linear(z_dim, self.out_ch))
        self.b_before = nn.Sequential(*before_skip)
        self.b_after = nn.Sequential(*after_skip)

    def forward(self, input_decoder):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: BxPx3, B=batch size, P = #points, typically P = NxDxHxW
            sampling_coor_fg: Bx(K-1)xPx3
            z_slots: BxKxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is Bx1x4x4 matrix nss2cam0, otherwise it is Bx1x3x3 azimuth rotation of nss2cam0
        """

        sampling_coor_bg = input_decoder['coor_bg']
        sampling_coor_fg = input_decoder['coor_fg']
        z_slots = input_decoder['slot_feat']
        fg_transform = input_decoder['fg_transform']


        B, K, C = z_slots.shape
        P = sampling_coor_bg.shape[1]

        if self.fixed_locality:
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # Bx(K-1)xP
            sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, :, 0:1])],
                                         dim=-1)  # Bx(K-1)xPx4
            sampling_coor_fg = torch.matmul(fg_transform[:, None, ...],
                                            sampling_coor_fg[..., None])  # Bx(K-1)xPx4x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :, 3]  # Bx(K-1)xPx3
        else:
            sampling_coor_fg = torch.matmul(fg_transform[:, None, ...],
                                            sampling_coor_fg[..., None])  # Bx(K-1)xPx3x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # Bx(K-1)xPx3
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # Bx(K-1)xP

        z_bg = z_slots[:, 0:1, :]  # Bx1xC
        z_fg = z_slots[:, 1:, :]  # Bx(K-1)xC
        query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # BxPx60, 60 means increased-freq feat dim
        input_bg = torch.cat([query_bg, z_bg.expand(-1, P, -1)], dim=-1)  # BxPx(60+C)

        sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=1, end_dim=2)  # Bx((K-1)xP)x3
        query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # Bx((K-1)xP)x60
        z_fg_ex = z_fg[:, :, None, :].expand(-1, -1, P, -1).flatten(start_dim=1, end_dim=2)  # Bx((K-1)xP)xC
        input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=-1)  # Bx((K-1)xP)x(60+C)

        tmp = self.b_before(input_bg)
        bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=-1)).view([B, 1, P, self.out_ch])  # BxPx5 -> Bx1xPx5
        tmp = self.f_before(input_fg)
        tmp = self.f_after(torch.cat([input_fg, tmp], dim=-1))  # Bx((K-1)xP)x64
        latent_fg = self.f_after_latent(tmp)  # Bx((K-1)xP)x64
        fg_raw_rgb = self.f_color(latent_fg).view([B, K - 1, P, 3])  # Bx((K-1)xP)x3 -> Bx(K-1)xPx3
        fg_raw_shape = self.f_after_shape(tmp).view([B, K - 1, P])  # B((K-1)xP)x1 -> Bx(K-1)xP, density
        if self.locality:
            fg_raw_shape[outsider_idx] *= 0
        fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # Bx(K-1)xPx4

        all_raws = torch.cat([bg_raws, fg_raws], dim=1)  # BxKxPx4
        raw_masks = F.relu(all_raws[:, :, :, -1:], True)  # BxKxPx1
        masks = raw_masks / (raw_masks.sum(dim=1, keepdim=True) + 1e-5)  # BxKxPx1
        raw_rgb = (all_raws[:, :, :, :3].tanh() + 1) / 2
        raw_sigma = raw_masks

        unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=-1)  # BxKxPx4
        masked_raws = unmasked_raws * masks
        raws = masked_raws.sum(dim=1)

        output_decoder = {'raws': raws,
                          'weighted_raws': masked_raws,
                          'unweighted_raws': unmasked_raws,
                          'occupancies': masks}

        return output_decoder


def sin_emb(x, n_freq=5, keep_ori=True):
    """
    create sin embedding for 3d coordinates
    input:
        x: Px3
        n_freq: number of raised frequency
    """
    embedded = []
    if keep_ori:
        embedded.append(x)
    emb_fns = [torch.sin, torch.cos]
    freqs = 2. ** torch.linspace(0., n_freq - 1, steps=n_freq)
    for freq in freqs:
        for emb_fn in emb_fns:
            embedded.append(emb_fn(freq * x))
    embedded_ = torch.cat(embedded, dim=-1)
    return embedded_

def raw2outputs(raw, z_vals, rays_d, render_mask=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [bsz, num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [bsz, num_rays, num_samples along ray]. Integration time.
        rays_d: [bsz, num_rays, 3]. Direction of each ray in cam coor.
    Returns:
        rgb_map: [bsz, num_rays, 3]. Estimated RGB color of a ray.
        depth_map: [bsz, num_rays]. Estimated distance to object.
    """
    assert len(raw.shape) == 4 and len(z_vals.shape) == 3 and len(rays_d.shape) == 3
    assert raw.shape[0] == z_vals.shape[0] == rays_d.shape[0]

    raw2alpha = lambda x, y: 1. - torch.exp(-x * y)
    device = raw.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e-2], device=device).expand(dists[..., :1].shape)], -1)  # [B, N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = raw[..., :3]

    alpha = raw2alpha(raw[..., 3], dists)  # [B, N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:,:, :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [B, N_rays, 3]

    weights_norm = weights.detach() + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True)
    depth_map = torch.sum(weights_norm * z_vals, -1)

    if render_mask:
        density = raw[..., 3]  # [B, N_rays, N_samples]
        mask_map = torch.sum(weights * density, dim=-1)  # [B, N_rays,]
        return rgb_map, depth_map, weights_norm, mask_map

    return rgb_map, depth_map, weights_norm