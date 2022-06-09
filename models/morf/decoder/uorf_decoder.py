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
        input_dim = 33 # this is coor dimension
        n_layers = 3
        z_dim = 64  # this is not self.opt.z_dim
        self.C = 0
        self.C += 64 if self.opt.use_pixel_feat else 0 # make this configurable
        self.C += 64 if self.opt.use_slot_feat else 0 # make this configurable
        self.C += 3 if self.opt.use_ray_dir_world else 0
        input_dim += self.C

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
        pixel_feat = input_decoder['pixel_feat']
        fg_transform = input_decoder['fg_transform']

        B, P, _ = sampling_coor_bg.shape
        K = self.opt.num_slots

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

        input_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # BxPx60, 60 means increased-freq feat dim
        sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=1, end_dim=2)  # Bx((K-1)xP)x3
        input_fg = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # Bx((K-1)xP)x60

        if self.opt.use_slot_feat:
            z_bg = z_slots[:, 0:1, :]  # Bx1xC
            z_fg = z_slots[:, 1:, :]  # Bx(K-1)xC
            z_fg_ex = z_fg[:, :, None, :].expand(-1, -1, P, -1).flatten(start_dim=1, end_dim=2)  # Bx((K-1)xP)xC

            input_bg = torch.cat([input_bg, z_bg.expand(-1, P, -1)], dim=-1)  # BxPx(60+C)
            input_fg = torch.cat([input_fg, z_fg_ex], dim=-1)  # Bx((K-1)xP)x(60+C)

        if self.opt.use_pixel_feat:
            pixel_feat_bg = pixel_feat[:, 0, ...] # BxPxC
            pixel_feat_fg = pixel_feat[:, 1:, ...].flatten(1, 2) # Bx((K-1)xP)xC

            input_bg = torch.cat([pixel_feat_bg, input_bg], dim=-1)
            input_fg = torch.cat([pixel_feat_fg, input_fg], dim=-1)

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

        output_decoder = {'all_raws': all_raws}

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