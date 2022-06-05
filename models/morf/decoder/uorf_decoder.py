from models.base_classes import Decoder

import torch
from torch import nn

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
        super().__init__(self, opt)
        self.n_freq = opt.n_freq
        self.locality = opt.locality
        self.locality_ratio = opt.locality_ratio
        self.fixed_locality = opt.fixed_locality
        self.out_ch = opt.out_ch

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
        cam2world = input_decoder['cam2world']
        cam2world_azi = input_decoder['cam2world_azi']
        fg_transform = cam2world[:, 0:1].inverse() if self.opt.fixed_locality else cam2world_azi[:, 0:1].inverse()

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

        return raws, masked_raws, unmasked_raws, masks