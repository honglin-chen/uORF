import torch
from models.base_classes import Renderer
from .projection import Projection

class UorfRenderer(Renderer):
    def __init__(self, opt):
        super().__init__(self, opt):
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
        self.projection = Projection(device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane,
                                     render_size=render_size)
        frustum_size_fine = [self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp]
        self.projection_fine = Projection(device=self.device, nss_scale=opt.nss_scale,
                                          frustum_size=frustum_size_fine, near=opt.near_plane, far=opt.far_plane,
                                          render_size=render_size)

    def render(self, input_renderer):
        encoder = input_renderer['encoder_obj']
        decoder = input_renderer['decoder_obj']
        cam2world = input_renderer['cam2world']
        cam2world_azi = input_renderer['cam2world_azi']
        x = input_renderer['input_img']
        mask = input_renderer['input_mask']

        B, NV, C, H, W = x.shape
        K = mask.shape[1]  # if no gt_seg, run mask = attn
        D = self.opt.n_samp

        dev = x[:, 0:1].device

        if self.opt.stage == 'coarse':
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            x = F.interpolate(x.flatten(0, 1), size=self.opt.supervision_size, mode='bilinear', align_corners=False)
            x = x.reshape(B, NV, 3, self.opt.supervision_size, self.opt.supervision_size)
            self.z_vals, self.ray_dir = z_vals, ray_dir
        else:
            W, H, D = self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp
            start_range = self.opt.frustum_size_fine - self.opt.render_size
            rs = self.opt.render_size
            frus_nss_coor, z_vals, ray_dir = self.projection_fine.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor.view([NV, D, H, W, 3]), z_vals.view([NV, H, W, D]), ray_dir.view([NV, H, W, 3])
            H_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            W_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            frus_nss_coor_, z_vals_, ray_dir_ = frus_nss_coor[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], z_vals[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], ray_dir[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor_.flatten(0, 3), z_vals_.flatten(0, 2), ray_dir_.flatten(0, 2)
            x = x[:, :, :, H_idx:H_idx + rs, W_idx:W_idx + rs]
            self.z_vals, self.ray_dir = z_vals, ray_dir
            W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp

        sampling_coor_fg = frus_nss_coor[:, None, ...].expand(-1, K - 1, -1, -1)  # (K-1)xPx3
        sampling_coor_bg = frus_nss_coor  # BxPx3

        uv = None
        xyz = None
        coor_feature = {'uv': uv,
                         'xyz': xyz}
        output_encoder = encoder.get_feature(coor_feature)

        input_decoder = {'coor_bg': sampling_coor_bg,
                         'coor_fg': sampling_coor_fg,
                         'slot_feat': output_encoder['slot_feat'],
                         'fg_transform': cam2world[:, 0:1].inverse() if self.opt.fixed_locality else cam2world_azi[:, 0:1].inverse()
                         }
        output_decoder = decoder(input_decoder)
        raws = output_decoder['raws'] # Bx(NxDxHxW)x4
        masked_raws = output_decoder['weighted_raws'] # BxKx(NxDxHxW)x4
        unmasked_raws = output_decoder['unweighted_raws'] # BxKx(NxDxHxW)x4
        masks = output_decoder['masks'] # BxKx(NxDxHxW)x1

        raws = raws.view([B, NV, D, H, W, 4]).permute([0, 1, 3, 4, 2, 5]).flatten(start_dim=1,
                                                                                 end_dim=3)  # Bx(NxHxW)xDx4
        masked_raws = masked_raws.view([B, K, NV, D, H, W, 4])
        unmasked_raws = unmasked_raws.view([B, K, NV, D, H, W, 4])
        rgb_map, _, _ = raw2outputs(raws, z_vals, ray_dir)
        # Bx(NxHxW)x3, Bx(NxHxW)

        rendered = rgb_map.view(B, N, H, W, 3).permute([0, 1, 4, 2, 3])  # BxNx3xHxW
        x_recon = rendered * 2 - 1

        output_renderer = {'x_recon': x_recon,
                           'weighted_raws': masked_raws,
                           'unweighted_raws': unmasked_raws}

        return output_renderer



