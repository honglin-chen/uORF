import torch
from torch import nn
import torch.nn.functional as F
from models.base_classes import Renderer
from .projection import Projection

class UorfRenderer(Renderer):
    def __init__(self, opt):
        super().__init__(opt)
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.projection = Projection(device=self.device, nss_scale=opt.nss_scale, focal_ratio=opt.focal_ratio,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane,
                                     render_size=render_size)
        frustum_size_fine = [self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp]
        self.projection_fine = Projection(device=self.device, nss_scale=opt.nss_scale, focal_ratio=opt.focal_ratio,
                                          frustum_size=frustum_size_fine, near=opt.near_plane, far=opt.far_plane,
                                          render_size=render_size)
        self.register_buffer('frustum_size', torch.tensor([i * 1.0 for i in self.projection.frustum_size]).cuda())

    def render(self, input_renderer):
        encoder = input_renderer['encoder_obj']
        decoder = input_renderer['decoder_obj']
        cam2world = input_renderer['cam2world']
        cam2world_azi = input_renderer['cam2world_azi']
        x = input_renderer['input_img']
        mask = input_renderer['input_mask']

        B, NV, C, H, W = x.shape
        K = self.opt.num_slots  # if no gt_seg, run mask = attn
        D = self.opt.n_samp
        self.B, self.NV, self.K, self.C, self.D, self.H, self.W = B, NV, K, C, D, H, W

        dev = self.device # this is different from x[:, 0:1].device. I do not know why no error (or why it is needed?)

        if self.opt.stage == 'coarse':
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            x = F.interpolate(x.flatten(0, 1), size=self.opt.supervision_size, mode='bilinear', align_corners=False)
            x = x.reshape(B, NV, 3, self.opt.supervision_size, self.opt.supervision_size)
            self.z_vals, self.ray_dir = z_vals, ray_dir
            W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp
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

        ray_dir_world = self.get_ray_dir_world(ray_dir.view([B*N, H, W, 3])[0].unsqueeze(0)) if self.opt.use_ray_dir_world else None
        uv = self.get_pixel_coor(cam2world, frus_nss_coor) if self.opt.use_pixel_feat else None
        xyz = self.get_voxel_coor() if self.opt.use_voxel_feat else None

        coor_feature = {'uv': uv,
                         'xyz': xyz}
        output_encoder = encoder.get_feature(coor_feature)

        input_decoder = {'coor_bg': sampling_coor_bg,
                         'coor_fg': sampling_coor_fg,
                         'slot_feat': output_encoder['slot_feat'],
                         'pixel_feat': output_encoder['pixel_feat'],
                         'voxel_feat': output_encoder['voxel_feat'],
                         'fg_transform': cam2world[:, 0:1].inverse() if self.opt.fixed_locality else cam2world_azi[:, 0:1].inverse(),
                         'ray_dir': ray_dir_world
                         }
        output_decoder = decoder(input_decoder)

        raws = output_decoder['raws'] # Bx(NxDxHxW)x4
        masked_raws = output_decoder['weighted_raws'] # BxKx(NxDxHxW)x4
        unmasked_raws = output_decoder['unweighted_raws'] # BxKx(NxDxHxW)x4
        masks = output_decoder['occupancies'] # BxKx(NxDxHxW)x1

        raws = raws.view([B, NV, D, H, W, 4]).permute([0, 1, 3, 4, 2, 5]).flatten(1, 3)  # Bx(NxHxW)xDx4
        masked_raws = masked_raws.view([B, K, NV, D, H, W, 4])
        unmasked_raws = unmasked_raws.view([B, K, NV, D, H, W, 4])
        rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir) # Bx(NxHxW)x3, Bx(NxHxW)
        rendered = rgb_map.view(B, NV, H, W, 3).permute([0, 1, 4, 2, 3])  # BxNx3xHxW
        x_recon = rendered * 2 - 1
        depth_map = depth_map.view(B, NV, H, W, 1).permute([0, 1, 4, 2, 3])

        output_renderer = {'x_recon': x_recon,
                           'weighted_raws': masked_raws,
                           'unweighted_raws': unmasked_raws,
                           'depth_map': depth_map,
                           'occupancies': masks,
                           'occl_silhouettes': None,
                           'unoccl_silhouettes': None,
                           'x_supervision': x}
        return output_renderer

    def get_pixel_coor(self, cam2world, frus_nss_coor):
        self.cam2spixel = self.projection.cam2spixel
        self.world2nss = self.projection.world2nss
        frustum_size = self.frustum_size

        # construct uv in the first image coordinates
        '''
        https://pytorch.org/docs/master/generated/torch.matmul.html#torch.matmul
        Rule of torch.matmul:
        (jx1xnxm), (kxmxp) --> (jxkxnxp)
        '''
        assert cam2world.shape[0] == 1, f'batch size larger than 1 in each gpu is not implemented yet'
        cam02world = cam2world[0][0:1]  # 1x4x4
        world2cam0 = cam02world.inverse()  # 1x4x4
        nss2world = self.world2nss.inverse()  # 1x4x4
        # print('frus_nss_coor.shape', frus_nss_coor.shape)
        # print('world2cam0.shape', world2cam0.shape)
        frus_nss_coor = frus_nss_coor.view(-1, 3)
        frus_nss_coor = torch.cat([frus_nss_coor, torch.ones_like(frus_nss_coor[:, 0:1])], dim=-1)  # Px4
        frus_world_coor = torch.matmul(nss2world[None, ...], frus_nss_coor[None, ..., None])  # 1xPx4x1
        frus_cam0_coor = torch.matmul(world2cam0[None, ...],
                                      frus_world_coor)  # 1x1x4x4, 1x(BxNxDxHxW)x4x1 -> 1x(BxNxDxHxW)x4x1
        pixel_cam0_coor = torch.matmul(self.cam2spixel[None, ...], frus_cam0_coor)  # 1x1x4x4, 1x(BxNxDxHxW)x4x1
        pixel_cam0_coor = pixel_cam0_coor.squeeze(-1)  # 1x(BxNxDxHxW)x4
        # print(pixel_cam0_coor[..., 2].max(), pixel_cam0_coor[..., 2].min(), pixel_cam0_coor[..., 2].mean(), pixel_cam0_coor[..., 2].std(), 'pixel_cam0_coor statistics')
        uv = pixel_cam0_coor[:, :, 0:2] / pixel_cam0_coor[:, :, 2].unsqueeze(-1)  # 1x(BxNxDxHxW)x2
        uv = (uv + 0.) / frustum_size[0:2][None, None, :] * 2 - 1
        return uv

    def get_voxel_coor(self):
        raise NotImplementedError

    def get_ray_dir_world(self, ray_dir):
        raise NotImplementedError

        ray_dir_input = ray_dir.view([B*N, H, W, 3])[0].unsqueeze(0)  # 1xHxWx3
        ray_dir_input /= torch.norm(ray_dir_input, dim=-1).unsqueeze(-1)  # 1xHxWx3
        ray_dir_input = ray_dir_input.squeeze(0).view([H * W, 3]).unsqueeze(-1)  # (HxW)x3x1
        cam2world_raydir = self.cam2world[:, :3, :3].unsqueeze(1)  # Nx4x4 -> Nx1x3x3
        ray_dir_input = torch.matmul(cam2world_raydir, ray_dir_input)  # (j x 1 x n x m, k x m x p)-->(j x k x n x p)
        # Nx1x3x3, (HxW)x3x1 --> Nx(HxW)x3x1
        ray_dir_input = ray_dir_input.view([B*N, H, W, 3]).unsqueeze(1).expand(-1, D, -1, -1, -1)
        ray_dir_input = ray_dir_input.flatten(0, 3)
        return ray_dir_input

    def compute_visual(self, raws):
        N, D, H, W = self.opt.n_img_each_scene, self.opt.n_samp, self.opt.supervision_size, self.opt.supervision_size
        z_vals, ray_dir = self.z_vals[0:1], self.ray_dir[0:1]
        raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
        rgb_map, depth_map, _ = raw2outputs(raws.unsqueeze(0), z_vals, ray_dir)
        rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
        x_recon = rendered * 2 - 1

        return x_recon


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


