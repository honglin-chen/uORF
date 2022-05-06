import math
from .op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd
from .networks import get_norm_layer
from .resnetfc import ResnetFC

import os


class Encoder(nn.Module):
    def __init__(self, input_nc=3, z_dim=64, bottom=False):

        super().__init__()

        self.bottom = bottom

        if self.bottom:
            self.enc_down_0 = nn.Sequential(nn.Conv2d(input_nc + 4, z_dim, 3, stride=1, padding=1),
                                            nn.ReLU(True))
        self.enc_down_1 = nn.Sequential(nn.Conv2d(z_dim if bottom else input_nc+4, z_dim, 3, stride=2 if bottom else 1, padding=1),
                                        nn.ReLU(True))
        self.enc_down_2 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_down_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_up_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_2 = nn.Sequential(nn.Conv2d(z_dim*2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_1 = nn.Sequential(nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True))

    def forward(self, x, masks=None):
        """
        input:
            x: input image, Bx3xHxW
        output:
            feature_map: BxCxHxW
        """
        W, H = x.shape[3], x.shape[2]
        X = torch.linspace(-1, 1, W)
        Y = torch.linspace(-1, 1, H)
        y1_m, x1_m = torch.meshgrid([Y, X])
        x2_m, y2_m = 2 - x1_m, 2 - y1_m  # Normalized distance in the four direction
        pixel_emb = torch.stack([x1_m, x2_m, y1_m, y2_m]).to(x.device).unsqueeze(0)  # 1x4xHxW
        x_ = torch.cat([x, pixel_emb], dim=1)

        if self.bottom:
            x_down_0 = self.enc_down_0(x_)
            x_down_1 = self.enc_down_1(x_down_0)
        else:
            x_down_1 = self.enc_down_1(x_)
        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_up_3 = self.enc_up_3(x_down_3)
        x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
        feature_map = self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))  # BxCxHxW
        return feature_map

"""
Implements image encoders
"""
import torchvision
# from model.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler


class PixelEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4, # 4 in pixelnerf
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        reduce_latent_size=True,
        mask_image=False,
        mask_image_feature=False,
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        self.reduce_latent_size = reduce_latent_size
        norm_layer = get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models, backbone)(
                pretrained=pretrained, norm_layer=norm_layer
            )
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]
            if self.reduce_latent_size:
                self.mlp = nn.Linear(self.latent_size, 64)

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        # self.latent (B, L, H, W)

        self.mask_image = mask_image
        self.mask_image_feature = mask_image_feature
        if self.mask_image or self.mask_image_feature:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.n_mask = 5 # TODO: make it configurable

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """

        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent,
                uv,
                # align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x, masks=None):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        H, W = x.shape[2], x.shape[3]

        if self.mask_image or self.mask_image_feature:
            K = masks.shape[0]
            masks = masks.view(K, H, W).unsqueeze(1)

        if self.mask_image:
            x = x.expand(K, -1, -1, -1).clone() # KxCxHxW
            x *= masks

        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                # mode=self.index_interp,
                # mode="bilinear" if self.feature_scale > 1.0 else "area",
                # align_corners=True if self.feature_scale > 1.0 else None,
                # recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        # print(x.shape, 'x.shape') # 5x3x64x64

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.latents = latents
            # align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    # align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        if self.reduce_latent_size:
            self.latent = self.mlp(self.latent.transpose(3, 1)).transpose(3, 1)

        if self.mask_image_feature:
            masks = self.downsample(masks)
            if self.latent.shape[0]==1:
                self.latent = self.latent.expand(K, -1, -1, -1).clone()
            self.latent *= masks

        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )

class Decoder(nn.Module):
    def __init__(self, n_freq=5, input_dim=33+64, pixel_dim=None, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7,
                 fixed_locality=False, no_concatenate=False, bg_no_pixel=False, use_ray_dir=False, small_latent=False, decoder_type=None,
                 restrict_world=False, reduce_color_decoder=False, density_as_color_input=False, mask_as_decoder_input=False,
                 ray_after_density=False, multiply_mask_pixelfeat=False, same_bg_fg_decoder=False, without_slot_feature=False,
                 color_after_density=False, weight_pixel_slot_mask=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        pixel_dim: pixel encoder latent dim (0 or None to disable)
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.bg_no_pixel = bg_no_pixel
        self.use_ray_dir = use_ray_dir
        self.out_ch = 4
        self.small_latent = small_latent
        self.no_concatenate = no_concatenate
        self.pixel_dim = pixel_dim
        self.restrict_world = restrict_world
        self.reduce_color_decoder = reduce_color_decoder
        self.density_as_color_input = density_as_color_input
        self.mask_as_decoder_input = mask_as_decoder_input
        self.ray_after_density = ray_after_density
        self.multiply_mask_pixelfeat = multiply_mask_pixelfeat
        self.same_bg_fg_decoder = same_bg_fg_decoder
        self.without_slot_feature = without_slot_feature
        self.color_after_density = color_after_density
        self.weight_pixel_slot_mask = weight_pixel_slot_mask
        print(self.color_after_density, 'put pixel feature after density decoder')
        print(self.without_slot_feature, 'self.without_slot_feature')
        if color_after_density:
            input_dim -= pixel_dim
            pixel_dim_color = pixel_dim
        else:
            pixel_dim_color = 0
        if without_slot_feature:
            input_dim -= z_dim
        if ray_after_density:
            assert use_ray_dir
            ray_dir_dim = 3
        else:
            ray_dir_dim = 0
        if multiply_mask_pixelfeat:
            assert mask_as_decoder_input
        if density_as_color_input:
            input_dim += 1
        if mask_as_decoder_input:
            input_dim += 1 if not self.multiply_mask_pixelfeat else pixel_dim
        if use_ray_dir:
            input_dim += 3 if not self.ray_after_density else 0
        if decoder_type=='color':
            input_dim += 1
            if not mask_as_decoder_input:
                input_dim += 1 if not self.multiply_mask_pixelfeat else pixel_dim
        if pixel_dim is not None or 0:
            if self.no_concatenate:
                pass
            else:
                input_dim += pixel_dim

        if small_latent:
            latent_dim = z_dim // 2
        else:
            latent_dim = z_dim
        # print(input_dim, "input_dim")
        # print(latent_dim, 'latent_dim')
        before_skip = [nn.Linear(input_dim, latent_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(latent_dim+input_dim, latent_dim), nn.ReLU(True)]
        if self.reduce_color_decoder:
            after_skip = []
        for i in range(n_layers-1):
            before_skip.append(nn.Linear(latent_dim, latent_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(latent_dim, latent_dim))
            after_skip.append(nn.ReLU(True))
        self.f_before = nn.Sequential(*before_skip)
        self.f_after = nn.Sequential(*after_skip)
        self.f_after_latent = nn.Linear(latent_dim, latent_dim)
        self.f_after_shape = nn.Linear(latent_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(latent_dim + ray_dir_dim + pixel_dim_color, latent_dim//4),
                                     nn.ReLU(True),
                                     nn.Linear(latent_dim//4, 3))
        if pixel_dim is not None and bg_no_pixel and not no_concatenate:
            input_dim += -pixel_dim
        before_skip = [nn.Linear(input_dim, latent_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(latent_dim + input_dim, latent_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(latent_dim, latent_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(latent_dim, latent_dim))
            after_skip.append(nn.ReLU(True))
        if self.reduce_color_decoder:
            after_skip = []
        if self.ray_after_density or self.same_bg_fg_decoder:
            pass
        else:
            after_skip.append(nn.Linear(latent_dim, self.out_ch))
        self.b_before = nn.Sequential(*before_skip)
        self.b_after = nn.Sequential(*after_skip)
        if self.ray_after_density or self.same_bg_fg_decoder:
            assert not self.reduce_color_decoder
            self.b_after_latent = nn.Linear(latent_dim, latent_dim)
            self.b_after_shape = nn.Linear(latent_dim, self.out_ch - 3)
            self.b_color = nn.Sequential(nn.Linear(latent_dim + ray_dir_dim + pixel_dim_color, latent_dim // 4),
                                         nn.ReLU(True),
                                         nn.Linear(latent_dim // 4, 3))

        if self.no_concatenate:
            self.change_dim = nn.Linear(pixel_dim, z_dim)
            self.pixel_norm = nn.LayerNorm(z_dim, elementwise_affine=True)
            self.object_norm = nn.LayerNorm(z_dim, elementwise_affine=True)
            self.norm2feat = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.ReLU(inplace=True),
                nn.Linear(z_dim, z_dim)
            )
        # if pixel_dim:
        #     self.substitute_pixel_feat = torch.nn.Parameter(torch.zeros(pixel_dim))

    def forward(self, sampling_coor_bg, sampling_coor_fg, z_slots, fg_transform, pixel_feat=None, no_concatenate=None,
                ray_dir_input=None, transmittance_samples=None, raw_masks_density=None, decoder_type=None, silhouettes=None):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        K, C = z_slots.shape
        P = sampling_coor_bg.shape[0]
        no_concatenate = self.no_concatenate

        # if self.pixel_dim and pixel_feat is None:
        #     pixel_feat = self.substitute_pixel_feat[None, None, ...].expand(K, P, -1)

        # if transmittance_samples is not None:
        #     pixel_feat *= transmittance_samples
        #     pixel_feat += self.substitute_pixel_feat[None, None, ...].expand(K, P, -1) * (1. - transmittance_samples) # this option was previous behavior, I thought that it is quite hard coding

        if self.fixed_locality:
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
            sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # (K-1)xPx4
            sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
        else:
            sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
            z_out_idx_fg = torch.any(sampling_coor_fg[..., 2:3] < 0.1, dim=-1)
            z_out_idx_bg = torch.any(sampling_coor_bg[None, ..., 2:3] < 0.1, dim=-1)
            if self.restrict_world:
                nss_scale = 7
                outside_world_x = torch.logical_not(torch.any(-5.0/nss_scale < sampling_coor_bg[..., 0:1], dim=-1).unsqueeze(-1) & torch.any(sampling_coor_bg[..., 0:1] < 5.0/nss_scale, dim=-1).unsqueeze(-1))
                outside_world_y = torch.logical_not(torch.any(-5.0/nss_scale < sampling_coor_bg[..., 1:2], dim=-1).unsqueeze(-1) & torch.any(sampling_coor_bg[..., 1:2] < 5.0/nss_scale, dim=-1).unsqueeze(-1))
                outside_world_z = torch.logical_not(torch.any(-0./nss_scale < sampling_coor_bg[..., 2:3], dim=-1).unsqueeze(-1) & torch.any(sampling_coor_bg[..., 2:3] < 6.0/nss_scale, dim=-1).unsqueeze(-1)) # 0, 6
                outside_world_idx = outside_world_x | outside_world_y | outside_world_z
                outside_world_idx = outside_world_idx.unsqueeze(0)

        query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
        sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
        query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
        if self.use_ray_dir:
            if not self.ray_after_density:
                query_bg = torch.cat([query_bg, ray_dir_input], dim=1)
                query_fg_ex = torch.cat([query_fg_ex, ray_dir_input[None, ...].expand(K-1, -1, -1).flatten(0, 1)], dim=1)

        if no_concatenate and pixel_feat is not None:
            if pixel_feat==None:
                raise NotImplementedError('there should be pixel_feat for no_concatenate')
            pixel_feat = self.change_dim(pixel_feat) # Kx(P)xC_z_slot
            # z_slots: KxC
            pixel_feat = self.pixel_norm(pixel_feat)
            z_slots = self.object_norm(z_slots)
            z_slots = z_slots[:, None, :].expand(-1, P, -1)
            if self.weight_pixel_slot_mask:
                silhouettes = silhouettes.permute([2, 0, 1, 3, 4]).flatten(1, 4).unsqueeze(-1)  # Kx(NxDxHxW)x1
                feat = pixel_feat * silhouettes + z_slots * (1-silhouettes)
                # print('you are using weight_pixel_slot_mask')
            else:
                feat = pixel_feat+z_slots
                # feat = self.norm2feat(feat)

            input_bg = torch.cat([feat[0:1].squeeze(0), query_bg], dim=1)
            input_fg = torch.cat([feat[1:].flatten(0, 1), query_fg_ex], dim=1)

        else:
            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            if self.without_slot_feature:
                # print(self.without_slot_feature, 'without_slot_feature')

                input_bg = query_bg
                # print(input_bg.shape, 'input_bg.shape')
            else:
                input_bg = torch.cat([query_bg, z_bg.expand(P, -1)], dim=1)  # Px(60+C)
            if pixel_feat is not None and not self.bg_no_pixel and not self.color_after_density:
                input_bg = torch.cat([pixel_feat[0:1].squeeze(0), input_bg], dim=1)
            if self.without_slot_feature:
                input_fg = query_fg_ex
            else:
                z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
                input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=1)  # ((K-1)xP)x(60+C)
            if pixel_feat is not None and not self.color_after_density:
                input_fg = torch.cat([pixel_feat[1:].flatten(0, 1), input_fg], dim=1)
                # print(input_fg.shape, 'input_fg.shape') # torch.Size([4194304, 225]) input_fg.shape in tdw

        if decoder_type == 'color':
            '''
            transmittance_samples: 1x(NxDxHxW)x1
            '''
            input_bg = torch.cat([input_bg, transmittance_samples.flatten(0, 1)], dim=1)
            input_fg = torch.cat([input_fg, transmittance_samples.expand(K-1, -1, -1).flatten(0, 1)], dim=1)

        if self.mask_as_decoder_input or decoder_type=='color':
            '''
            silhouettes: # NxDxKxHxW
            '''
            silhouettes = silhouettes.permute([2, 0, 1, 3, 4]).flatten(1, 4).unsqueeze(-1)  # Kx(NxDxHxW)x1
            if self.multiply_mask_pixelfeat:
                input_bg = torch.cat([input_bg, silhouettes[0] * pixel_feat[0:1].squeeze(0)], dim=1)
                input_fg = torch.cat([input_fg, silhouettes[1:].flatten(0, 1) * pixel_feat[1:].flatten(0, 1)], dim=1)
            else:
                input_bg = torch.cat([input_bg, silhouettes[0]], dim=1)
                input_fg = torch.cat([input_fg, silhouettes[1:].flatten(0, 1)], dim=1)

        if self.density_as_color_input:
            assert raw_masks_density != None, 'density should not be None'
            input_bg = torch.cat([input_bg, raw_masks_density[0]], dim=1)
            input_fg = torch.cat([input_fg, raw_masks_density[1:].flatten(0, 1)], dim=1)

        # print(input_bg.shape, 'input_bg.shape')
        # print(input_fg.shape, 'input_fg.shape')
        tmp = self.b_before(input_bg)
        if (self.use_ray_dir and self.ray_after_density) or self.same_bg_fg_decoder:
            tmp = self.b_after(torch.cat([input_bg, tmp], dim=1))
            latent_bg = self.b_after_latent(tmp)
            latent_bg = torch.cat([latent_bg, ray_dir_input], dim=1)
            if self.color_after_density:
                latent_bg = torch.cat([pixel_feat[0:1].squeeze(0), latent_bg], dim=1)
            bg_raw_rgb = self.b_color(latent_bg).view(1, P, 3)
            bg_raw_shape = self.b_after_shape(tmp).view(1, P, 1)
            bg_raws = torch.cat([bg_raw_rgb, bg_raw_shape], dim=-1)
        else:
            if self.reduce_color_decoder:
                bg_raws = self.b_after(tmp).view([1, P, self.out_ch])
            else:
                bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=1)).view([1, P, self.out_ch])  # Px5 -> 1xPx5

        tmp = self.f_before(input_fg)
        if self.reduce_color_decoder:
            tmp = self.f_after(tmp)
        else:
            tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # ((K-1)xP)x64
        latent_fg = self.f_after_latent(tmp)  # ((K-1)xP)x64
        if self.use_ray_dir:
            if self.ray_after_density:
                latent_fg = torch.cat([latent_fg, ray_dir_input[None, ...].expand(K-1, -1, -1).flatten(0, 1)], dim=1)
        if self.color_after_density:
            latent_fg = torch.cat([pixel_feat[1:].flatten(0, 1), latent_fg], dim=1)
        fg_raw_rgb = self.f_color(latent_fg).view([K-1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
        fg_raw_shape = self.f_after_shape(tmp).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
        if self.locality:
            # fg_raw_shape[outsider_idx] *= 0
            fg_raw_shape[z_out_idx_fg] *= 0
            # debug
            bg_raw_shape = bg_raws[..., 3:4]
            bg_raw_shape[z_out_idx_bg] *= 0
            bg_raws[..., 3:4] = bg_raw_shape
        if self.restrict_world:
            fg_raw_shape[outside_world_idx.squeeze(-1).expand(K-1, -1)] *= 0
            bg_raw_shape = bg_raws[..., -1:]
            bg_raw_shape[outside_world_idx] *= 0  # 1xPx1
            bg_raws[..., -1:] = bg_raw_shape
        fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

        all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
        raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
        if decoder_type=='color':
            raw_masks = raw_masks_density
        masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
        raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
        raw_sigma = raw_masks

        masked_rgb = raw_rgb * masks
        unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
        masked_raws = torch.cat([masked_rgb, raw_sigma], dim=2)
        # at_home
        # masked_raws = unmasked_raws * masks
        raws = masked_raws.sum(dim=0)

        if decoder_type=='density':
            return raws, masked_raws, unmasked_raws, masks, raw_masks

        return raws, masked_raws, unmasked_raws, masks

class PixelDecoder(nn.Module):
    def __init__(self, n_freq=5, input_dim=33+64+128, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False, pixel_positional_encoding=None, use_ray_dir=None, slot_repeat=None):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.out_ch = 4
        self.d_out = 4
        self.pixel_positional_encoding = pixel_positional_encoding
        self.use_ray_dir = use_ray_dir
        if pixel_positional_encoding:
            self.positional_encoding = PositionalEncoding(num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True)
            input_dim += self.positional_encoding.d_out
            print(self.positional_encoding.d_out)
        if use_ray_dir:
            input_dim += 3

        if slot_repeat:
            self.bg = ResnetFC(d_in=input_dim-64, d_out=4, n_blocks=4, d_latent=64, d_hidden=64,
                            beta=0.0, combine_layer=2, combine_type="average", use_spade=False, slot_repeat=slot_repeat,)
            self.fg = ResnetFC(d_in=input_dim-64, d_out=4, n_blocks=4, d_latent=64, d_hidden=64,
                            beta=0.0, combine_layer=2, combine_type="average", use_spade=False, slot_repeat=slot_repeat,)
        else:
            self.bg = ResnetFC(d_in=input_dim-128, d_out=4, n_blocks=3, d_latent=128, d_hidden=64,
                               beta=0.0, combine_layer=1, combine_type="average", use_spade=False,)
            self.fg = ResnetFC(d_in=input_dim-128, d_out=4, n_blocks=3, d_latent=128, d_hidden=64,
                               beta=0.0, combine_layer=1, combine_type="average", use_spade=False,)

    def forward(self, sampling_coor_bg, sampling_coor_fg, z_slots, fg_transform, pixel_feat, ray_dir=None, use_background=True, slot_repeat=None):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        if self.pixel_positional_encoding:
            K = 2 # because it is pixelnerf
        else:
            K, C = z_slots.shape
        P = sampling_coor_bg.shape[0]

        if self.fixed_locality:
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
            sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # (K-1)xPx4
            sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
        else:
            sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

        if not self.pixel_positional_encoding:
            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
        if not self.pixel_positional_encoding:
            query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
            input_bg = torch.cat([query_bg, z_bg.expand(P, -1)], dim=1)  # Px(60+C)
        else:
            input_bg = self.positional_encoding(sampling_coor_bg)
        input_bg = torch.cat([pixel_feat[0:1].squeeze(0), input_bg], dim=1)
        if ray_dir is not None:
            input_bg = torch.cat([input_bg, ray_dir.expand(P, -1)], dim=1)

        sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
        if not self.pixel_positional_encoding:
            query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
        else:
            input_fg = self.positional_encoding(sampling_coor_fg_)
        if not self.pixel_positional_encoding:
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=1)  # ((K-1)xP)x(60+C)
        input_fg = torch.cat([pixel_feat[1:].flatten(0, 1), input_fg], dim=1)
        if ray_dir is not None:
            input_fg = torch.cat([input_fg, ray_dir.expand(P, -1)], dim=1)
            #TODO: need to generalize for the case K-1 != 1
        # print(input_fg.shape, 'input_fg.shape') # torch.Size([4194304, 225]) input_fg.shape

        # Camera frustum culling stuff, currently disabled
        combine_index = None
        dim_size = None
        self.num_views_per_obj = 1
        B = input_fg.shape[0] # B is batch of points (in rays)

        # Run main NeRF network
        mlp_output_fg = self.fg(input_fg, combine_inner_dims=(self.num_views_per_obj, B),)
        if use_background:
            mlp_output_bg = self.bg(input_bg, combine_inner_dims=(self.num_views_per_obj, B//(K-1)),)

            mlp_output_fg = mlp_output_fg.reshape(-1, B//(K-1), self.d_out)
            fg_raw_shape = mlp_output_fg[..., 3]
            fg_raw_rgb = mlp_output_fg[..., :3]
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

            mlp_output_bg = mlp_output_bg.reshape(-1, B//(K-1), self.d_out)
            bg_raws = mlp_output_bg

            all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
            raw_sigma = raw_masks

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)
        else:
            mlp_output_fg = mlp_output_fg.reshape(-1, B // (K - 1), self.d_out)
            fg_raw_shape = mlp_output_fg[..., 3]
            fg_raw_rgb = mlp_output_fg[..., :3]
            if self.locality:
                fg_raw_shape[outsider_idx] *= 0
            fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # 1xPx4

            all_raws = fg_raws  # 1xPx4
            raw_masks = F.relu(all_raws[:, :, -1:], True)  # 1xPx1
            masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # 1xPx1
            raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
            raw_sigma = raw_masks

            unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
            masked_raws = unmasked_raws * masks
            raws = masked_raws.sum(dim=0)

        return raws, masked_raws, unmasked_raws, masks


class SlotAttention(nn.Module):
    def __init__(self, num_slots, in_dim=64, slot_dim=64, iters=3, eps=1e-8, hidden_dim=128, gt_seg=False):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5
        self.gt_seg = gt_seg

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma_bg)

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim

    def forward(self, feat, num_slots=None, masks=None):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(B, K-1, -1)
        sigma = self.slots_logsigma.exp().expand(B, K-1, -1)
        slot_fg = mu + sigma * torch.randn_like(mu)
        mu_bg = self.slots_mu_bg.expand(B, 1, -1)
        sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
        slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)

        feat = self.norm_feat(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        attn = None
        for _ in range(self.iters):
            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg
            q_fg = self.to_q(slot_fg)
            q_bg = self.to_q_bg(slot_bg)

            dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale
            dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale
            dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN

            if self.gt_seg:
                # Replace slot attention with GT segmentations
                attn_bg, attn_fg = masks[:, 0:1], masks[:, 1:]
                attn = torch.cat([attn_bg, attn_fg], dim=1)
                # print(masks.shape, 'mask.shape') # 1x5x4096
                # print(attn.shape, 'attn.shape') # 1x5x4096
            else:
                attn = dots.softmax(dim=1) + self.eps  # BxKxN
                attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN

            # Add small epsilon to prevent division by zero
            attn_weights_bg = attn_bg / (attn_bg.sum(dim=-1, keepdim=True) + 1e-12)  # Bx1xN
            attn_weights_fg = attn_fg / (attn_fg.sum(dim=-1, keepdim=True) + 1e-12)  # Bx(K-1)xN

            updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
            updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

            slot_bg = self.gru_bg(
                updates_bg.reshape(-1, self.slot_dim),
                slot_prev_bg.reshape(-1, self.slot_dim)
            )
            slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
            slot_bg = slot_bg + self.to_res_bg(slot_bg)

            slot_fg = self.gru(
                updates_fg.reshape(-1, self.slot_dim),
                slot_prev_fg.reshape(-1, self.slot_dim)
            )
            slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
            slot_fg = slot_fg + self.to_res(slot_fg)

        slots = torch.cat([slot_bg, slot_fg], dim=1)
        return slots, attn


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
    embedded_ = torch.cat(embedded, dim=1)
    return embedded_

def raw2outputs(raw, z_vals, rays_d, render_mask=False, weight_pixelfeat=None, raws_slot=None, uvw=None, wuv=None, KNDHW=None,
                return_transmittance_cam0=False, input_transmittance_cam0=None, return_silhouettes=None,
                transmittance_samples=None, masks=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray in cam coor.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # uvw: 1x(NxDxHxW)x3
    # raw: (NxHxW)xDx4

    raw2alpha = lambda x, y: 1. - torch.exp(-x * y)
    device = raw.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e-2], device=device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples] # add dist 0 in the end

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
    transmittance = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], dim=-1)
        , dim=-1)[:,:-1]
    weights = alpha * transmittance # [N_rays, N_samples]

    if weight_pixelfeat:
        K, N, D, H, W = KNDHW
        if input_transmittance_cam0 is not None:
            transmittance_cam0 = input_transmittance_cam0
        else:
            transmittance_cam0 = transmittance.view(N, H, W, D)[0] #HxWxD
            transmittance_cam0 = transmittance_cam0.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0) # 1x1xDxHxW

        uvw = uvw.unsqueeze(2).unsqueeze(3)  # (1, (NxDxHxW), 1, 1, 3)
        # self.latent(B, L, H, W)
        transmittance_samples = F.grid_sample(
            transmittance_cam0,
            uvw,
            # align_corners=True,
            mode='bilinear',
            padding_mode='zeros',
        ) # 1x1x(NxDxHxW)x1x1

        transmittance_samples = transmittance_samples[0, 0, :, 0, 0] # (NxDxHxW)
        transmittance_samples = transmittance_samples.view(N, D, H, W).permute([0, 2, 3, 1]).flatten(0, 2) # (NxHxW)xD

        # print(weights_samples.max(), weights_samples.min(), weights_samples.median(), weights_samples.mean(), 'weights_samples.max(), weights_samples.min(), weights_samples.median(), weights_samples.mean()')
        weights_pixel = weights * transmittance_samples
        weights_slot = weights * (1.- transmittance_samples)
        rgb_pixel = raw[..., :3]
        rgb_slot = raws_slot[..., :3]
        rgb_map = torch.sum(weights_pixel[..., None] * rgb_pixel + weights_slot[..., None] * rgb_slot, -2)

    else:
        rgb = raw[..., :3]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        transmittance_cam0 = None
        transmittance_samples = None

    # at_home
    # weights_norm = weights.detach() + 1e-5
    weights_norm = weights + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True)
    depth_map = torch.sum(weights_norm * z_vals, -1)

    if render_mask:
        density = raw[..., 3]  # [N_rays, N_samples]
        mask_map = torch.sum(weights * density, dim=1)  # [N_rays,]
    else:
        mask_map = None

    if return_silhouettes is not None:
        masks = return_silhouettes # [K, N, D, H, W]
        masks = masks.permute([1, 3, 4, 2, 0]) # [N, H, W, D, K]
        weights_silhouettes = weights.view([masks.shape[0], masks.shape[1], masks.shape[2], masks.shape[3]]).unsqueeze(-1) # (NxHxW)xD -> NxHxWxDx1
        silhouettes = torch.sum(weights_silhouettes * masks, dim=-2) # NxHxWxK
        silhouettes = silhouettes.permute([0, 3, 1, 2]) # NxKxHxW
    else:
        silhouettes = None

    return rgb_map, depth_map, weights_norm, mask_map, transmittance_cam0, silhouettes

def raw2transmittances(raw, z_vals, rays_d, uvw, KNDHW, masks):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray in cam coor.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # uvw: 1x(NxDxHxW)x3
    # raw: (NxHxW)xDx4

    raw2alpha = lambda x, y: 1. - torch.exp(-x * y)
    device = raw.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e-2], device=device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples] # add dist 0 in the end

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
    transmittance = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], dim=-1)
        , dim=-1)[:,:-1]
    weights = alpha * transmittance # [N_rays, N_samples]


    K, N, D, H, W = KNDHW
    transmittance_cam0 = transmittance.view(N, H, W, D)[0] #HxWxD
    transmittance_cam0 = transmittance_cam0.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0) # 1x1xDxHxW

    uvw = uvw.unsqueeze(2).unsqueeze(3)  # (1, (NxDxHxW), 1, 1, 3)
    # self.latent(B, L, H, W)
    transmittance_samples = F.grid_sample(
        transmittance_cam0,
        uvw,
        # align_corners=True,
        mode='bilinear',
        padding_mode='zeros',
    ) # 1x1x(NxDxHxW)x1x1

    transmittance_samples = transmittance_samples[0, 0, :, 0, 0] # (NxDxHxW)
    transmittance_samples = transmittance_samples.view(N, D, H, W).permute([0, 2, 3, 1]).flatten(0, 2) # Nx(HxWxD)

    masks = masks.permute([1, 3, 4, 2, 0])  # [N, H, W, D, K]
    weights_silhouettes = weights.view([masks.shape[0], masks.shape[1], masks.shape[2], masks.shape[3]]).unsqueeze(
        -1)  # (NxHxW)xD -> NxHxWxDx1
    silhouettes = torch.sum(weights_silhouettes * masks, dim=-2)  # NxHxWxK
    silhouettes = silhouettes.permute([0, 3, 1, 2])  # NxKxHxW

    return weights, transmittance_samples, silhouettes

def raw2colors(raw, weights, z_vals):
    rgb = raw[..., :3]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    weights_norm = weights + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True)
    depth_map = torch.sum(weights_norm * z_vals, -1)

    density = raw[..., 3]  # [N_rays, N_samples]
    mask_map = torch.sum(weights * density, dim=1)  # [N_rays,] # render_mask

    return rgb_map, depth_map, weights_norm, mask_map


def get_perceptual_net(layer=4):
    assert layer > 0
    idx_set = [None, 4, 9, 16, 23, 30]
    idx = idx_set[layer]
    vgg = vgg16(pretrained=True)
    loss_network = nn.Sequential(*list(vgg.features)[:idx]).eval()
    for param in loss_network.parameters():
        param.requires_grad = False
    return loss_network


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean(), fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = F.leaky_relu(out, 0.2, inplace=True) * 1.4

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )



class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        stride=1,
        padding=1
    ):
        layers = []

        if downsample:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, stride=1, padding=1)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, stride=1, padding=1)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False, stride=1, padding=0
        )

    def forward(self, input):
        out = self.conv1(input) * 1.4
        out = self.conv2(out) * 1.4

        skip = self.skip(input) * 1.4
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, ndf, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: ndf*2,
            8: ndf*2,
            16: ndf,
            32: ndf,
            64: ndf//2,
            128: ndf//2
        }

        convs = [ConvLayer(3, channels[size], 1, stride=1, padding=1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, stride=1, padding=1)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input) * 1.4

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out) * 1.4

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


import torch.autograd.profiler as profiler
import numpy as np

class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1)
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get_int("num_freqs", 6),
            d_in,
            conf.get_float("freq_factor", np.pi),
            conf.get_bool("include_input", True),
        )

class CentroidDecoder(nn.Module):
    def __init__(self, input_dim, z_dim, cam2pixel, world2nss, near, far, small_latent=False, n_layers=3):
        """
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        """
        super().__init__()

        latent_dim = z_dim // 2 if small_latent else z_dim

        # [Create centroid decoder]
        before_skip = [nn.Linear(input_dim, latent_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(latent_dim + input_dim, latent_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(latent_dim, latent_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(latent_dim, latent_dim))
            after_skip.append(nn.ReLU(True))

        self.p_before = nn.Sequential(*before_skip)
        self.p_after = nn.Sequential(*after_skip)
        self.p_after_latent = nn.Linear(latent_dim, latent_dim)
        self.p_pos = nn.Sequential(nn.Linear(latent_dim, latent_dim // 4),
                                     nn.ReLU(True),
                                     nn.Linear(latent_dim // 4, 3))

        # [Set attributes]
        self.cam2pixel = cam2pixel
        self.world2nss = world2nss
        self.near = near
        self.far = far
        self.frustum_size = None
        self.cam2world = None

    def forward(self, p_slots):

        assert p_slots is not None
        p_fg = p_slots[1:]  # (K-1)xC
        tmp = self.p_before(p_fg)
        tmp = self.p_after(torch.cat([p_fg, tmp], dim=1))
        latent = self.p_after_latent(tmp)
        fg_pos = self.p_pos(latent)
        fg_pos = fg_pos.sigmoid() # (K-1)x3

        cam02world = self.cam2world[0]  # 4x4
        world2cam = self.cam2world.inverse()  # Nx4x4
        cam2pixel = self.cam2pixel[None]  # 1x4x4
        pixel2cam = self.cam2pixel.inverse()  # 1x4x4

        frustum_size = self.frustum_size

        # [Project predicted centroid]
        z_cam = fg_pos[..., -1] * (self.far - self.near) + self.near  # (K-1)
        fg_pos = fg_pos * (frustum_size.unsqueeze(0) - 1)  # [0, 1] -> [0, frustum_size]

        unorm_fg_pos = torch.zeros_like(fg_pos)  # (K-1)x3
        unorm_fg_pos[..., 0:2] = fg_pos[..., 0:2] * z_cam.unsqueeze(-1)
        unorm_fg_pos[..., 2] = z_cam

        fg_pixel0_pos = torch.cat([unorm_fg_pos, torch.ones_like(fg_pos[:, 0:1])], dim=-1)  # (K-1)x4
        fg_pixel0_pos = fg_pixel0_pos.permute(1, 0)  # 4x(K-1)
        fg_cam0_pos = torch.matmul(pixel2cam, fg_pixel0_pos)  # 4x4, 4x(K-1) -> 4x(K-1)
        fg_centroid_world = torch.matmul(cam02world, fg_cam0_pos)  # 4x4, 4x(K-1) -> 4x(K-1)
        fg_centroid_cam = torch.matmul(world2cam, fg_centroid_world)  # Nx4x4,4x(K-1) -> Nx4x(K-1)
        fg_centroid_pixel = torch.matmul(cam2pixel, fg_centroid_cam)  # 1x4x4, Nx4x(K-1) -> Nx4x(K-1)
        fg_centroid_pixel = fg_centroid_pixel.permute(0, 2, 1)  # Nx(K-1)x4

        fg_centroid_pixel = fg_centroid_pixel[:, :, 0:2] / fg_centroid_pixel[:, :, 2:3]  # Nx(K-1)x2
        fg_centroid_pixel = torch.flip(fg_centroid_pixel, dims=(-1,))

        fg_centroid_nss = torch.matmul(self.world2nss, fg_centroid_world)  # 1x4x4, 4x(K-1) -> 1x4x(K-1)

        fg_centroid_world = fg_centroid_world.permute(1, 0) # (K-1)x4
        fg_centroid_nss = fg_centroid_nss[0].permute(1, 0)  # (K-1)x4

        return fg_centroid_world, fg_centroid_nss, fg_centroid_pixel

    def transform_coords(self, coords, centroids):
        # coords: (K-1)xPx3,  cenroids: (K-1)x4
        centroids = centroids[:, 0:3].unsqueeze(1)  # (K-1)x1x3
        return coords - centroids  # (K-1)xPx3

    def centroid_loss(self, centroid_pixel, margin, segment_centers, segment_masks, epoch):
        # Target centroid positions
        target_pos = segment_centers[:, 1:]  # Nx(K-1)x2
        segment_area = segment_masks[:, 1:].sum(-1)  # Nx(K-1)x1
        loss_mask = (segment_area > 0).float().detach()

        # Normalize coordinates:
        pixel_pos = centroid_pixel / self.frustum_size[0:2].view(1, 1, 2)
        target_pos = target_pos / self.frustum_size[0:2].view(1, 1, 2)

        # Loss as the euclidean distance between the predicted and target centroid positions
        distance = ((pixel_pos - target_pos) ** 2).sum(-1) ** 0.5
        self.loss_centroid_raw = (distance - margin).clamp(min=0.) * loss_mask
        self.loss_centroid = ((distance - margin).clamp(min=0.) * loss_mask).sum()

        # [Visualization]
        if not os.path.exists('tmp/%d.png' % epoch):
            self.visualize_centroid(pixel_pos, target_pos, segment_masks, epoch=epoch)

        return self.loss_centroid

    def visualize_centroid(self, pred_center, target_center, segment_masks, epoch, savedir=None):
        if True:
            pass
        else:
            fig, axs = plt.subplots(segment_masks.shape[0], segment_masks.shape[1], figsize=(5, 5))  # Nx3

            center_x = target_center[..., 0] * self.frustum_size[0]  # Nx3
            center_y = target_center[..., 1] * self.frustum_size[1]  # Nx3

            pred_x = pred_center[..., 0] * self.frustum_size[0]  # Nx3
            pred_y = pred_center[..., 1] * self.frustum_size[1]  # Nx3

            for i in range(segment_masks.shape[0]):
                _x = ((self.x[i] + 1) / 2).clone()

                axs[i, 0].imshow(_x.permute(1, 2, 0).cpu())
                axs[i, 0].set_axis_off()
                for j in range(1, segment_masks.shape[1]):
                    axs[i, j].imshow(segment_masks[i, j].cpu().reshape([int(self.frustum_size[0]), int(self.frustum_size[1])]))
                    center = [center_y[i, j - 1], center_x[i, j - 1]]
                    pred = [pred_y[i, j - 1], pred_x[i, j - 1]]
                    axs[i, j].add_patch(plt.Circle(center, 2.0, color='g'))
                    axs[i, j].add_patch(plt.Circle(pred, 2.0, color='r'))
                    axs[i, j].set_axis_off()
                    axs[i, j].set_title('dist: %.2f' % self.loss_centroid_raw[i, j-1], fontsize=10)
            plt.show()
            fig.suptitle('Centroid loss: %.3f' % self.loss_centroid)
            if savedir is None:
                plt.savefig('tmp/%d.png' % epoch, bbox_inches='tight')
            else:
                plt.savefig('tmp/%s.png' % savedir, bbox_inches='tight')

            plt.close()

        # Save data for visualization
        # data = {
        #     'img': (self.x + 1) / 2,
        #     'world_pos': world_pos,
        #     'segment': segment_masks
        # }
        # print('Save data to ', './save_tensor/%d.pt' % epoch)
        # torch.save(data, './save_tensor/%d.pt' % epoch)

        # Visualize the line of projection with varying depth
        # if epoch > 100000:
        #
        #     num = 20
        #     interval = 1.0 / num
        #     temp_pos = save_fg_pos.clone()
        #
        #     for i in range(num):
        #         temp_pos[:, -1] = i * interval
        #         temp_uv = project_pos(temp_pos)
        #         plot(temp_uv, savedir='%d-%d' % (epoch, i))
        #     breakpoint()