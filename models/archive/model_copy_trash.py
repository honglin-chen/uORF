import math
from models.op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd
from models.networks import get_norm_layer
from models.resnetfc import ResnetFC


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

    def forward(self, x):
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
        if self.mask_image_feature:
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
        if self.mask_image_feature or self.mask_image:
            #then, the self.latent_list is a list of self.latent for each mask
            sample_list = []
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
                for idx, latent in enumerate(self.latent_list):
                    samples = F.grid_sample(
                        latent,
                        uv,
                        align_corners=True,
                        mode=self.index_interp,
                        padding_mode=self.index_padding,
                    )
                    sample_list.append(samples[:, :, :, 0])  # (B, C, N)
            return sample_list

        else:
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
                    align_corners=True,
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
        # print(x.shape, 'x.shape')
        # print(masks.shape, 'mask.shape after x')
        shape = x.shape
        if self.mask_image:
            self.latent_list = []
            xs = []
            for _ in range(self.n_mask):
                xs.append(x.clone())
            for idx, x in enumerate(xs):
                if self.feature_scale != 1.0:
                    x = F.interpolate(
                        x,
                        scale_factor=self.feature_scale,
                        mode="bilinear" if self.feature_scale > 1.0 else "area",
                        align_corners=True if self.feature_scale > 1.0 else None,
                        recompute_scale_factor=True,
                    )
                x = x.to(device=self.latent.device)

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
                    align_corners = None if self.index_interp == "nearest " else True
                    latent_sz = latents[0].shape[-2:]
                    for i in range(len(latents)):
                        latents[i] = F.interpolate(
                            latents[i],
                            latent_sz,
                            mode=self.upsample_interp,
                            align_corners=align_corners,
                        )
                    self.latent = torch.cat(latents, dim=1)
                self.latent_scaling[0] = self.latent.shape[-1]
                self.latent_scaling[1] = self.latent.shape[-2]
                self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
                if self.reduce_latent_size:
                    self.latent = self.mlp(self.latent.transpose(3, 1)).transpose(3, 1)

                if self.mask_image_feature:
                    mask = masks[idx].view(shape[-2], shape[-1])
                    mask = self.downsample(mask[None, :])
                    self.latent_list.append(self.latent.clone() * mask[None, :, :, :])
                else:
                    self.latent_list.append(self.latent.clone())

            return self.latents

        else:
            if self.feature_scale != 1.0:
                x = F.interpolate(
                    x,
                    scale_factor=self.feature_scale,
                    mode="bilinear" if self.feature_scale > 1.0 else "area",
                    align_corners=True if self.feature_scale > 1.0 else None,
                    recompute_scale_factor=True,
                )
            x = x.to(device=self.latent.device)

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
                align_corners = None if self.index_interp == "nearest " else True
                latent_sz = latents[0].shape[-2:]
                for i in range(len(latents)):
                    latents[i] = F.interpolate(
                        latents[i],
                        latent_sz,
                        mode=self.upsample_interp,
                        align_corners=align_corners,
                    )
                self.latent = torch.cat(latents, dim=1)
            self.latent_scaling[0] = self.latent.shape[-1]
            self.latent_scaling[1] = self.latent.shape[-2]
            self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
            if self.reduce_latent_size:
                self.latent = self.mlp(self.latent.transpose(3, 1)).transpose(3, 1)

            if self.mask_image_feature:
                self.latent_list = []
                for idx in range(self.n_mask):
                    # print(masks.shape, 'masks.shape')
                    # print(self.latent.shape, 'self.latent.shape')
                    mask = masks[idx].view(shape[-2], shape[-1])
                    mask = self.downsample(mask[None, :])
                    self.latent_list.append(self.latent.clone() * mask[None, :, :, :])
                return self.latent_list

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
    def __init__(self, n_freq=5, input_dim=33+64, pixel_dim=None, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False, no_concatenate=False, bg_no_pixel=False, use_ray_dir=False, small_latent=False):
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

        if use_ray_dir:
            input_dim += 3
        if pixel_dim is not None:
            if self.no_concatenate:
                pass
            else:
                input_dim += pixel_dim

        if small_latent:
            latent_dim = z_dim // 2
        else:
            latent_dim = z_dim
        before_skip = [nn.Linear(input_dim, latent_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(latent_dim+input_dim, latent_dim), nn.ReLU(True)]
        for i in range(n_layers-1):
            before_skip.append(nn.Linear(latent_dim, latent_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(latent_dim, latent_dim))
            after_skip.append(nn.ReLU(True))
        self.f_before = nn.Sequential(*before_skip)
        self.f_after = nn.Sequential(*after_skip)
        self.f_after_latent = nn.Linear(latent_dim, latent_dim)
        self.f_after_shape = nn.Linear(latent_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(latent_dim, latent_dim//4),
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
        after_skip.append(nn.Linear(latent_dim, self.out_ch))
        self.b_before = nn.Sequential(*before_skip)
        self.b_after = nn.Sequential(*after_skip)

        if self.no_concatenate:
            self.change_dim = nn.Linear(pixel_dim, z_dim)
            self.pixel_norm = nn.LayerNorm(z_dim, elementwise_affine=True)
            self.object_norm = nn.LayerNorm(z_dim, elementwise_affine=True)
            self.norm2feat = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.ReLU(inplace=True),
                nn.Linear(z_dim, z_dim)
            )

    def forward(self, sampling_coor_bg, sampling_coor_fg, z_slots, fg_transform, pixel_feat=None, no_concatenate=None, ray_dir_input=None):
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

        if self.fixed_locality:
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
            sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # (K-1)xPx4
            sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
        else:
            sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

        query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
        sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
        query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
        if self.use_ray_dir:
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
            feat = pixel_feat+z_slots
            feat = self.norm2feat(feat)
            input_bg = torch.cat([feat[0:1].squeeze(0), query_bg], dim=1)
            input_fg = torch.cat([feat[1:].flatten(0, 1), query_fg_ex], dim=1)

        else:
            z_bg = z_slots[0:1, :]  # 1xC
            z_fg = z_slots[1:, :]  # (K-1)xC
            input_bg = torch.cat([query_bg, z_bg.expand(P, -1)], dim=1)  # Px(60+C)
            if pixel_feat is not None and not self.bg_no_pixel:
                input_bg = torch.cat([pixel_feat[0:1].squeeze(0), input_bg], dim=1)
            z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
            input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=1)  # ((K-1)xP)x(60+C)
            if pixel_feat is not None:
                input_fg = torch.cat([pixel_feat[1:].flatten(0, 1), input_fg], dim=1)
                # print(input_fg.shape, 'input_fg.shape') # torch.Size([4194304, 225]) input_fg.shape in tdw

        # print(input_bg.shape, 'input_bg.shape')
        # print(input_fg.shape, 'input_fg.shape')
        tmp = self.b_before(input_bg)
        bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=1)).view([1, P, self.out_ch])  # Px5 -> 1xPx5
        tmp = self.f_before(input_fg)
        tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # ((K-1)xP)x64
        latent_fg = self.f_after_latent(tmp)  # ((K-1)xP)x64
        fg_raw_rgb = self.f_color(latent_fg).view([K-1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
        fg_raw_shape = self.f_after_shape(tmp).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
        if self.locality:
            fg_raw_shape[outsider_idx] *= 0
        fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

        all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
        raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
        masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
        raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
        raw_sigma = raw_masks

        unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
        masked_raws = unmasked_raws * masks
        raws = masked_raws.sum(dim=0)

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

def raw2outputs(raw, z_vals, rays_d, render_mask=False, weigh_pixelfeat=None, raws_slot=None, wuv=None, KNDHW=None, div_by_max=None):
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
    dists = torch.cat([dists, torch.tensor([1e-2], device=device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:,:-1] # [N_rays, N_samples]

    if weigh_pixelfeat:

        K, N, D, H, W = KNDHW

        weights_norm = weights.clone() + 1e-5
        weights_norm /= weights_norm.sum(dim=-1, keepdim=True)

        weights_cam0 = weights_norm.view(N, H, W, D)[0]  # H, W, D
        weights_cam0 = weights_cam0.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)

        wuv = wuv.unsqueeze(2).unsqueeze(3)  # (1, (NxDxHxW), 1, 1, 3)
        # self.latent(B, L, H, W)
        weights_samples = F.grid_sample(
            weights_cam0,
            wuv,
            align_corners=True,
            mode='bilinear',
            padding_mode='zeros',
        ) # 1x1x(NxDxHxW)x1x1

        weights_samples = weights_samples[0, 0, :, 0, 0] # (NxDxHxW)
        if div_by_max:
            weights_samples = weights_samples.view(N, D, H, W)
            max = torch.max(weights_samples, dim=1, keepdim=True)
            weights_samples = weights_samples/max
            weights_samples = weights_samples.permute([0, 2, 3, 1]).flatten(0, 2)
        else:
            weights_samples = weights_samples.view(N, D, H, W).permute([0, 2, 3, 1]).flatten(0, 2)
        # print(weights_samples.max(), weights_samples.min(), weights_samples.median(), weights_samples.mean(), 'weights_samples.max(), weights_samples.min(), weights_samples.median(), weights_samples.mean()')
        weights_pixel = weights * weights_samples
        weights_slot = weights * (1.- weights_samples)
        rgb_pixel = raw[..., :3]
        rgb_slot = raws_slot[..., :3]
        rgb_map = torch.sum(weights_pixel[..., None] * rgb_pixel + weights_slot[..., None] * rgb_slot, -2)

    else:
        rgb = raw[..., :3]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    weights_norm = weights.detach() + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True)
    depth_map = torch.sum(weights_norm * z_vals, -1)

    if render_mask:
        density = raw[..., 3]  # [N_rays, N_samples]
        mask_map = torch.sum(weights * density, dim=1)  # [N_rays,]
        return rgb_map, depth_map, weights_norm, mask_map

    return rgb_map, depth_map, weights_norm


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