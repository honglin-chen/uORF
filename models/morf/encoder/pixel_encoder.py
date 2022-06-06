from models.base_classes import Encoder
import torch
from torch import nn
import torch.nn.functional as F

import torchvision
# from model.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler


class PixelEncoder(Encoder):
    def __init__(self, opt):
        super().__init__(opt)
        self.pixel_encoder = PixelNerfEncoder(mask_image=self.opt.mask_image_slot,
                                              mask_image_feature=self.opt.mask_image_feature_slot,
                                              index_interp=self.opt.interp_mode)

    def forward(self, input_encoder):
        masks = input_encoder['input_mask'] # (BxK)xHxWx3
        masks = masks.squeeze(0) if self.opt.mask_image or self.opt.mask_image_feature else None
        feature_map_pixel = self.pixel_encoder(input['input_img'], masks=masks)
        return masks

    def get_feature(self, coor_feature):
        if not self.opt.use_pixel_feat:
            return None
        uv = coor_feature['pixel_coor']
        B, K = self.opt.batch_size, self.opt.num_slots

        if self.opt.mask_image or self.opt.mask_image_feature:
            uv = uv.expand(B*K, -1, -1) # 1x(NxDxHxW)x2 -> Kx(NxDxHxW)x2
            pixel_feat = self.pixel_encoder.index(uv)  # Kx(NxDxHxW)x2 -> KxCx(NxDxHxW)
            pixel_feat = pixel_feat.transpose(1, 2)
        else:
            pixel_feat = self.pixel_encoder.index(uv)  # 1x(NxDxHxW)x2 -> 1xCx(NxDxHxW)
            pixel_feat = pixel_feat.transpose(1, 2)  # 1x(NxDxHxW)xC
            pixel_feat = pixel_feat.expand(K, -1, -1) # Kx(NxDxHxW)xC
            uv = uv.expand(K, -1, -1)  # 1x(NxDxHxW)x2 -> Kx(NxDxHxW)x2
        return pixel_feat


"""
Implements image encoders
"""

class PixelNerfEncoder(nn.Module):
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
        B = x.shape[0]

        if self.mask_image or self.mask_image_feature:
            K = masks.shape[0] # this is BK
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