from models.base_classes import Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetEncoder(Encoder):
    def __init__(self, opt):
        super().__init__(opt)
        self.feature_map_encoder = UNet(n_channels=3, n_classes=64)
        self.decoder = UNetDecodeFC(n_channels=64, n_classes=64)

    def forward(self, input_encoder):
        # Dict{input image, input masks} # {BxHxWx3, BxKxHxWx1}
        input_mask = input_encoder['input_mask']
        feature_map = self.feature_map_encoder(input_encoder['input_image'])
        feature_map = feature_map.unsqueeze(1).expand(-1, opt.num_slots, -1, -1, -1)
        obj_feature = feature_map * input_mask / torch.sum(input_mask, dim=3, keepdim=True) # dim?
        self.obj_feature = obj_feature

        obj_feature = obj_feature.flatten(0, 1)
        pixel_feature = self.decoder(obj_feature)
        self.pixel_feature = pixel_feature
        raise NotImplementedError

    def get_feature(self, coor_feature):
        # coor_feature: coordinates from which you want to get features (uv, xyz, ...)
        raise NotImplementedError

""" Parts of the U-Net model """

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
    if (type(m) == nn.Conv2d) or (type(m) == nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            #             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        #         print("before up", x1.shape)
        x1 = self.up(x1)
        #         print("after up", x1.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        #         print("after ", x1.shape)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        #         print(x2.shape, x1.shape)
        op = self.conv(x)
        #         print(op.shape)
        return op

    def forward_sing(self, x1):
        #         print("before up", x1.shape)
        x1 = self.up(x1)
        #         print("after up", x1.shape)
        # input is CHW
        #         diffY = x2.size()[2] - x1.size()[2]
        #         diffX = x2.size()[3] - x1.size()[3]

        #         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                         diffY // 2, diffY - diffY // 2])
        #         print("after ", x1.shape)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #         x = torch.cat([x2, x1], dim=1)
        #         print(x2.shape, x1.shape)
        op = self.conv(x1)
        #         print(op.shape)
        return op


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # x : B x C x H x W
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #         print(x.shape)
        x = self.up1(x5, x4)
        #         print(x5.shape, x4.shape, x.shape)
        x = self.up2(x, x3)
        #         print(x.shape, x3.shape)
        x = self.up3(x, x2)
        #         print(x.shape)
        x = self.up4(x, x1)
        #         print(x.shape)
        logits = self.outc(x)  # B x D x H x W
        return logits


class UNetDecodeFC(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDecodeFC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.up1 = Up(n_channels, 64, bilinear)
        self.up2 = Up(64, 64, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.up4 = Up(32, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        self.decoder_input = nn.Sequential(nn.Linear(self.n_channels * 16 * 16, self.n_channels), nn.ReLU(),
                                           nn.Linear(256, 256))

        self.decoder_output = nn.Sequential(nn.Linear(self.n_channels, self.n_channels * 16 * 16), nn.ReLU())

        self.apply(init_weights)

    def forward(self, x):
        # x : B*K x D
        #         print(x.reshape(x.shape[0], -1).shape)
        #         x = self.decoder_input(x.reshape(x.shape[0], -1))
        #         x = self.down1(x)

        #         x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        x = self.decoder_output(x).reshape(x.shape[0], self.n_channels, 16, 16)

        #         x = torch.stack([torch.stack([x]*32, -1)]*32, -1)

        #         print(x.shape)

        x = self.up1.forward_sing(x)
        #         print(x.shape)
        x = self.up2.forward_sing(x)
        #         print(x.shape)
        x = self.up3.forward_sing(x)
        #         print(x.shape)
        x = self.up4.forward_sing(x)
        #         print(x.shape)
        logits = self.outc(x)  # B x C x H x W
        #         print(logits.shape)

        return logits