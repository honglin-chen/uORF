from models.base_classes import Encoder
import torch
from torch import nn
import torch.nn.functional as F


class PixelEncoder(Encoder):
    def __init__(self, opt):
        super().__init__(self, opt)

    def forward(self, input_encoder):
        raise NotImplementedError

    def get_feature(self, coor_feature):
        raise NotImplementedError