from torch import nn
from models.base_classes import Encoder
from .uorf_encoder import SlotEncoder
from .pixel_encoder import PixelEncoder

class EncoderWrapper(Encoder):
    def __init__(self, opt):
        super().__init__(self, opt)
        self.slot_encoder = SlotEncoder(opt)
        self.pixel_encoder = PixelEncoder(opt)

    def forward(self, input_encoder):
        output_slot_encoder = self.slot_encoder(input_encoder)
        output_pixel_encoder = self.pixel_encoder(input_encoder)

        output_encoder = {'mask': output_slot_encoder['mask'],
                          }
        return output_encoder

    def get_feature(self, coor_feature):
        feature = {'slot_feat': self.slot_encoder.get_feature(coor_feature),
                   'pixel_feat': self.pixel_encoder.get_feature(coor_feature),
                   'voxel_feat': None
                   }
        return feature
