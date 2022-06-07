from torch import nn
from models.base_classes import Encoder
from .uorf_encoder import SlotEncoder
from .pixel_encoder import PixelEncoder
from models import networks

class EncoderWrapper(Encoder):
    def __init__(self, opt):
        super().__init__(opt)
        self.gpu_ids = opt.gpu_ids
        self.slot_encoder = networks.init_net(SlotEncoder(opt), gpu_ids=self.gpu_ids, init_type=None) if self.opt.use_slot_feat else None
        self.pixel_encoder = PixelEncoder(opt) if self.opt.use_pixel_feat else None

    def forward(self, input_encoder):
        output_slot_encoder = self.slot_encoder(input_encoder) if self.opt.use_slot_feat else None
        output_pixel_encoder = self.pixel_encoder(input_encoder) if self.opt.use_pixel_feat else None

        if self.opt.use_slot_feat:
            output_mask = output_slot_encoder['output_mask']
        elif self.opt.use_pixel_feat:
            output_mask = output_pixel_encoder['output_mask']

        output_encoder = {'output_mask': output_mask,
                          }
        return output_encoder

    def get_feature(self, coor_feature):
        slot_feat = self.slot_encoder.get_feature(coor_feature) if self.opt.use_slot_feat else None
        pixel_feat = self.pixel_encoder.get_feature(coor_feature) if self.opt.use_pixel_feat else None
        feature = {'slot_feat': slot_feat,
                   'pixel_feat': pixel_feat,
                   'voxel_feat': None
                   }
        return feature
