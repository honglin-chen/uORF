from torch import nn
from models.base_classes import End2end
from models.morf.encoder import PixelNerfEncoder
from models.morf.decoder import UorfDecoder
from models.morf.Renderer import UorfRenderer

class MorfEnd2end(End2end):
    def __init__(self):
        super().__init__(self, opt)
        self.encoder = PixelNerfEncoder(opt)
        self.decoder = UorfDecoder(opt)
        self.renderer = UorfRenderer(opt)

    def forward(self, input_end2end):
        '''
        Help: get novel view and geometry of the scene given an image
        Input: Dict{input image, input masks, cam_matrices} # {BxHxWx3, BxKxHxWx1, BxNx4x4}
        Output: Dict{images, slot_images, optional_outputs} # {BxNxHxWx3, BxNxKxHxWx3, etc}
            Optional outputs:
                depth_maps # BxNxHxWx1
                occ_silhouettes (occluded) # BxNxKxHxWx1
                unocc_silhouettes (unoccluded) # BxNxKxHxWx1
                obj_masks # BxNxKxHxWx1
        '''

        input_encoder = {'input_img': input_end2end['input_img'],
                         'input_mask': input_end2end['input_mask']}
        self.encoder(input_encoder)

        input_renderer = {'encoder_obj': self.encoder,
                          'decoder_obj': self.decoder,
                          'cam2world': input_end2end['cam2world'],
                          'input_img': input_end2end['input_img']}
        output_renderer = self.renderer(input_renderer)

        output_end2end = output_renderer
        return output_end2end




