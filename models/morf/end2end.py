from torch import nn
from models.base_classes import End2end
from models.morf.encoder.encoder_wrapper import EncoderWrapper
from models.morf.decoder.uorf_decoder import UorfDecoder
from models.morf.renderer.uorf_renderer import UorfRenderer

class MorfEnd2end(End2end):
    def __init__(self, opt):
        super().__init__(opt)
        self.encoder = EncoderWrapper(opt)
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

        # input_img: BxNx3xHxW
        # Consider changing the input image size
        input_encoder = {'input_img': input_end2end['input_img'][:, 0, ...],
                         'input_mask': input_end2end['input_mask']}
        output_encoder = self.encoder(input_encoder)

        input_renderer = {'encoder_obj': self.encoder,
                          'decoder_obj': self.decoder,
                          'cam2world': input_end2end['cam2world'],
                          'cam2world_azi': input_end2end['cam2world_azi'],
                          'input_img': input_end2end['input_img'],
                          'input_mask': input_end2end['input_mask']}
        output_renderer = self.renderer.render(input_renderer)

        output_end2end = output_renderer
        output_end2end['output_mask'] = output_encoder['output_mask']
        return output_end2end




