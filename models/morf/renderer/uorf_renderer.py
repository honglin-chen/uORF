from models.base_classes import Renderer

class UorfRenderer(Renderer):
    def __init__(self, opt):
        super().__init__(self, opt):

    def render(self, input_renderer):
        encoder = input_renderer['encoder_obj']
        decoder = input_renderer['decoder_obj']
        cam2world = input_renderer['cam2world']
        x = input_renderer['input_img']

        B, NV, C, H, W = x.shape
        K = attn.shape[1]

        z_slots, attn