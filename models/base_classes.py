from torch import nn
from abc import ABC, abstractmethod

'''
    B = batch size (num of scene) = NS
    N = number of view = NV
    K = number of slots (objs), background is counted as one obj
    D = depth / D_ = coarse depth
    H = height / H_ = coarse height
    W = width / W_ = coarse width
    F = feature dim
'''

class End2end(ABC, nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @abstractmethod
    def forward(self, *args, **kwargs):
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
        raise NotImplementedError

class Encoder(ABC, nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @abstractmethod
    def forward(self, *args, **kwargs):
        '''
        Help: Encode the features of images
        Input: Dict{input image, input masks} # {BxHxWx3, BxKxHxWx1}
        Output: None
        '''
        raise NotImplementedError

    @abstractmethod
    def get_feature(self, *args, **kwargs):
        '''
        Help: Get the encoded feature according to the {slot, pixel, or voxel} information
        Input: Dict{slot, uv, xyz} # {BxK, BxNxKxHxWx2, BxNxKxDxHxWx3}
        Output: Dict{slot_feat, pixel_feat, voxel_feat} # {BxKxF, BxNxKxHxWxF, BxNxKxDxHxWxF}
        Caution: Call encode first, and then get_feature
        '''
        raise NotImplementedError


class Decoder(ABC, nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @abstractmethod
    def forward(self, *args, **kwargs):
        '''
        Help: Encode the features of images
        Input: Dict{xyz_coord, ray_dir, features} # (BxNxKxDxHxW)x{in_dims}
        Output: Dict{color, density} # (BxNxKxDxHxW)x{3,1}
        '''
        raise NotImplementedError

class Renderer(ABC, object):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @abstractmethod
    def render(self, *args, **kwargs):
        '''
        Help: Render the image based on volumetric rendering equation
        Input: Dict{cam_matrices, optional_inputs} # {BxNx4x4, etc}
            Optional Inputs:
                Dict{coarse_color, coarse_density} # BxNxKxD_xH_xW_x{3,1}
                features
                Decoder object
        Output: Dict{images, slot_images, optional_outputs} # {BxNxHxWx3, BxNxKxHxWx3, etc}
            Optional outputs:
                depth_maps # BxNxHxWx1
                occ_silhouettes (occluded) # BxNxKxHxWx1
                unocc_silhouettes (unoccluded) # BxNxKxHxWx1
                obj_masks # BxNxKxHxWx1
        '''
        raise NotImplementedError