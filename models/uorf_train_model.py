from itertools import chain

import torch
from torch import nn, optim
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import os
import time
from .projection import Projection
from torchvision.transforms import Normalize
from .model import Encoder, Decoder, SlotAttention, get_perceptual_net, raw2outputs, PixelEncoder, PixelDecoder, raw2transmittances, raw2colors, ImageEncoder, CentroidDecoder
from .position_encoding import position_encoding_image
import pdb
from util import util

import math

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe



class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, mode='dice'):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        # input shape: NxKxHxW
        # N, K, H, W = inputs.shape.as_list()
        inputs = inputs.flatten(2, 3)
        targets = targets.flatten(2, 3)
        # targets = 1 - F.threshold(1 - targets, 0.1, 0)

        if mode =='dice':
            intersection = (inputs * targets).sum(dim=-1)
            dice = (2. * intersection + smooth) / (inputs.sum(dim=-1) + targets.sum(dim=-1) + smooth)
            dice = dice.mean()

            loss = 1 - dice

        if mode=='tversky':
            alpha = 0.2
            beta = 0.8

            TP = (inputs * targets).sum()
            FP = ((1 - targets) * inputs).sum()
            FN = (targets * (1 - inputs)).sum()

            Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

            loss = 1 - Tversky

        return loss

class uorfTrainModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--num_slots', metavar='K', type=int, default=8, help='Number of supported slots')
        parser.add_argument('--z_dim', type=int, default=64, help='Dimension of individual z latent per slot')
        parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
        parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
        parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
        parser.add_argument('--supervision_size', type=int, default=64)
        parser.add_argument('--obj_scale', type=float, default=4.5, help='Scale for locality on foreground objects')
        parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
        parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
        parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
        parser.add_argument('--weight_percept', type=float, default=0.006)
        parser.add_argument('--percept_in', type=int, default=100)
        parser.add_argument('--no_locality_epoch', type=int, default=300)
        parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
        parser.add_argument('--input_size', type=int, default=64)
        parser.add_argument('--frustum_size', type=int, default=64)
        parser.add_argument('--frustum_size_fine', type=int, default=128)
        parser.add_argument('--attn_decay_steps', type=int, default=2e5)
        parser.add_argument('--coarse_epoch', type=int, default=600)
        parser.add_argument('--near_plane', type=float, default=6)
        parser.add_argument('--far_plane', type=float, default=20)
        parser.add_argument('--fixed_locality', action='store_true', help='enforce locality in world space instead of transformed view space')
        parser.add_argument('--gt_seg', action='store_true', help='replace slot attention with GT segmentation')
        parser.add_argument('--pixel_decoder', action='store_true', help='change decoder to pixel_decoder')
        parser.add_argument('--pixel_encoder', action='store_true', help='change encoder to pixel_encoder')
        parser.add_argument('--pixel_nerf', action='store_true', help='change model to pixelnerf')
        parser.add_argument('--mask_image_feature', action='store_true', help='mask image feature in pixelnerf decoder. if pixelencoder==False: this does not make sense')
        parser.add_argument('--mask_image', action='store_true', help='mask image in pixelnerf encoder. if pixelencoder==False: this does not make sense')
        parser.add_argument('--slot_repeat', action='store_true', help='put slot features repeatedly in the uORF decoder setting')
        parser.add_argument('--no_concatenate', action='store_true', help='do not concatenate object feature and pixel feature; instead, add them')
        parser.add_argument('--visualize_obj_feat', action='store_true', help='visualize object feature (after slot attention)')
        parser.add_argument('--input_no_loss', action='store_true', help='for the first n (=100) epochs (hyperparameter), do not include first image in the recon loss for learning geometry first')
        parser.add_argument('--bg_no_pixel', action='store_true', help='do not provide bg with pixel features, so that it can learn bg only from slot (object) features')
        parser.add_argument('--use_ray_dir', action='store_true', help='concatenate ray direction on the view. now only work on decoder (not pixel)')
        parser.add_argument('--weight_pixelfeat', action='store_true', help='weigh pixel_color and slot color using the transmittance from the input view')
        parser.add_argument('--silhouette_loss', action='store_true', help='enable silhouette loss')
        parser.add_argument('--reduce_latent_size', action='store_true', help='reduce latent size of pixel encoder. this option is not true for default and not configurable. TODO')
        parser.add_argument('--small_latent', action='store_true', help='reduce latent dim of decoder by 2')
        parser.add_argument('--shared_pixel_slot_decoder', action='store_true', help='use pixel decoder for slot decoder. need to check whether this reduce memory burden')
        parser.add_argument('--div_by_max', action='store_true', help='divide pixel feature importance by max per each ray')
        parser.add_argument('--learn_only_silhouette', action='store_true', help='loss function contains only silhouette loss')
        parser.add_argument('--focal_ratio', nargs='+', default=(350. / 320., 350. / 240.), help='set the focal ratio in projection.py')
        parser.add_argument('--density_n_color', action='store_true', help='separate density encoder and color encoder')
        parser.add_argument('--debug', action='store_true', help='debugging option')
        parser.add_argument('--density_no_pixel', action='store_true', help='density decoder no pixel feature')

        parser.add_argument('--rgb_loss_density_decoder', action='store_true', help='put rgb loss twice. on color decoder and density decoder')
        parser.add_argument('--restrict_world', action='store_true', help='restrict the world. the range is hyperparam')
        parser.add_argument('--reduce_color_decoder', action='store_true', help='reduce the color decoder for memory')
        parser.add_argument('--density_as_color_input', action='store_true', help='put density as an input of color decoder. this would be diff from both mask and transmittance')
        parser.add_argument('--mask_as_decoder_input', action='store_true', help='put mask as an input of both density and color decoder.')
        parser.add_argument('--dice_loss', action='store_true', help='convert silhouette loss from L1 to dice loss')

        parser.add_argument('--unified_decoder', action='store_true', help='do not divide color and density decoder.')
        parser.add_argument('--same_bg_fg_decoder', action='store_true', help='use same decoder architecture for bg and fg. currently only support unified decoder')
        parser.add_argument('--bilinear_mask', action='store_true', help='instead of nearest interpolation, use bilinear interpolation')
        parser.add_argument('--antialias', action='store_true', help='antialias for mask')

        parser.add_argument('--fine_encoder', action='store_true', help='do not interpolate the image')
        parser.add_argument('--resnet_encoder', action='store_true', help='use resnet encoder for slot features')

        parser.add_argument('--ray_after_density', action='store_true', help='concatenate ray dir after getting density')
        # ray after density require use_ray_dir
        parser.add_argument('--multiply_mask_pixelfeat', action='store_true', help='instead of concatenating mask on decoder input, apply mask on pixelfeat and concatenate')
        # multiply_mask_pixelfeat require mask_as_decoder_input
        parser.add_argument('--without_slot_feature', action='store_true', help='remove slot features')
        parser.add_argument('--uorf', action='store_true', help='remove pixel features')
        parser.add_argument('--color_after_density', action='store_true', help='unified decoder')
        parser.add_argument('--debug2', action='store_true')
        parser.add_argument('--mask_div_by_max', action='store_true')
        parser.add_argument('--kldiv_loss', action='store_true')
        parser.add_argument('--silhouette_l2_loss', action='store_true')
        parser.add_argument('--combine_masks', action='store_true', help='combine all the masks for pixel nerf')
        parser.add_argument('--weight_pixel_slot_mask', action='store_true')

        parser.add_argument('--silhouette_loss_nobackprop', action='store_true', help='only compute but do not back propagate silhouette loss')
        parser.add_argument('--bg_no_silhouette_loss', action='store_true', help='do not include bg in the silhouette loss')
        parser.add_argument('--silhouette_epoch', type=int, default=100)
        parser.add_argument('--mask_div_by_mean', action='store_true')
        parser.add_argument('--only_make_silhouette_not_delete', action='store_true')
        parser.add_argument('--weight_silhouette_loss', action='store_true')
        parser.add_argument('--threshold_silhouette_loss', action='store_true')
        parser.add_argument('--mask_image_slot', action='store_true')
        parser.add_argument('--nearest_interp', action='store_true', help='put nearest interp for pixel and so on')
        parser.add_argument('--slot_positional_encoding', action='store_true', help='put 2d positional encoding')
        parser.add_argument('--mask_image_feature_slot', action='store_true')
        parser.add_argument('--predict_centroid', action='store_true')
        parser.add_argument('--learn_only_centroid', action='store_true', help='loss function contains only centroid loss')

        parser.add_argument('--loss_centroid_margin', type=float, default=0.05, help='max margin for the centroid loss')
        parser.add_argument('--progressive_silhouette', action='store_true', help='epoch 0-10: learn only fg silhouette, epoch 10-20: learn silhouette and rgb, epoch 20- rgb')
        parser.add_argument('--learn_fg_first', action='store_true')
        parser.add_argument('--silhouette_expand', action='store_true', help='expand the silhouette for 1 pixel')
        parser.add_argument('--fg_only_delete_bg', action='store_true', help='only remove the background from bg')
        parser.add_argument('--debug3', action='store_true', help='20001')

        parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
                            dataset_mode='multiscenes', niter=2000, custom_lr=True, lr_policy='warmup')

        parser.set_defaults(exp_id='run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        print(self.isTrain, 'self.isTrain')
        self.loss_names = ['recon', 'perc']
        if self.opt.silhouette_loss:
            self.loss_names.append('silhouette')
        if self.opt.predict_centroid:
            self.loss_names.append('centroid')
        n = opt.n_img_each_scene
        self.visual_names = ['x{}'.format(i) for i in range(n)] + \
                            ['x_rec{}'.format(i) for i in range(n)] + \
                            ['slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                            ['unmasked_slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                            ['slot{}_attn'.format(k) for k in range(opt.num_slots)]
        self.perceptual_net = get_perceptual_net().cuda()
        self.vgg_norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
        self.projection = Projection(focal_ratio=opt.focal_ratio, device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        frustum_size_fine = [self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp]

        self.projection_fine = Projection(focal_ratio=opt.focal_ratio, device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size_fine, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        z_dim = opt.z_dim
        self.model_names = []
        self.parameters = []
        self.nets = []

        if self.opt.color_after_density:
            assert self.opt.unified_decoder

        self.interp_mode = 'nearest' if self.opt.nearest_interp else 'bilinear'

        # [uORF Encoder]
        if self.opt.resnet_encoder:
            self.netEncoder = networks.init_net(PixelEncoder(mask_image=self.opt.mask_image_slot, mask_image_feature=self.opt.mask_image_feature_slot, index_interp=self.interp_mode), gpu_ids=self.gpu_ids, init_type='None')
            self.parameters.append(self.netEncoder.parameters())
            self.nets.append(self.netEncoder)
            self.model_names.append('Encoder')
        else:
            self.netEncoder = networks.init_net(Encoder(3, z_dim=z_dim, bottom=opt.bottom), gpu_ids=self.gpu_ids, init_type='normal')
            self.parameters.append(self.netEncoder.parameters())
            self.nets.append(self.netEncoder)
            self.model_names.append('Encoder')

        if self.opt.slot_positional_encoding:
            # [Position encoding]
            p_dim = z_dim
            if self.opt.resnet_encoder:
                self.pos_emb = position_encoding_image(size=[opt.input_size//2, opt.input_size//2], num_pos_feats=p_dim // 2).to(
                self.device)
            else:
                self.pos_emb = position_encoding_image(size=[opt.input_size, opt.input_size],
                                                            num_pos_feats=p_dim // 2).to(self.device)
            slot_in_dim = z_dim + p_dim
            slot_slot_dim = p_dim
        else:
            slot_in_dim = z_dim
            slot_slot_dim = z_dim

        if self.opt.predict_centroid:
            p_dim = z_dim
            self.netSlotAttention_pos = networks.init_net(
                SlotAttention(num_slots=opt.num_slots, in_dim=p_dim + z_dim, slot_dim=p_dim, iters=opt.attn_iter,
                              gt_seg=opt.gt_seg), gpu_ids=self.gpu_ids, init_type='normal')
            self.nets.append(self.netSlotAttention_pos)
            self.parameters.append(self.netSlotAttention_pos.parameters())
            self.model_names.append('SlotAttention_pos')

            self.netCentroidDecoder = networks.init_net(
                CentroidDecoder(input_dim=p_dim,
                                z_dim=p_dim,
                                cam2pixel=self.projection.cam2spixel,
                                world2nss=self.projection.world2nss,
                                near=self.projection.near,
                                far=self.projection.far,
                                small_latent=self.opt.small_latent,
                                n_layers=opt.n_layer),
                gpu_ids=self.gpu_ids, init_type='xavier')
            self.parameters.append(self.netCentroidDecoder.parameters())
            self.nets.append(self.netCentroidDecoder)
            self.model_names.append('CentroidDecoder')

            if not self.opt.slot_positional_encoding:

                if self.opt.resnet_encoder:
                    self.pos_emb = position_encoding_image(size=[opt.input_size // 2, opt.input_size // 2],
                                                                num_pos_feats=p_dim // 2).to(self.device)
                else:
                    self.pos_emb = position_encoding_image(size=[opt.input_size, opt.input_size],
                                                                num_pos_feats=p_dim // 2).to(self.device)

        # [Slot attention]
        self.num_slots = opt.num_slots
        if not self.opt.without_slot_feature:
            self.netSlotAttention = networks.init_net(
                SlotAttention(num_slots=opt.num_slots, in_dim=slot_in_dim, slot_dim=slot_slot_dim, iters=opt.attn_iter, gt_seg=opt.gt_seg), gpu_ids=self.gpu_ids, init_type='normal')
            self.nets.append(self.netSlotAttention)
            self.parameters.append(self.netSlotAttention.parameters())
            self.model_names.append('SlotAttention')

        # [Pixel Encoder]
        if not self.opt.uorf:
            self.netPixelEncoder = networks.init_net(PixelEncoder(mask_image=self.opt.mask_image, mask_image_feature=self.opt.mask_image_feature, index_interp=self.interp_mode), gpu_ids=self.gpu_ids, init_type='None')
            self.parameters.append(self.netPixelEncoder.parameters())
            self.nets.append(self.netPixelEncoder)
            self.model_names.append('PixelEncoder')
            pixel_dim = 64
        if self.opt.uorf:
            pixel_dim = 0

        if self.opt.unified_decoder:
            # [Unified Decoder]
            self.netDecoder = networks.init_net(
                Decoder(n_freq=opt.n_freq, input_dim=6 * opt.n_freq + 3 + z_dim, pixel_dim=pixel_dim, z_dim=opt.z_dim,
                        n_layers=opt.n_layer,
                        locality_ratio=opt.obj_scale / opt.nss_scale, fixed_locality=opt.fixed_locality,
                        bg_no_pixel=self.opt.bg_no_pixel,
                        use_ray_dir=self.opt.use_ray_dir, small_latent=self.opt.small_latent, decoder_type='unified',
                        restrict_world=self.opt.restrict_world, mask_as_decoder_input=self.opt.mask_as_decoder_input,
                        ray_after_density=self.opt.ray_after_density, multiply_mask_pixelfeat=self.opt.multiply_mask_pixelfeat,
                        without_slot_feature=self.opt.without_slot_feature, same_bg_fg_decoder=self.opt.same_bg_fg_decoder,
                        color_after_density=self.opt.color_after_density, no_concatenate=self.opt.no_concatenate,
                        weight_pixel_slot_mask=self.opt.weight_pixel_slot_mask),
                gpu_ids=self.gpu_ids, init_type='xavier')
            self.parameters.append(self.netDecoder.parameters())
            self.nets.append(self.netDecoder)
            self.model_names.append('Decoder')

        if self.isTrain:  # only defined during training time
            self.optimizer = optim.Adam(chain(*self.parameters), lr=opt.lr)
            self.optimizers = [self.optimizer]

        self.L2_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.dice_loss = DiceLoss()
        if self.opt.silhouette_expand:
            self.maxpool2d = nn.MaxPool2d(3, stride=1, padding=1)

    def setup(self, opt):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_{}'.format(opt.load_iter) if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.x = input['img_data'].to(self.device)
        self.cam2world = input['cam2world'].to(self.device)
        if not self.opt.fixed_locality:
            self.cam2world_azi = input['azi_rot'].to(self.device)

        if 'masks' in input.keys():
            bg_masks = input['bg_mask'][0:1].to(self.device)
            obj_masks = input['obj_masks'][0:1].to(self.device)
            masks = torch.cat([bg_masks, obj_masks], dim=1)
            if self.opt.combine_masks:
                obj_mask = torch.zeros_like(obj_masks[:, 0, ...])
                for i in range(obj_masks.shape[1]):
                    obj_mask = obj_mask | obj_masks[:, i, ...]
                masks = torch.cat([bg_masks, obj_mask.unsqueeze(1)], dim=1)
            else:
                masks = masks[:, :self.opt.num_slots, ...]
            if self.opt.bilinear_mask:
                mode_ = 'bilinear'
            else:
                mode_ = 'nearest'

            masks = F.interpolate(masks.float(), size=[self.opt.input_size, self.opt.input_size], mode=mode_)
            self.masks = masks.flatten(2, 3)
            bg_masks = input['bg_mask'].to(self.device)
            obj_masks = input['obj_masks'].to(self.device)
            masks = torch.cat([bg_masks, obj_masks], dim=1)
            if self.opt.combine_masks:
                obj_mask = torch.zeros_like(obj_masks[:, 0, ...])
                for i in range(obj_masks.shape[1]):
                    obj_mask = obj_mask | obj_masks[:, i, ...]
                masks = torch.cat([bg_masks, obj_mask.unsqueeze(1)], dim=1)
            else:
                masks = masks[:, :self.opt.num_slots, ...]
            masks = F.interpolate(masks.float(), size=[128, 128], mode=mode_)
            self.silhouette_masks_fine = masks
            masks = F.interpolate(masks.float(), size=[64, 64], mode=mode_)
            self.silhouette_masks = masks

        # [Compute GT segment centroid]
        if self.opt.predict_centroid:
            frustum_size = self.projection.frustum_size
            _masks = F.interpolate(masks.float(), size=frustum_size[0:2], mode='nearest').flatten(2, 3)  # NxKx(HxW)
            x = torch.arange(frustum_size[0]).to(self.device)  # frustum_size
            y = torch.arange(frustum_size[1]).to(self.device)  # frustum_size
            x, y = torch.meshgrid([x, y])  # frustum_sizexfrustum_size
            x = x.flatten()[None, None]  # NxKx(frustum_size)^2
            y = y.flatten()[None, None]  # NxKx(frustum_size)^2
            center_x = (_masks * x).sum(-1) / (_masks.sum(-1) + 1e-12)  # NxK
            center_y = (_masks * y).sum(-1) / (_masks.sum(-1) + 1e-12)  # NxK
            self.segment_masks = _masks
            self.segment_centers = torch.stack([center_x, center_y], dim=-1)  # NxKx2


    def forward(self, epoch=0):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.weight_percept = self.opt.weight_percept if epoch >= self.opt.percept_in else 0
        self.loss_recon = 0
        self.loss_perc = 0
        self.loss_silhouette = 0
        dev = self.x[0:1].device
        nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()

        # Replace slot attention with GT segmentations
        if self.opt.gt_seg:
            masks = self.masks
            attn_bg, attn_fg = masks[:, 0:1], masks[:, 1:]
            attn = torch.cat([attn_bg, attn_fg], dim=1)
        else:
            attn = None

        if self.opt.resnet_encoder:
            attn = attn.view([1, self.opt.num_slots, self.opt.input_size, self.opt.input_size])
            attn = attn.transpose(0, 1)
            attn = F.interpolate(attn, size=self.opt.input_size//2, mode='nearest')
            attn = attn.transpose(0, 1).flatten(2, 3)

        # Encoding images
        if self.opt.mask_image_slot or self.opt.mask_image_feature_slot:
            assert self.opt.resnet_encoder
            if self.opt.resnet_encoder:
                masks_slot = attn
            else:
                masks_slot = self.masks.squeeze(0)
        else:
            masks_slot = None

        feature_map = self.netEncoder(F.interpolate(self.x[0:1], size=self.opt.input_size, mode=self.interp_mode), masks=masks_slot)  # BxCxHxW # 0506
        feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC

        if self.opt.slot_positional_encoding:
            pos_embed = self.pos_emb  # self.netPositionEmbedding()
            pos_feat = torch.cat([pos_embed, feat], dim=-1)
            feat = pos_feat

        # Slot Attention
        if self.opt.pixel_nerf:
            if self.opt.combine_masks:
                K = 2
                z_slots = torch.zeros([K, 64])
                attn = torch.zeros([K, self.opt.input_size**2])
            else:
                K = self.opt.num_slots
                z_slots = torch.zeros([K, 64])
                attn = torch.zeros([K, self.opt.input_size**2])
        else:
            z_slots, attn = self.netSlotAttention(feat, masks=attn)  # 1xKxC, 1xKxN
            z_slots, attn = z_slots.squeeze(0), attn.squeeze(0)  # KxC, KxN (N = HxW)
            K = attn.shape[0]
            if self.opt.debug:
                z_slots *= 0

        # Pixel Encoder Forward (to get feature values in pixel coordinates (uv), call pixelEncoder.index(uv), not forward)
        if not self.opt.uorf:
            masks = self.masks.squeeze(0) if self.opt.mask_image or self.opt.mask_image_feature else None
            feature_map_pixel = self.netPixelEncoder(F.interpolate(self.x[0:1], size=self.opt.input_size, mode=self.interp_mode), masks = masks)

        # Get rays and coordinates
        cam2world = self.cam2world
        N = cam2world.shape[0]

        if self.opt.stage == 'coarse':
            W, H, D = self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            x = F.interpolate(self.x, size=self.opt.supervision_size, mode=self.interp_mode)
            self.z_vals, self.ray_dir = z_vals, ray_dir
            frustum_size = torch.Tensor(self.projection.frustum_size).to(self.device)
        else:
            W, H, D = self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp
            start_range = self.opt.frustum_size_fine - self.opt.render_size
            rs = self.opt.render_size
            frus_nss_coor, z_vals, ray_dir = self.projection_fine.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor.view([N, D, H, W, 3]), z_vals.view([N, H, W, D]), ray_dir.view([N, H, W, 3])
            H_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            W_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            frus_nss_coor_, z_vals_, ray_dir_ = frus_nss_coor[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], z_vals[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], ray_dir[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor_.flatten(0, 3), z_vals_.flatten(0, 2), ray_dir_.flatten(0, 2)
            x = F.interpolate(self.x, size=128, mode=self.interp_mode)
            x = x[:, :, H_idx:H_idx + rs, W_idx:W_idx + rs]
            self.silhouette_masks = self.silhouette_masks_fine[..., H_idx:H_idx + rs, W_idx:W_idx + rs]
            self.z_vals, self.ray_dir = z_vals, ray_dir
            W, H, D = self.opt.render_size, self.opt.render_size, self.opt.n_samp
            frustum_size = torch.Tensor(self.projection_fine.frustum_size).to(self.device)

        if self.opt.use_ray_dir:
            # ray_dir_input = ray_dir.view([N, H, W, 3]).unsqueeze(1).expand(-1, D, -1, -1, -1)

            ray_dir_input = ray_dir.view([N, H, W, 3])[0].unsqueeze(0) # 1xHxWx3
            ray_dir_input /= torch.norm(ray_dir_input, dim=-1).unsqueeze(-1) # 1xHxWx3
            ray_dir_input = ray_dir_input.squeeze(0).view([H*W, 3]).unsqueeze(-1) # (HxW)x3x1
            cam2world_raydir = self.cam2world[:, :3, :3].unsqueeze(1) # Nx4x4 -> Nx1x3x3
            ray_dir_input = torch.matmul(cam2world_raydir, ray_dir_input) # (j x 1 x n x m, k x m x p)-->(j x k x n x p)
            # Nx1x3x3, (HxW)x3x1 --> Nx(HxW)x3x1
            ray_dir_input = ray_dir_input.view([N, H, W, 3]).unsqueeze(1).expand(-1, D, -1, -1, -1)

            ray_dir_input = ray_dir_input.flatten(0, 3)
        else:
            ray_dir_input = None

        sampling_coor_fg = frus_nss_coor[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
        sampling_coor_bg = frus_nss_coor  # Px3

        W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp

        # Get pixel feature if using [pixel Encoder]
        if self.opt.pixel_encoder or self.opt.mask_as_decoder_input:
            # get cam matrices
            # W, H, D = self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp # what is this?
            # self.cam2spixel = self.projection.cam2spixel
            # self.world2nss = self.projection.world2nss
            frustum_size_ = torch.Tensor(self.projection.frustum_size).to(self.device)
            # print(frus_nss_coor.max(), frus_nss_coor.min(), frus_nss_coor.mean(), frus_nss_coor.std(), 'frus_nss_coor max, min, mean, std')

            self.cam2spixel = self.projection_fine.cam2spixel
            self.world2nss = self.projection_fine.world2nss
            frustum_size = torch.Tensor(self.projection_fine.frustum_size).to(self.device)

            # construct uv in the first image coordinates
            cam02world = cam2world[0:1] # 1x4x4
            world2cam0 = cam02world.inverse() # 1x4x4
            nss2world = self.world2nss.inverse() # 1x4x4
            frus_nss_coor = torch.cat([frus_nss_coor, torch.ones_like(frus_nss_coor[:, 0].unsqueeze(1))], dim=-1) # Px4
            frus_world_coor = torch.matmul(nss2world[None, ...], frus_nss_coor[None, ..., None]) # 1xPx4x1
            frus_cam0_coor = torch.matmul(world2cam0[None, ...], frus_world_coor) #1x1x4x4, 1x(NxDxHxW)x4x1 -> 1x(NxDxHxW)x4x1 # TODO: check this
            pixel_cam0_coor = torch.matmul(self.cam2spixel[None, ...], frus_cam0_coor) # 1x1x4x4, 1x(NxDxHxW)x4x1
            pixel_cam0_coor = pixel_cam0_coor.squeeze(-1) # 1x(NxDxHxW)x4
            # print(pixel_cam0_coor[..., 2].max(), pixel_cam0_coor[..., 2].min(), pixel_cam0_coor[..., 2].mean(), pixel_cam0_coor[..., 2].std(), 'pixel_cam0_coor statistics')
            uv = pixel_cam0_coor[:, :, 0:2]/ pixel_cam0_coor[:, :, 2].unsqueeze(-1) # 1x(NxDxHxW)x2
            uv = (uv + 0.)/frustum_size[0:2][None, None, :] * 2 - 1 # nomalize to fit in with [-1, 1] grid # TODO: check this

            if self.opt.pixel_encoder:
                if self.opt.mask_image or self.opt.mask_image_feature:
                    uv = uv.expand(K, -1, -1) # 1x(NxDxHxW)x2 -> Kx(NxDxHxW)x2
                    pixel_feat = self.netPixelEncoder.index(uv)  # Kx(NxDxHxW)x2 -> KxCx(NxDxHxW)
                    pixel_feat = pixel_feat.transpose(1, 2)
                else:
                    pixel_feat = self.netPixelEncoder.index(uv)  # 1x(NxDxHxW)x2 -> 1xCx(NxDxHxW)
                    pixel_feat = pixel_feat.transpose(1, 2)  # 1x(NxDxHxW)xC
                    pixel_feat = pixel_feat.expand(K, -1, -1) # Kx(NxDxHxW)xC
                    uv = uv.expand(K, -1, -1)  # 1x(NxDxHxW)x2 -> Kx(NxDxHxW)x2
            else:
                uv = uv.expand(K, -1, -1)

        # Decoder centroid
        if self.opt.predict_centroid:
            if not self.opt.slot_positional_encoding:
                pos_embed = self.pos_emb  # self.netPositionEmbedding()
                pos_feat = torch.cat([pos_embed, feat], dim=-1)
            else:
                pos_feat = feat

            # Get slot features
            p_slots, _ = self.netSlotAttention_pos(pos_feat, masks=attn.unsqueeze(0))  # 1xKxC

            # Predict centroid
            self.netCentroidDecoder.frustum_size = frustum_size_
            self.netCentroidDecoder.cam2world = self.cam2world
            _, centroid_nss, centroid_pixel = self.netCentroidDecoder(p_slots.squeeze(0))

            # Transform sampling coordinates
            sampling_coor_fg = self.netCentroidDecoder.transform_coords(coords=sampling_coor_fg, centroids=centroid_nss)

            # Compute centroid loss
            self.netCentroidDecoder.x = self.x
            self.loss_centroid = self.netCentroidDecoder.centroid_loss(
                centroid_pixel=centroid_pixel, margin=self.opt.loss_centroid_margin,
                segment_centers=self.segment_centers, segment_masks=self.segment_masks, epoch=epoch)

            if self.opt.learn_only_centroid:
                return

        if self.opt.mask_as_decoder_input or self.opt.weight_pixel_slot_mask:
            silhouette0 = self.silhouette_masks[0:1].transpose(0, 1)  # Kx1xHxW
            # uv = uv.unsqueeze(1)  # Kx1x(NxDxHxW)x2
            silhouettes_for_density = F.grid_sample(silhouette0, uv.unsqueeze(1), mode='bilinear',
                                                  padding_mode='zeros', )  # Kx1(C)x1x(NxDxHxW)
            silhouettes_for_density = silhouettes_for_density.flatten(0, 3).view(K, N, D, H, W).permute([1, 2, 0, 3, 4])  # NxDxKxHxW  # NxDxKxHxW
        else:
            silhouettes_for_density = None

        if self.opt.debug2:
            pixel_feat *= 0
        if self.opt.uorf:
            pixel_feat = None

        if self.opt.weight_pixel_slot_mask:
            # pixelfeat: Kx(NxDxHxW)xC
            # slotfeat: KxC
            # silhouettes_for_density = None
            pass
        if self.opt.unified_decoder:
            render_bg = True
            if self.opt.learn_fg_first:
                render_bg = False if epoch < 60 else True
            raws, masked_raws, unmasked_raws, masks_for_silhouette = \
                self.netDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0, pixel_feat=pixel_feat,
                                       ray_dir_input=ray_dir_input, decoder_type='unified',
                                       silhouettes=silhouettes_for_density, render_bg=render_bg)
            # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
            raws = raws.view([N, D, H, W, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
            masked_raws = masked_raws.view([K, N, D, H, W, 4])
            unmasked_raws = unmasked_raws.view([K, N, D, H, W, 4])
            masks_for_silhouette = masks_for_silhouette.view([K, N, D, H, W])

            z_vals, ray_dir = self.z_vals, self.ray_dir
            # raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
            rgb_map, depth_map, _, _, _, silhouettes = raw2outputs(raws, z_vals, ray_dir, return_silhouettes=masks_for_silhouette)
            self.silhouettes = silhouettes

        rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
        x_recon = rendered * 2 - 1

        if self.opt.learn_only_silhouette:
            self.loss_recon *= 0.
            self.loss_perc *= 0.
        else:
            if self.opt.learn_fg_first:
                if epoch > 60:
                    self.opt.learn_fg_first = False
                obj_masks = self.silhouette_masks[:, 1:, :, :] # NxKxHxW
                obj_mask = torch.zeros_like(obj_masks[:, 0, ...])
                for i in range(obj_masks.shape[1]):
                    obj_mask = obj_mask + obj_masks[:, i, ...]
                obj_mask = obj_mask.unsqueeze(1)
                self.loss_recon = self.L2_loss(x_recon, x * obj_mask + -(1-obj_mask)*torch.ones_like(x)) #Nx3xHxW
            else:
                self.loss_recon = self.L2_loss(x_recon, x)
            x_norm, rendered_norm = self.vgg_norm((x + 1) / 2), self.vgg_norm(rendered)
            rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
            self.loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)

        if self.opt.silhouette_loss:
            # print(torch.sum(torch.sum(self.silhouette_masks, dim=1).flatten(1, 2), dim=1), 'mask sum') this is 4096
            # print('opt.silhouette_loss')
            idx = 1 if self.opt.bg_no_silhouette_loss else 0
            # print(idx, 'idx')
            silhouette_masks_ = self.silhouette_masks[:, idx:, ...]
            silhouettes_ = self.silhouettes[:, idx:, ...]
            if self.opt.silhouette_expand:
                assert self.opt.bg_no_silhouette_loss
                silhouette_masks_ = self.maxpool2d(silhouette_masks_)
            if self.opt.debug3:
                for _ in range(5):
                    silhouette_masks_ = self.maxpool2d(silhouette_masks_)
            if self.opt.fg_only_delete_bg:
                # print(silhouette_masks_.max(), silhouette_masks_.min(), 'max', 'min') # 1, 0
                silhouettes_ *= (1 - silhouette_masks_)
                silhouette_masks_ *= 0
            if self.opt.only_make_silhouette_not_delete:
                silhouettes_ *= silhouette_masks_
            if self.opt.mask_div_by_mean:
                # silhouettes_ /= (1e-10 + silhouettes_.flatten(2, 3).mean(dim=-1).unsqueeze(-1).unsqueeze(-1))
                # silhouettes_masks_ /= (1e-10 + silhouettes_masks_.flatten(2, 3).mean(dim=-1).unsqueeze(-1).unsqueeze(-1))
                silhouettes_ /= (1e-10 + silhouettes_.flatten(0, 3).mean())
                silhouette_masks_ /= (1e-10 + silhouette_masks_.flatten(0, 3).mean())
            if self.opt.mask_div_by_max:
                # self.silhouettes /= (1e-10+self.silhouettes.flatten(2, 3).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
                silhouettes_ /= (1e-10 + silhouettes_.flatten(0, 3).max(dim=-1)[0])
            if self.opt.dice_loss:
                # self.loss_silhouette = self.dice_loss(silhouette_masks_, silhouettes_, mode='tversky') # NxKxHxW # 128x128
                self.loss_silhouette = self.dice_loss(silhouette_masks_, silhouettes_, mode='dice')
                # print('dice_loss', self.loss_silhouette)
            elif self.opt.kldiv_loss:
                targets = silhouette_masks_
                inputs = silhouettes_
            elif self.opt.silhouette_l2_loss:
                # print('silhouette_l2_loss')
                self.loss_silhouette = self.L2_loss(silhouette_masks_, silhouettes_) # NxKxHxW
            else:
                self.loss_silhouette = self.L1_loss(silhouette_masks_, silhouettes_) # NxKxHxW
            if self.opt.weight_silhouette_loss:
                mask_area = torch.sum(silhouette_masks_.flatten(2, 3), dim=-1)
                mask_area_complement = torch.sum((1-silhouette_masks_).flatten(2, 3), dim=-1)
                silhouettes_mask = silhouettes_ * silhouette_masks_
                silhouettes_mask_complement = (1 - silhouettes_) * (1 - silhouette_masks_)
                loss = 0
                for i in range(silhouettes_.shape[0]):
                    for j in range(silhouettes_.shape[1]):
                        loss += self.L2_loss(silhouettes_mask[i][j], silhouette_masks_[i][j]) / mask_area[i, j]
                        loss += self.L2_loss(silhouettes_mask_complement[i][j], (1-silhouette_masks_[i][j])) / mask_area_complement[i][j]
                self.loss_silhouette = loss
            if (self.opt.uorf or self.opt.color_after_density) and self.opt.silhouette_epoch!=0:
                if epoch < self.opt.silhouette_epoch:
                    self.loss_silhouette = 0

        with torch.no_grad():
            attn = attn.detach().cpu()  # KxN
            H_, W_ = feature_map.shape[2], feature_map.shape[3]
            attn = attn.view(self.opt.num_slots, 1, H_, W_)
            if H_ != H:
                attn = F.interpolate(attn, size=[H, W], mode='nearest')
            for i in range(self.opt.n_img_each_scene):
                setattr(self, 'x_rec{}'.format(i), x_recon[i])
                setattr(self, 'x{}'.format(i), x[i])
            setattr(self, 'masked_raws', masked_raws.detach())
            setattr(self, 'unmasked_raws', unmasked_raws.detach())
            setattr(self, 'attn', attn)


    def compute_visuals(self):
        with torch.no_grad():
            _, N, D, H, W, _ = self.masked_raws.shape
            masked_raws = self.masked_raws  # KxNxDxHxWx4
            unmasked_raws = self.unmasked_raws  # KxNxDxHxWx4
            for k in range(self.num_slots):
                raws = masked_raws[k]  # NxDxHxWx4
                z_vals, ray_dir = self.z_vals, self.ray_dir
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _, _, _, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i])

                raws = unmasked_raws[k]  # (NxDxHxW)x4
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _, _, _, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'unmasked_slot{}_view{}'.format(k, i), x_recon[i])

                setattr(self, 'slot{}_attn'.format(k), self.attn[k] * 2 - 1)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        loss = self.loss_recon + self.loss_perc
        if self.opt.silhouette_loss and self.loss_silhouette != 0:
            loss += self.loss_silhouette * 100
        if self.opt.rgb_loss_density_decoder:
            loss += self.loss_recon_density
        if self.opt.predict_centroid:
            loss += self.loss_centroid
        loss.backward()
        self.loss_perc = self.loss_perc / self.weight_percept if self.weight_percept > 0 else self.loss_perc

    def optimize_parameters(self, ret_grad=False, epoch=0):
        """Update network weights; it will be called in every training iteration."""
        self.forward(epoch)
        for opm in self.optimizers:
            opm.zero_grad()
        self.backward()
        avg_grads = []
        layers = []
        if ret_grad:
            # for n, p in chain(self.netEncoder.named_parameters(), self.netSlotAttention.named_parameters(), self.netDecoder.named_parameters()):
            for n, p in chain(*[x.named_parameters() for x in self.nets]):
                if p.grad is not None and "bias" not in n:
                    with torch.no_grad():
                        layers.append(n)
                        avg_grads.append(p.grad.abs().mean().cpu().item())
        for opm in self.optimizers:
            opm.step()
        return layers, avg_grads

    def save_networks(self, surfix):
        """Save all the networks to the disk.

        Parameters:
            surfix (int or str) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        super().save_networks(surfix)
        for i, opm in enumerate(self.optimizers):
            save_filename = '{}_optimizer_{}.pth'.format(surfix, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(opm.state_dict(), save_path)

        for i, sch in enumerate(self.schedulers):
            save_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(sch.state_dict(), save_path)

    def load_networks(self, surfix):
        """Load all the networks from the disk.

        Parameters:
            surfix (int or str) -- current epoch; used in he file name '%s_net_%s.pth' % (epoch, name)
        """
        super().load_networks(surfix)

        if self.isTrain:
            for i, opm in enumerate(self.optimizers):
                load_filename = '{}_optimizer_{}.pth'.format(surfix, i)
                load_path = os.path.join(self.save_dir, load_filename)
                print('loading the optimizer from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                opm.load_state_dict(state_dict)

            for i, sch in enumerate(self.schedulers):
                load_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
                load_path = os.path.join(self.save_dir, load_filename)
                print('loading the lr scheduler from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                sch.load_state_dict(state_dict)


if __name__ == '__main__':
    pass