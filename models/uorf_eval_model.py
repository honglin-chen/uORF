from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import os
import time
from .projection import Projection
from .model import Encoder, Decoder, SlotAttention, get_perceptual_net, raw2outputs, PixelEncoder, PixelDecoder, raw2transmittances, raw2colors
import pdb
from util import util
# from .model import Encoder, Decoder, SlotAttention, raw2outputs
from util.util import AverageMeter
from sklearn.metrics import adjusted_rand_score
import lpips
from piq import ssim as compute_ssim
from piq import psnr as compute_psnr
from .evaluate_mesh import *
import h5py
from pytorch3d.loss import chamfer_distance

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        # input shape: NxKxHxW
        # N, K, H, W = inputs.shape.as_list()
        inputs = inputs.flatten(2, 3)
        targets = targets.flatten(2, 3)

        intersection = (inputs * targets).sum(dim=-1)
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice = dice.sum()

        return 1 - dice

class uorfEvalModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--num_slots', metavar='K', type=int, default=4, help='Number of supported slots')
        parser.add_argument('--z_dim', type=int, default=64, help='Dimension of individual z latent per slot')
        parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
        parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
        parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
        parser.add_argument('--obj_scale', type=float, default=4.5, help='Scale for locality on foreground objects')
        parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
        parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
        parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
        parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
        parser.add_argument('--input_size', type=int, default=128)
        parser.add_argument('--frustum_size', type=int, default=64, help='Size of rendered images')
        parser.add_argument('--near_plane', type=float, default=1)
        parser.add_argument('--far_plane', type=float, default=15)
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
        parser.add_argument('--focal_ratio', nargs='+', default=(0.9605, 0.9605), help='set the focal ratio in projection.py')
        parser.add_argument('--density_n_color', action='store_true', help='separate density encoder and color encoder')
        parser.add_argument('--debug', action='store_true', help='debugging option')
        parser.add_argument('--extract_mesh', action='store_true', help='construct mesh')

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


        parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
                            dataset_mode='multiscenes', niter=1200, custom_lr=True, lr_policy='warmup')

        # parser.set_defaults(exp_id='run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))

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
        self.loss_names = ['ari', 'fgari', 'nvari', 'psnr', 'ssim', 'lpips']
        n = opt.n_img_each_scene
        self.visual_names = ['input_image',] + ['gt_novel_view{}'.format(i+1) for i in range(n-1)] + \
                            ['x_rec{}'.format(i) for i in range(n)] + \
                            ['gt_mask{}'.format(i) for i in range(n)] + \
                            ['render_mask{}'.format(i) for i in range(n)]
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
        self.projection = Projection(focal_ratio=opt.focal_ratio, device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        self.projection_coarse = Projection(focal_ratio=opt.focal_ratio, device=self.device, nss_scale=opt.nss_scale,
                                            frustum_size=[64, 64, 64], near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        if self.opt.extract_mesh:
            print('extract_mesh mode')
            self.projection_mesh = Projection(focal_ratio=opt.focal_ratio, device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size, extract_mesh=True)

        z_dim = opt.z_dim
        self.parameters = []
        self.nets = []
        self.model_names = []

        # [uORF Encoder]
        if self.opt.resnet_encoder:
            self.netEncoder = networks.init_net(PixelEncoder(mask_image=False, mask_image_feature=False),
                                                gpu_ids=self.gpu_ids, init_type='None')
            self.parameters.append(self.netEncoder.parameters())
            self.nets.append(self.netEncoder)
            self.model_names.append('Encoder')
        else:
            self.netEncoder = networks.init_net(Encoder(3, z_dim=z_dim, bottom=opt.bottom), gpu_ids=self.gpu_ids,
                                                init_type='normal')
            self.parameters.append(self.netEncoder.parameters())
            self.nets.append(self.netEncoder)
            self.model_names.append('Encoder')

        # [Slot attention]
        self.num_slots = opt.num_slots
        if not self.opt.without_slot_feature:
            self.netSlotAttention = networks.init_net(
                SlotAttention(num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter,
                              gt_seg=opt.gt_seg), gpu_ids=self.gpu_ids, init_type='normal')
            self.nets.append(self.netSlotAttention)
            self.parameters.append(self.netSlotAttention.parameters())
            self.model_names.append('SlotAttention')

        # [Pixel Encoder]
        if not self.opt.uorf:
            self.netPixelEncoder = networks.init_net(
                PixelEncoder(mask_image=self.opt.mask_image, mask_image_feature=self.opt.mask_image_feature),
                gpu_ids=self.gpu_ids, init_type='None')
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
                        ray_after_density=self.opt.ray_after_density,
                        multiply_mask_pixelfeat=self.opt.multiply_mask_pixelfeat,
                        without_slot_feature=self.opt.without_slot_feature,
                        same_bg_fg_decoder=self.opt.same_bg_fg_decoder,
                        color_after_density=self.opt.color_after_density, no_concatenate=self.opt.no_concatenate,
                        weight_pixel_slot_mask=self.opt.weight_pixel_slot_mask),
                gpu_ids=self.gpu_ids, init_type='xavier')
            self.parameters.append(self.netDecoder.parameters())
            self.nets.append(self.netDecoder)
            self.model_names.append('Decoder')
        else:
            # [Density Decoder]
            if self.opt.density_no_pixel:
                pixel_dim = None
            self.netDensityDecoder = networks.init_net(
                Decoder(n_freq=opt.n_freq, input_dim=6 * opt.n_freq + 3 + z_dim, pixel_dim=pixel_dim, z_dim=opt.z_dim,
                        n_layers=opt.n_layer,
                        locality_ratio=opt.obj_scale / opt.nss_scale, fixed_locality=opt.fixed_locality,
                        bg_no_pixel=self.opt.bg_no_pixel,
                        use_ray_dir=False, small_latent=self.opt.small_latent, decoder_type='density',
                        restrict_world=self.opt.restrict_world, mask_as_decoder_input=self.opt.mask_as_decoder_input),
                gpu_ids=self.gpu_ids, init_type='xavier')

            self.parameters.append(self.netDensityDecoder.parameters())
            self.nets.append(self.netDensityDecoder)
            self.model_names.append('DensityDecoder')

            # [Color Decoder]
            pixel_dim = 64
            self.netColorDecoder = networks.init_net(
                Decoder(n_freq=opt.n_freq, input_dim=6 * opt.n_freq + 3 + z_dim, pixel_dim=pixel_dim, z_dim=opt.z_dim,
                        n_layers=opt.n_layer,
                        locality_ratio=opt.obj_scale / opt.nss_scale, fixed_locality=opt.fixed_locality,
                        bg_no_pixel=self.opt.bg_no_pixel,
                        use_ray_dir=self.opt.use_ray_dir, small_latent=self.opt.small_latent, decoder_type='color',
                        reduce_color_decoder=self.opt.reduce_color_decoder, restrict_world=self.opt.restrict_world,
                        density_as_color_input=self.opt.density_as_color_input,
                        mask_as_decoder_input=self.opt.mask_as_decoder_input), gpu_ids=self.gpu_ids, init_type='xavier')
            self.parameters.append(self.netColorDecoder.parameters())
            self.nets.append(self.netColorDecoder)
            self.model_names.append('ColorDecoder')

        self.L2_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        self.LPIPS_loss = lpips.LPIPS().cuda()
        self.dice_loss = DiceLoss()

        # [Extract Mesh]
        if self.opt.extract_mesh:
            self.opt.gt_seg = True
            self.opt.pixel_encoder = True
            self.opt.mask_image_feature = True
            self.opt.mask_image = True
            self.opt.use_ray_dir = True
            self.opt.silhouette_loss = True
            self.opt.weight_pixelfeat = True
            self.opt.bg_no_pixel = True

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
        self.input_paths = input['paths']
        if not self.opt.fixed_locality:
            self.cam2world_azi = input['azi_rot'].to(self.device)

        self.image_paths = input['paths']
        if 'masks' in input:
            self.gt_masks = input['masks']
            self.mask_idx = input['mask_idx']
            self.fg_idx = input['fg_idx']
            self.obj_idxs = input['obj_idxs']  # NxKxHxW

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
            if self.opt.antialias:
                antialias_ = True
            else:
                antialias_ = False

            masks = F.interpolate(masks.float(), size=[self.opt.input_size, self.opt.input_size], mode='nearest')
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


    def forward(self, epoch=0, frus_nss_coor=None):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        dev = self.x[0:1].device
        nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()

        # Replace slot attention with GT segmentations
        if self.opt.gt_seg:
            masks = self.masks
            attn_bg, attn_fg = masks[:, 0:1], masks[:, 1:]
            attn = torch.cat([attn_bg, attn_fg], dim=1)

        if self.opt.resnet_encoder:
            attn = attn.view([1, self.opt.num_slots, self.opt.input_size, self.opt.input_size])
            attn = attn.transpose(0, 1)
            attn = F.interpolate(attn, size=self.opt.input_size//2, mode='nearest')
            attn = attn.transpose(0, 1).flatten(2, 3)

        # Encoding images
        feature_map = self.netEncoder(F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False))  # BxCxHxW
        feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC

        # Slot Attention
        if self.opt.pixel_nerf:
            if self.opt.combine_masks:
                K = 2
                z_slots = torch.zeros([K, 64])
                attn = torch.zeros([K, self.opt.input_size ** 2])
            else:
                K = self.opt.num_slots
                z_slots = torch.zeros([K, 64])
                attn = torch.zeros([K, self.opt.input_size ** 2])
        else:
            z_slots, attn = self.netSlotAttention(feat, masks=attn)  # 1xKxC, 1xKxN
            z_slots, attn = z_slots.squeeze(0), attn.squeeze(0)  # KxC, KxN (N = HxW)
            K = attn.shape[0]
            if self.opt.debug:
                z_slots *= 0
                print('z_slots are zero now')

        # Pixel Encoder Forward (to get feature values in pixel coordinates (uv), call pixelEncoder.index(uv), not forward)
        if not self.opt.uorf:
            masks = self.masks.squeeze(0) if self.opt.mask_image or self.opt.mask_image_feature else None
            feature_map_pixel = self.netPixelEncoder(
                F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False), masks=masks)

        # Get rays and coordinates
        cam2world = self.cam2world
        N = cam2world.shape[0]
        W, H, D = self.projection.frustum_size
        scale = H // self.opt.render_size
        if self.opt.extract_mesh:
            frus_nss_coor, z_vals, ray_dir = self.projection_mesh.construct_sampling_coor(cam2world, partitioned=True)
        else:
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world, partitioned=True)
        # 4x(NxDx(H/2)x(W/2))x3, 4x(Nx(H/2)x(W/2))xD, 4x(Nx(H/2)x(W/2))x3
        x = self.x
        x_recon = torch.zeros([N, 3, H, W], device=dev)
        rendered = torch.zeros([N, 3, H, W], device=dev)
        masked_raws = torch.zeros([K, N, D, H, W, 4], device=dev)
        unmasked_raws = torch.zeros([K, N, D, H, W, 4], device=dev)
        transmittance_samples = torch.zeros([N, H, W, D], device=dev)
        raws = torch.zeros([N, D, H, W, 4], device=dev)
        raws_density = torch.zeros([N, D, H, W, 4], device=dev)
        masks_for_silhouette_density = torch.zeros([K, N, D, H, W], device=dev)
        uvw = torch.zeros([1, N, D, H, W, 3], device=dev) # 1x(NxDxHxW)x3

        z_vals_not_partitioned = torch.zeros([N, H, W, D], device=dev)
        ray_dir_not_partitioned = torch.zeros([N, H, W, 3], device=dev)

        silhouettes = torch.zeros([N, K, H, W], device=dev)

        # if self.opt.extract_mesh:
        #     raw_masks_density = torch.zeros([K, N, D, H, W, 1])

        # print(K, N, D, H, W, 'K, N, D, H, W')

        raw_masks_density_list = []
        pixel_feat_list = []


        for (j, (frus_nss_coor_, z_vals_, ray_dir_)) in enumerate(zip(frus_nss_coor, z_vals, ray_dir)):
            # print(j, 'j')
            h, w = divmod(j, scale)
            H_, W_ = H // scale, W // scale
            sampling_coor_fg_ = frus_nss_coor_[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
            sampling_coor_bg_ = frus_nss_coor_  # Px3

            if self.opt.use_ray_dir:
                ray_dir_input_ = ray_dir_.view([N, H_, W_, 3]).unsqueeze(1).expand(-1, D, -1, -1, -1)
                ray_dir_input_ = ray_dir_input_.flatten(0, 3)
            else:
                ray_dir_input_ = None

            if self.opt.pixel_encoder or self.opt.mask_as_decoder_input:
                # get cam matrices
                # W, H, D = self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp # what is this?
                self.cam2spixel = self.projection_coarse.cam2spixel
                self.world2nss = self.projection_coarse.world2nss
                frustum_size = torch.Tensor(self.projection_coarse.frustum_size).to(self.device)
                # print(frustum_size, 'frustum_size') # 128x128x256

                # construct uv in the first image coordinates
                cam02world = cam2world[0:1]  # 1x4x4
                world2cam0 = cam02world.inverse()  # 1x4x4
                nss2world = self.world2nss.inverse()  # 1x4x4
                frus_nss_coor_ = torch.cat([frus_nss_coor_, torch.ones_like(frus_nss_coor_[:, 0].unsqueeze(1))],
                                          dim=-1)  # Px4
                frus_world_coor_ = torch.matmul(nss2world[None, ...], frus_nss_coor_[None, ..., None])  # 1xPx4x1
                frus_cam0_coor_ = torch.matmul(world2cam0[None, ...],
                                              frus_world_coor_)  # 1x1x4x4, 1x(NxDxHxW)x4x1 -> 1x(NxDxHxW)x4x1 # TODO: check this
                pixel_cam0_coor_ = torch.matmul(self.cam2spixel[None, ...], frus_cam0_coor_)  # 1x1x4x4, 1x(NxDxHxW)x4x1
                pixel_cam0_coor_ = pixel_cam0_coor_.squeeze(-1)  # 1x(NxDxHxW)x4
                uv_ = pixel_cam0_coor_[:, :, 0:2] / pixel_cam0_coor_[:, :, 2].unsqueeze(-1)  # 1x(NxDxHxW)x2
                uv_ = (uv_ + 0.) / frustum_size[0:2][None, None, :] * 2 - 1  # nomalize to fit in with [-1, 1] grid # TODO: check this
                # 0 -> 0.5/frustum_size, 1 -> 1.5/frustum_size, ..., frustum_size-1 -> (frustum_size-0.5/frustum_size)
                # then, change [0, 1] -> [-1, 1]
                # if self.opt.debug:
                #     uv = torch.cat([uv[..., 1:2], uv[..., 0:1]], dim=-1) # flip uv

                if self.opt.weight_pixelfeat:
                    w_ = pixel_cam0_coor_[:, :, 2:3]
                    w_ = (w_ - self.opt.near_plane) / (
                                self.opt.far_plane - self.opt.near_plane)  # [0, 1] torch linspace is inclusive (include final value)
                    # w_ = (w_ + 0.) * 2 - 1  # [-1, 1]
                    w = w * frustum_size[2:3][None, None, :]  # [0, 63]
                    w = (w + 0.) / frustum_size[2] * 2 - 1  # [-1, 1]

                    # uvw = torch.cat([w, uv], dim=-1)  # 1x(NxDxHxW)x3 # this is wrong. think about the x y z coordinate system! (H, W, D)
                    uvw_ = torch.cat([uv_, w_], dim=-1)  # 1x(NxDxHxW)x3

                    uvw[:, :, :, h::scale, w::scale, :] = uvw_.view([1, N, D, H_, W_, 3])

                if self.opt.mask_image or self.opt.mask_image_feature:
                    uv_ = uv_.expand(K, -1, -1)  # 1x(NxDxHxW)x2 -> Kx(NxDxHxW)x2
                    pixel_feat_ = self.netPixelEncoder.index(uv_)  # Kx(NxDxHxW)x2 -> KxCx(NxDxHxW)
                    pixel_feat_ = pixel_feat_.transpose(1, 2)
                else:
                    pixel_feat_ = self.netPixelEncoder.index(uv_)  # 1x(NxDxHxW)x2 -> 1xCx(NxDxHxW)
                    pixel_feat_ = pixel_feat_.transpose(1, 2)  # 1x(NxDxHxW)xC
                    pixel_feat_ = pixel_feat_.expand(K, -1, -1)  # Kx(NxDxHxW)xC
                    uv_ = uv_.expand(K, -1, -1)

                if self.opt.mask_as_decoder_input or self.opt.weight_pixel_slot_mask:
                    silhouette0 = self.silhouette_masks[0:1].transpose(0, 1)  # Kx1xHxW
                    # uv = uv.unsqueeze(1)  # Kx1x(NxDxHxW)x2
                    silhouettes_for_density_ = F.grid_sample(silhouette0, uv_.unsqueeze(1), mode='bilinear',
                                                            padding_mode='zeros', )  # Kx1(C)x1x(NxDxHxW)
                    silhouettes_for_density_ = silhouettes_for_density_.flatten(0, 3).view(K, N, D, H_, W_).permute(
                        [1, 2, 0, 3, 4])  # NxDxKxHxW  # NxDxKxHxW
                else:
                    silhouettes_for_density_ = None

                if self.opt.unified_decoder:
                    if self.opt.debug2:
                        pixel_feat_ *= 0
                    raws_, masked_raws_, unmasked_raws_, masks_for_silhouette_ = \
                        self.netDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0, pixel_feat=pixel_feat_,
                                        ray_dir_input=ray_dir_input_, decoder_type='unified',
                                        silhouettes=silhouettes_for_density_)
                    # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
                    raws_ = raws_.view([N, D, H_, W_, 4])#.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                    masked_raws_ = masked_raws_.view([K, N, D, H_, W_, 4])
                    save_masked_raws = masked_raws_.clone()
                    unmasked_raws_ = unmasked_raws_.view([K, N, D, H_, W_, 4])
                    masks_for_silhouette_ = masks_for_silhouette_.view([K, N, D, H_, W_])

                    raws[..., h::scale, w::scale, :] = raws_
                    masked_raws[..., h::scale, w::scale, :] = masked_raws_
                    unmasked_raws[..., h::scale, w::scale, :] = unmasked_raws_

                    z_vals, ray_dir = z_vals_, ray_dir_
                    raws_ = raws_.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                    rgb_map_, depth_map_, _, _, _, silhouettes_ = raw2outputs(raws_, z_vals_, ray_dir_,
                                                                           return_silhouettes=masks_for_silhouette_)
                    rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                    x_recon_ = rendered_ * 2 - 1
                    x_recon[..., h::scale, w::scale] = x_recon_
                    silhouettes[..., h::scale, w::scale] = silhouettes_ # NxKxHxW
                    raw_masks_density_list.append(masked_raws_[..., -1:])

                else:
                    raws_density_, masked_raws_density_, unmasked_raws_density_, masks_for_silhouette_density_, raw_masks_density_ = \
                        self.netDensityDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0, pixel_feat_,
                                               ray_dir_input=ray_dir_input_, decoder_type='density')

                    raws_density_ = raws_density_.view([N, D, H_, W_, 4])
                    masks_for_silhouette_density_ = masks_for_silhouette_density_.view([K, N, D, H_, W_])
                    raws_density[:, :, h::scale, w::scale, :] = raws_density_
                    masks_for_silhouette_density[:, :, :, h::scale, w::scale] = masks_for_silhouette_density_

                    raw_masks_density_list.append(raw_masks_density_)
                    if not self.opt.extract_mesh:
                        pixel_feat_list.append(pixel_feat_)
                        z_vals_not_partitioned[:, h::scale, w::scale, :] = z_vals_.view([N, H_, W_, D])
                        ray_dir_not_partitioned[:, h::scale, w::scale, :] = ray_dir_.view([N, H_, W_, 3])
                    # raw_masks_density[:, :, :, h::scale, w::scale, :] = raw_masks_density_.view(K, N, D, H_, W_, 1)

        if self.opt.extract_mesh:
            # evaluate mesh
            voxels = raws[1:, ..., -1].cpu().numpy()  # remove background
            num_points = 30
            num_objects = voxels.shape[0]
            hdf_path = self.input_paths[0].replace('_frame5_img0.png', '.hdf5')

            d = 5e-7 / (voxels.shape[0] ** 2)
            gt_obj_vtx, gt_obj_face, gt_scene_vtx, gt_scene_face = load_gt_mesh_from_hdf(hdf_path, frame='0005', num_objects=num_objects)
            pred_obj_vtx, pred_obj_face, pred_scene_vtx, pred_scene_face = compute_mesh_from_voxel(voxels, threshold=1.0)

            pred_obj_pts = []
            gt_obj_pts = []

            # object mesh loss
            total_mesh_loss = 0.
            for obj_id, data in enumerate(zip(pred_obj_vtx, pred_obj_face, gt_obj_vtx, gt_obj_face)):
                pred_vtx, pred_face, gt_vtx, gt_face = data
                pred_pts = get_surface_points_from_mesh(pred_vtx, pred_face, d=d, num=num_points)
                gt_pts = get_surface_points_from_mesh(gt_vtx, gt_face, d=d, num=num_points)
                pred_pts = normalize_points(pred_pts)
                gt_pts = normalize_points(gt_pts)
                pred_obj_pts.append(pred_pts)
                gt_obj_pts.append(gt_pts)

                distance = chamfer_distance(torch.tensor(pred_pts[None]), torch.tensor(gt_pts[None]))
                total_mesh_loss += distance[0]
            avg_mesh_loss = total_mesh_loss / num_objects
            print('Average object mesh loss: ', avg_mesh_loss)

            # scene mesh loss
            pred_scene_pts = get_surface_points_from_mesh(pred_scene_vtx, pred_scene_face, d=d, num=num_points)
            gt_scene_pts = get_surface_points_from_mesh(gt_scene_vtx, gt_scene_face, d=d, num=num_points)
            pred_scene_pts = normalize_points(pred_scene_pts)
            gt_scene_pts = normalize_points(gt_scene_pts)
            distance = chamfer_distance(torch.tensor(pred_scene_pts[None]), torch.tensor(gt_scene_pts[None]))[0]

            visualize_dict = {
                'image': self.x[0],
                'gt_obj_vtx': gt_obj_vtx,
                'gt_obj_face': gt_obj_face,
                'gt_scene_vtx': gt_scene_vtx,
                'gt_scene_face': gt_scene_face,
                'gt_obj_pts': gt_obj_pts,
                'gt_scene_pts': gt_scene_pts,
                'pred_obj_vtx': pred_obj_vtx,
                'pred_obj_face': pred_obj_face,
                'pred_scene_vtx': pred_scene_vtx,
                'pred_scene_face': pred_scene_face,
                'pred_obj_pts': pred_obj_pts,
                'pred_scene_pts': pred_scene_pts,
            }

            np.save('test.npy', visualize_dict)
            breakpoint()

            return raw_masks_density_list
        if self.opt.unified_decoder:
            pass
        else:
            raws_density = raws_density.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
            uvw = uvw.view([1, N, D, H, W, 3]).flatten(1, 4)
            uv = uvw[..., 0:2].expand(K, -1, -1)

            z_vals_not_partitioned = z_vals_not_partitioned.flatten(0, 2)
            ray_dir_not_partitioned = ray_dir_not_partitioned.flatten(0, 2)

            weights, transmittance_samples, silhouettes = \
                raw2transmittances(raws_density, z_vals_not_partitioned, ray_dir_not_partitioned, uvw=uvw, KNDHW=(K, N, D, H, W),
                                   masks=masks_for_silhouette_density)

            transmittance_samples = transmittance_samples.view([N, H, W, D]).permute([0, 3, 1, 2]).flatten(0, 3)[
                None, ..., None]  # 1x(NxDxHxW)x1

            silhouette0 = silhouettes[0:1].transpose(0, 1)  # Kx1xHxW
            uv = uv.unsqueeze(1)  # Kx1x(NxDxHxW)x2
            silhouettes_for_color = F.grid_sample(silhouette0, uv, mode='bilinear',
                                                  padding_mode='zeros', )  # Kx1(C)x1x(NxDxHxW)
            silhouettes_for_color = silhouettes_for_color.flatten(0, 3).view(K, N, D, H, W).permute(
                [1, 2, 0, 3, 4])  # NxDxKxHxW

            for (j, (frus_nss_coor_, z_vals_, ray_dir_)) in enumerate(zip(frus_nss_coor, z_vals, ray_dir)):
                h, w = divmod(j, scale)
                H_, W_ = H // scale, W // scale
                sampling_coor_fg_ = frus_nss_coor_[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
                sampling_coor_bg_ = frus_nss_coor_  # Px3

                if self.opt.use_ray_dir:
                    ray_dir_input_ = ray_dir_.view([N, H_, W_, 3]).unsqueeze(1).expand(-1, D, -1, -1, -1)
                    ray_dir_input_ = ray_dir_input_.flatten(0, 3)
                else:
                    ray_dir_input_ = None

                pixel_feat_ = pixel_feat_list[j]
                raw_masks_density_ = raw_masks_density_list[j]
                # raw_masks_density_ = raw_masks_density[:, :, :, h::scale, w::scale, :].flatten(1, 4)
                silhouettes_for_color_ = silhouettes_for_color[:, :, :, h::scale, w::scale] # NxDxKxHxW
                transmittance_samples_ = transmittance_samples.view([N, D, H, W])[:, :, h::scale, w::scale].flatten(0, 3)[
                None, ..., None]

                raws_color_, masked_raws_color_, unmasked_raws_color_, masks_for_silhouette_color_ = \
                    self.netColorDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0, pixel_feat=pixel_feat_,
                                         ray_dir_input=ray_dir_input_, transmittance_samples=transmittance_samples_,
                                         raw_masks_density=raw_masks_density_,
                                         silhouettes=silhouettes_for_color_,
                                         decoder_type='color')

                raws_ = raws_color_.view([N, D, H_, W_, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0,
                                                                                         end_dim=2)  # (NxHxW)xDx4
                masked_raws_ = masked_raws_color_.view([K, N, D, H_, W_, 4])
                unmasked_raws_ = unmasked_raws_color_.view([K, N, D, H_, W_, 4])
                masked_raws[..., h::scale, w::scale, :] = masked_raws_
                unmasked_raws[..., h::scale, w::scale, :] = unmasked_raws_
                masks_for_silhouette_ = masks_for_silhouette_color_.view([K, N, D, H_, W_])

                weights_ = weights.view([N, H, W, D])[:, h::scale, w::scale, :] # [N_rays, N_samples]
                weights_ = weights_.flatten(0, 2)

                rgb_map_, _, _, _, = raw2colors(raws_, weights_, z_vals_)

                rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon_ = rendered_ * 2 - 1
                x_recon[..., h::scale, w::scale] = x_recon_

        x_recon_novel, x_novel = x_recon[1:], F.interpolate(x[1:], size=[self.opt.frustum_size, self.opt.frustum_size], mode='bilinear')
        self.loss_recon = self.L2_loss(x_recon_novel, x_novel)
        self.loss_lpips = self.LPIPS_loss(x_recon_novel, x_novel).mean()
        self.loss_psnr = compute_psnr(x_recon_novel / 2 + 0.5, x_novel / 2 + 0.5, data_range=1.)
        self.loss_ssim = compute_ssim(x_recon_novel / 2 + 0.5, x_novel / 2 + 0.5, data_range=1.)

        if self.opt.silhouette_loss:
            if self.opt.dice_loss:
                self.loss_silhouette = self.dice_loss(self.silhouette_masks_fine, silhouettes) # NxKxHxW # 128x128
                print('dice_loss : ', self.loss_silhouette)
            else:
                self.loss_silhouette = self.L1_loss(self.silhouette_masks_fine, silhouettes) # NxKxHxW
                print('L1_loss : ', self.loss_silhouette)


        with torch.no_grad():
            for i in range(self.opt.n_img_each_scene):
                setattr(self, 'x_rec{}'.format(i), x_recon[i])
                if i == 0:
                    setattr(self, 'input_image', F.interpolate(x[i].unsqueeze(0), size=[self.opt.frustum_size, self.opt.frustum_size], mode='bilinear').squeeze(0))
                else:
                    setattr(self, 'gt_novel_view{}'.format(i), F.interpolate(x[i].unsqueeze(0), size=[self.opt.frustum_size, self.opt.frustum_size], mode='bilinear').squeeze(0))
            setattr(self, 'masked_raws', masked_raws.detach())
            setattr(self, 'unmasked_raws', unmasked_raws.detach())


    def compute_visuals(self):
        with torch.no_grad():
            cam2world = self.cam2world[:self.opt.n_img_each_scene]
            _, N, D, H, W, _ = self.masked_raws.shape
            masked_raws = self.masked_raws  # KxNxDxHxWx4
            mask_maps = []
            for k in range(self.num_slots):
                raws = masked_raws[k]  # NxDxHxWx4
                _, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _, mask_map, _, _ = raw2outputs(raws, z_vals, ray_dir, render_mask=True)
                mask_maps.append(mask_map.view(N, H, W))
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i])

            mask_maps = torch.stack(mask_maps)  # KxNxHxW
            mask_idx = mask_maps.cpu().argmax(dim=0)  # NxHxW
            predefined_colors = []
            obj_idxs = self.obj_idxs  # Kx1xHxW
            gt_mask0 = self.gt_masks[0]  # 3xHxW
            for k in range(self.num_slots):
                mask_idx_this_slot = mask_idx[0:1] == k  # 1xHxW
                iou_this_slot = []
                for kk in range(self.num_slots):
                    try:
                        obj_idx = obj_idxs[kk, ...]  # 1xHxW
                    except IndexError:
                        break
                    iou = (obj_idx & mask_idx_this_slot).type(torch.float).sum() / (obj_idx | mask_idx_this_slot).type(torch.float).sum()
                    iou_this_slot.append(iou)
                target_obj_number = torch.tensor(iou_this_slot).argmax()
                target_obj_idx = obj_idxs[target_obj_number, ...].squeeze()  # HxW
                obj_first_pixel_pos = target_obj_idx.nonzero()[0]  # 2
                obj_color = gt_mask0[:, obj_first_pixel_pos[0], obj_first_pixel_pos[1]]
                predefined_colors.append(obj_color)
            predefined_colors = torch.stack(predefined_colors).permute([1,0])
            mask_visuals = predefined_colors[:, mask_idx]  # 3xNxHxW

            nvari_meter = AverageMeter()
            for i in range(N):
                setattr(self, 'render_mask{}'.format(i), mask_visuals[:, i, ...])
                setattr(self, 'gt_mask{}'.format(i), self.gt_masks[i])
                this_mask_idx = mask_idx[i].flatten(start_dim=0)
                gt_mask_idx = self.mask_idx[i]  # HW
                fg_idx = self.fg_idx[i]
                fg_idx_map = fg_idx.view([self.opt.frustum_size, self.opt.frustum_size])[None, ...]
                fg_map = mask_visuals[0:1, i, ...].clone()
                fg_map[fg_idx_map] = -1.
                fg_map[~fg_idx_map] = 1.
                setattr(self, 'bg_map{}'.format(i), fg_map)
                if i == 0:
                    ari_score = adjusted_rand_score(gt_mask_idx, this_mask_idx)
                    fg_ari = adjusted_rand_score(gt_mask_idx[fg_idx], this_mask_idx[fg_idx])
                    self.loss_ari = ari_score
                    self.loss_fgari = fg_ari
                else:
                    ari_score = adjusted_rand_score(gt_mask_idx, this_mask_idx)
                    nvari_meter.update(ari_score)
                self.loss_nvari = nvari_meter.val

    def backward(self):
        pass

    def optimize_parameters(self, ret_grad=False, epoch=0):
        """Update network weights; it will be called in every training iteration."""
        self.forward(epoch)
        for opm in self.optimizers:
            opm.zero_grad()
        self.backward()
        avg_grads = []
        layers = []
        if ret_grad:
            for n, p in chain(self.netEncoder.named_parameters(), self.netSlotAttention.named_parameters(), self.netDecoder.named_parameters()):
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