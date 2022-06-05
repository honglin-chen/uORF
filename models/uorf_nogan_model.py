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
from .model import Encoder, Decoder, SlotAttention, get_perceptual_net, raw2outputs, PixelEncoder, PixelDecoder
import pdb
from util import util

class uorfNoGanModel(BaseModel):

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
        parser.add_argument('--bottom', action='store_true', help='one more unet_model layer on bottom')
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
        parser.add_argument('--pixel_encoder', action='store_true', help='change unet_model to pixel_encoder')
        parser.add_argument('--pixel_nerf', action='store_true', help='change model to pixelnerf')
        parser.add_argument('--mask_image_feature', action='store_true', help='mask image feature in pixelnerf decoder. if pixelencoder==False: this does not make sense')
        parser.add_argument('--mask_image', action='store_true', help='mask image in pixelnerf unet_model. if pixelencoder==False: this does not make sense')
        parser.add_argument('--slot_repeat', action='store_true', help='put slot features repeatedly in the uORF decoder setting')
        parser.add_argument('--no_concatenate', action='store_true', help='do not concatenate object feature and pixel feature; instead, add them')
        parser.add_argument('--visualize_obj_feat', action='store_true', help='visualize object feature (after slot attention)')
        parser.add_argument('--input_no_loss', action='store_true', help='for the first n (=100) epochs (hyperparameter), do not include first image in the recon loss for learning geometry first')
        parser.add_argument('--bg_no_pixel', action='store_true', help='do not provide bg with pixel features, so that it can learn bg only from slot (object) features')
        parser.add_argument('--use_ray_dir', action='store_true', help='concatenate ray direction on the view. now only work on decoder (not pixel)')
        parser.add_argument('--weight_pixelfeat', action='store_true', help='weigh pixel_color and slot color using the transmittance from the input view')
        parser.add_argument('--silhouette_loss', action='store_true', help='enable silhouette loss')
        parser.add_argument('--reduce_latent_size', action='store_true', help='reduce latent size of pixel unet_model. this option is not true for default and not configurable. TODO')
        parser.add_argument('--small_latent', action='store_true', help='reduce latent dim of decoder by 2')
        parser.add_argument('--shared_pixel_slot_decoder', action='store_true', help='use pixel decoder for slot decoder. need to check whether this reduce memory burden')
        parser.add_argument('--div_by_max', action='store_true', help='divide pixel feature importance by max per each ray')
        parser.add_argument('--learn_only_silhouette', action='store_true', help='loss function contains only silhouette loss')
        parser.add_argument('--focal_ratio', nargs='+', default=(350. / 320., 350. / 240.), help='set the focal ratio in projection.py')
        parser.add_argument('--density_n_color', action='store_true', help='separate density unet_model and color unet_model')

        parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
                            dataset_mode='multiscenes', niter=1200, custom_lr=True, lr_policy='warmup')

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
        n = opt.n_img_each_scene
        self.visual_names = ['x{}'.format(i) for i in range(n)] + \
                            ['x_rec{}'.format(i) for i in range(n)] + \
                            ['slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                            ['unmasked_slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                            ['slot{}_attn'.format(k) for k in range(opt.num_slots)]
        self.model_names = []
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
        self.parameters = []
        self.nets = []

        # [uORF Encoder]
        self.num_slots = opt.num_slots
        self.netEncoder = networks.init_net(Encoder(3, z_dim=z_dim, bottom=opt.bottom), gpu_ids=self.gpu_ids, init_type='normal')
        self.parameters.append(self.netEncoder.parameters())
        self.nets.append(self.netEncoder)
        self.model_names.append('Encoder')

        # [Slot attention]
        self.netSlotAttention = networks.init_net(
            SlotAttention(num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter, gt_seg=opt.gt_seg), gpu_ids=self.gpu_ids, init_type='normal')
        self.nets.append(self.netSlotAttention)
        self.parameters.append(self.netSlotAttention.parameters())
        self.model_names.append('SlotAttention')

        # Add [pixel Encoder] or do not add
        if self.opt.pixel_encoder:
            self.netPixelEncoder = networks.init_net(PixelEncoder(mask_image=self.opt.mask_image, mask_image_feature=self.opt.mask_image_feature), gpu_ids=self.gpu_ids, init_type='None')
            self.parameters.append(self.netPixelEncoder.parameters())
            self.nets.append(self.netPixelEncoder)
            self.model_names.append('PixelEncoder')
            pixel_dim = 64
        else:
            pixel_dim = None

        # [pixel Decoder] or [uORF Decoder]
        if self.opt.pixel_decoder:
            self.netPixelDecoder = networks.init_net(PixelDecoder(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim+64, z_dim=opt.z_dim, n_layers=opt.n_layer,
                                   locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality, slot_repeat=self.opt.slot_repeat), gpu_ids=self.gpu_ids, init_type='None')
            self.parameters.append(self.netPixelDecoder.parameters())
            self.nets.append(self.netPixelDecoder)
            self.model_names.append('PixelDecoder')

        else:
            if self.opt.density_n_color:
                self.netDensityDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6 * opt.n_freq + 3 + z_dim, pixel_dim=pixel_dim, z_dim=opt.z_dim, n_layers=opt.n_layer,
                            locality_ratio=opt.obj_scale / opt.nss_scale, fixed_locality=opt.fixed_locality, no_concatenate=self.opt.no_concatenate, bg_no_pixel=self.opt.bg_no_pixel,
                                                            use_ray_dir=self.opt.use_ray_dir, small_latent=self.opt.small_latent), gpu_ids=self.gpu_ids, init_type='xavier')
                self.parameters.append(self.netDensityDecoder.parameters())
                self.nets.append(self.netDensityDecoder)
                self.model_names.append('DensityDecoder')
                self.netColorDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6 * opt.n_freq + 3 + z_dim, pixel_dim=pixel_dim, z_dim=opt.z_dim, n_layers=opt.n_layer,
                            locality_ratio=opt.obj_scale / opt.nss_scale, fixed_locality=opt.fixed_locality, no_concatenate=self.opt.no_concatenate, bg_no_pixel=self.opt.bg_no_pixel,
                                                            use_ray_dir=self.opt.use_ray_dir, small_latent=self.opt.small_latent), gpu_ids=self.gpu_ids, init_type='xavier')
                self.parameters.append(self.netColorDecoder.parameters())
                self.nets.append(self.netColorDecoder)
                self.model_names.append('ColorDecoder')

            else:
                if self.opt.weight_pixelfeat and self.opt.shared_pixel_slot_decoder:
                    self.netDecoder = networks.init_net(
                        Decoder(n_freq=opt.n_freq, input_dim=6 * opt.n_freq + 3 + z_dim, pixel_dim=pixel_dim,
                                z_dim=opt.z_dim, n_layers=opt.n_layer,
                                locality_ratio=opt.obj_scale / opt.nss_scale, fixed_locality=opt.fixed_locality,
                                no_concatenate=self.opt.no_concatenate, bg_no_pixel=self.opt.bg_no_pixel,
                                use_ray_dir=self.opt.use_ray_dir, small_latent=True), gpu_ids=self.gpu_ids, init_type='xavier')
                else:
                    self.netDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6 * opt.n_freq + 3 + z_dim, pixel_dim=pixel_dim, z_dim=opt.z_dim, n_layers=opt.n_layer,
                            locality_ratio=opt.obj_scale / opt.nss_scale, fixed_locality=opt.fixed_locality, no_concatenate=self.opt.no_concatenate, bg_no_pixel=self.opt.bg_no_pixel,
                                                            use_ray_dir=self.opt.use_ray_dir, small_latent=self.opt.small_latent), gpu_ids=self.gpu_ids, init_type='xavier')
                self.parameters.append(self.netDecoder.parameters())
                self.nets.append(self.netDecoder)
                self.model_names.append('Decoder')

        if self.isTrain:  # only defined during training time
            self.optimizer = optim.Adam(chain(*self.parameters), lr=opt.lr)
            self.optimizers = [self.optimizer]

        self.L2_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss(reduction='sum')

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
        # print(self.x.shape, 'self.x.shape')
        self.cam2world = input['cam2world'].to(self.device)
        if not self.opt.fixed_locality:
            self.cam2world_azi = input['azi_rot'].to(self.device)

        if 'masks' in input.keys():
            bg_masks = input['bg_mask'][0:1].to(self.device)
            obj_masks = input['obj_masks'][0:1].to(self.device)

            masks = torch.cat([bg_masks, obj_masks], dim=1)
            masks = F.interpolate(masks.float(), size=[64, 64], mode='nearest')
            self.masks = masks.flatten(2, 3)

        if self.opt.silhouette_loss:
            bg_masks = input['bg_mask'].to(self.device)
            obj_masks = input['obj_masks'].to(self.device)

            masks = torch.cat([bg_masks, obj_masks], dim=1)
            masks = F.interpolate(masks.float(), size=[128, 128], mode='nearest')
            self.silhouette_masks_fine = masks
            masks = F.interpolate(masks.float(), size=[64, 64], mode='nearest')
            # self.silhouette_masks = masks.flatten(2, 3) # NxKx(HxW)
            self.silhouette_masks = masks # NxKxHxW


    def forward(self, epoch=0):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.weight_percept = self.opt.weight_percept if epoch >= self.opt.percept_in else 0
        self.loss_recon = 0
        self.loss_perc = 0
        dev = self.x[0:1].device
        nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()

        # Replace slot attention with GT segmentations
        if self.opt.gt_seg:
            masks = self.masks
            attn_bg, attn_fg = masks[:, 0:1], masks[:, 1:]
            attn = torch.cat([attn_bg, attn_fg], dim=1)

        # Encoding images
        feature_map = self.netEncoder(F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False))  # BxCxHxW
        feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC

        # Slot Attention
        z_slots, attn = self.netSlotAttention(feat, masks=self.masks)  # 1xKxC, 1xKxN
        z_slots, attn = z_slots.squeeze(0), attn.squeeze(0)  # KxC, KxN (N = HxW)
        K = attn.shape[0]

        # Pixel Encoder Forward (to get feature values in pixel coordinates (uv), call pixelEncoder.index(uv), not forward)
        if self.opt.pixel_encoder:
            masks = attn if self.opt.mask_image or self.opt.mask_image_feature else None
            feature_map_pixel = self.netPixelEncoder(F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False), masks = masks)

        # Get rays and coordinates
        cam2world = self.cam2world
        N = cam2world.shape[0]
        if self.opt.stage == 'coarse':
            W, H, D = self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            x = F.interpolate(self.x, size=self.opt.supervision_size, mode='bilinear', align_corners=False)
            self.z_vals, self.ray_dir = z_vals, ray_dir
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
            x = self.x[:, :, H_idx:H_idx + rs, W_idx:W_idx + rs]
            self.silhouette_masks = self.silhouette_masks_fine[..., H_idx:H_idx + rs, W_idx:W_idx + rs]
            self.z_vals, self.ray_dir = z_vals, ray_dir
            W, H, D = self.opt.render_size, self.opt.render_size, self.opt.n_samp

        if self.opt.use_ray_dir:
            ray_dir_input = ray_dir.view([N, H, W, 3]).unsqueeze(1).expand(-1, D, -1, -1, -1)
            ray_dir_input = ray_dir_input.flatten(0, 3)
        else:
            ray_dir_input = None
        sampling_coor_fg = frus_nss_coor[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
        sampling_coor_bg = frus_nss_coor  # Px3

        W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp

        # Get pixel feature if using [pixel Encoder]
        if self.opt.pixel_encoder:
            # get cam matrices
            W, H, D = self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp
            self.cam2spixel = self.projection.cam2spixel
            self.world2nss = self.projection.world2nss
            frustum_size = torch.Tensor(self.projection.frustum_size).to(self.device)
            # construct uv in the first image coordinates
            cam02world = cam2world[0:1] # 1x4x4
            world2cam0 = cam02world.inverse() # 1x4x4
            nss2world = self.world2nss.inverse() # 1x4x4
            frus_nss_coor = torch.cat([frus_nss_coor, torch.ones_like(frus_nss_coor[:, 0].unsqueeze(1))], dim=-1) # Px4
            frus_world_coor = torch.matmul(nss2world[None, ...], frus_nss_coor[None, ..., None]) # 1xPx4x1
            frus_cam0_coor = torch.matmul(world2cam0[None, ...], frus_world_coor) #1x1x4x4, 1x(NxDxHxW)x4x1 -> 1x(NxDxHxW)x4x1 # TODO: check this
            pixel_cam0_coor = torch.matmul(self.cam2spixel[None, ...], frus_cam0_coor) # 1x1x4x4, 1x(NxDxHxW)x4x1
            pixel_cam0_coor = pixel_cam0_coor.squeeze(-1) # 1x(NxDxHxW)x4
            uv = pixel_cam0_coor[:, :, 0:2]/pixel_cam0_coor[:, :, 2].unsqueeze(-1) # 1x(NxDxHxW)x2
            uv = ((uv+0.5)/frustum_size[0:2][None, None, :] - 0.5) * 2 # nomalize to fit in with [-1, 1] grid # TODO: check this
            # 0 -> 0.5/frustum_size, 1 -> 1.5/frustum_size, ..., frustum_size-1 -> (frustum_size-0.5/frustum_size)
            # then, change [0, 1] -> [-1, 1]
            
            if self.opt.weight_pixelfeat:
                w = pixel_cam0_coor[:, :, 2].unsqueeze(-1)
                w = (w - self.opt.near_plane) / (self.opt.far_plane - self.opt.near_plane) # [0, 1] torch linspace is inclusive (include final value)
                w = w * frustum_size[2:3][None, None, :] * 2 - 1 # [0, 63]
                wuv = torch.cat([w, uv], dim=-1) # 1x(NxDxHxW)x3
            else:
                wuv = None

            if self.opt.mask_image or self.opt.mask_image_feature:
                uv = uv.expand(K, -1, -1).clone() # 1x(NxDxWxH)x2 -> Kx(NxDxWxH)x2
                pixel_feat = self.netPixelEncoder.index(uv)  # Kx(NxDxWxH)x2 -> KxCx(NxDxWxH)
                pixel_feat = pixel_feat.transpose(1, 2)
            else:
                pixel_feat = self.netPixelEncoder.index(uv)  # 1x(NxDxWxH)x2 -> 1xCx(NxDxWxH)
                pixel_feat = pixel_feat.transpose(1, 2)  # 1x(NxDxWxH)xC
                pixel_feat = pixel_feat.expand(K, -1, -1).clone() # Kx(NxDxWxH)xC
        else:
            pixel_feat = None
            wuv = None

        # Run [uORF Decoder] or [pixel Decoder]
        # if self.opt.pixel_decoder: # TODO: this option should be deprecated
        #     raws, masked_raws, unmasked_raws, masks = \
        #         self.netPixelDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0, pixel_feat, slot_repeat = self.opt.slot_repeat)
        #     # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1, Kx(NxDxHxW)xC

        raws, masked_raws, unmasked_raws, masks_for_silhouette = \
            self.netDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0, pixel_feat, no_concatenate=self.opt.no_concatenate, ray_dir_input=ray_dir_input)
        # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1

        raws = raws.view([N, D, H, W, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
        masked_raws = masked_raws.view([K, N, D, H, W, 4])
        unmasked_raws = unmasked_raws.view([K, N, D, H, W, 4])
        masks_for_silhouette = masks_for_silhouette.view([K, N, D, H, W])

        if self.opt.weight_pixelfeat:
            raws_slot, masked_raws_slot, unmasked_raws_slot, masks_slot_for_silhouette = \
                self.netDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0, pixel_feat=None,
                                no_concatenate=self.opt.no_concatenate, ray_dir_input=ray_dir_input)
            # self.netSlotDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0, pixel_feat=None, no_concatenate=self.opt.no_concatenate, ray_dir_input=ray_dir_input)
            # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1

            raws_slot = raws_slot.view([N, D, H, W, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
            masked_raws_slot = masked_raws_slot.view([K, N, D, H, W, 4])
            unmasked_raws_slot = unmasked_raws_slot.view([K, N, D, H, W, 4])

            self.masked_raws_slot = masked_raws_slot
            self.wuv = wuv
            self.KNDHW = (K, N, D, H, W)  # not sure the location of this part
        else:
            raws_slot = None

        return_transmittance_cam0 = True if self.opt.weight_pixelfeat else False
        return_silhouettes = masks_for_silhouette if self.opt.silhouette_loss else None
        rgb_map, _, _, _, transmittance_cam0, silhouettes = raw2outputs(raws, z_vals,
                                                                     ray_dir,
                                                                     render_mask=True,
                                                                     weight_pixelfeat=self.opt.weight_pixelfeat,
                                                                     raws_slot=raws_slot,
                                                                     wuv=wuv,
                                                                     KNDHW=(K, N, D, H, W),
                                                                     return_transmittance_cam0=return_transmittance_cam0,
                                                                     input_transmittance_cam0=None,
                                                                     return_silhouettes=return_silhouettes)

        if self.opt.weight_pixelfeat:
            self.transmittance_cam0 = transmittance_cam0
        else:
            self.transmittance_cam0 = None
        if self.opt.silhouette_loss:
            self.silhouettes = silhouettes
        else:
            self.silhouettes = None

        rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
        x_recon = rendered * 2 - 1

        if self.opt.learn_only_silhouette:
            self.loss_recon = 0.
            self.loss_perc = 0.
        else:
            self.loss_recon = self.L2_loss(x_recon, x)
            x_norm, rendered_norm = self.vgg_norm((x + 1) / 2), self.vgg_norm(rendered)
            rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
            self.loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)

        if self.opt.silhouette_loss:
            self.loss_silhouette = self.L2_loss(self.silhouette_masks, self.silhouettes) # NxKxHxW

        with torch.no_grad():
            attn = attn.detach().cpu()  # KxN
            H_, W_ = feature_map.shape[2], feature_map.shape[3]
            attn = attn.view(self.opt.num_slots, 1, H_, W_)
            if H_ != H:
                attn = F.interpolate(attn, size=[H, W], mode='bilinear')
            for i in range(self.opt.n_img_each_scene):
                setattr(self, 'x_rec{}'.format(i), x_recon[i])
                setattr(self, 'x{}'.format(i), x[i])
            setattr(self, 'masked_raws', masked_raws.detach())
            setattr(self, 'unmasked_raws', unmasked_raws.detach())
            setattr(self, 'attn', attn)

            if self.opt.weight_pixelfeat:
                self.masked_raws_slot = masked_raws_slot.detach()
                self.unmasked_raws_slot = unmasked_raws_slot.detach()
                self.wuv = wuv.detach()
                self.KNDHW = (K, N, D, H, W)

    def compute_visuals(self):
        with torch.no_grad():
            _, N, D, H, W, _ = self.masked_raws.shape
            masked_raws = self.masked_raws  # KxNxDxHxWx4
            unmasked_raws = self.unmasked_raws  # KxNxDxHxWx4
            for k in range(self.num_slots):
                raws = masked_raws[k]  # NxDxHxWx4
                z_vals, ray_dir = self.z_vals, self.ray_dir
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                if self.opt.weight_pixelfeat:
                    raws_slot = self.masked_raws_slot[k]
                    raws_slot = raws_slot.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                    rgb_map, depth_map, _, _, _, _ = raw2outputs(raws, z_vals, ray_dir, weight_pixelfeat=self.opt.weight_pixelfeat, raws_slot=raws_slot, wuv=self.wuv, KNDHW=self.KNDHW, input_transmittance_cam0=self.transmittance_cam0)
                else:
                    rgb_map, depth_map, _, _, _, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i])

                raws = unmasked_raws[k]  # (NxDxHxW)x4
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                if self.opt.weight_pixelfeat:
                    raws_slot = self.unmasked_raws_slot[k]
                    raws_slot = raws_slot.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                    rgb_map, depth_map, _, _, _, _ = raw2outputs(raws, z_vals, ray_dir, weight_pixelfeat=self.opt.weight_pixelfeat, raws_slot=raws_slot, wuv=self.wuv, KNDHW=self.KNDHW, input_transmittance_cam0=self.transmittance_cam0)
                else:
                    rgb_map, depth_map, _, _, _, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'unmasked_slot{}_view{}'.format(k, i), x_recon[i])

                setattr(self, 'slot{}_attn'.format(k), self.attn[k] * 2 - 1)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        loss = self.loss_recon + self.loss_perc
        if self.opt.silhouette_loss:
            loss += self.loss_silhouette
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