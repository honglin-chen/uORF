from itertools import chain

import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import os
import time
from .projection import Projection
from .model import Encoder, Decoder, SlotAttention, get_perceptual_net, raw2outputs, PixelEncoder, PixelDecoder
import pdb
from util import util
# from .model import Encoder, Decoder, SlotAttention, raw2outputs
from util.util import AverageMeter
from sklearn.metrics import adjusted_rand_score
import lpips
from piq import ssim as compute_ssim
from piq import psnr as compute_psnr


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
        parser.add_argument('--num_slots', metavar='K', type=int, default=8, help='Number of supported slots')
        parser.add_argument('--z_dim', type=int, default=64, help='Dimension of individual z latent per slot')
        parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
        parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
        parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
        parser.add_argument('--obj_scale', type=float, default=4.5, help='Scale for locality on foreground objects')
        parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
        parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
        parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
        parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
        parser.add_argument('--input_size', type=int, default=64)
        parser.add_argument('--frustum_size', type=int, default=128, help='Size of rendered images')
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
        parser.add_argument('--shared_pixel_slot_decoder', action='store_true', help='use pixel decoder for slot decoder. need to check whether this reduce memory burden')
        parser.add_argument('--div_by_max', action='store_true', help='divide pixel feature importance by max per each ray')
        parser.add_argument('--learn_only_silhouette', action='store_true', help='loss function contains only silhouette loss')
        parser.add_argument('--focal_ratio', nargs='+', default=(350. / 320., 350. / 240.), help='set the focal ratio in projection.py')

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
        self.loss_names = ['ari', 'fgari', 'nvari', 'psnr', 'ssim', 'lpips']
        n = opt.n_img_each_scene
        self.visual_names = ['input_image',] + ['gt_novel_view{}'.format(i+1) for i in range(n-1)] + \
                            ['x_rec{}'.format(i) for i in range(n)] + \
                            ['gt_mask{}'.format(i) for i in range(n)] + \
                            ['render_mask{}'.format(i) for i in range(n)]
        self.model_names = []
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
        self.projection = Projection(focal_ratio=opt.focal_ratio, device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        z_dim = opt.z_dim
        self.parameters = []
        self.nets = []
        self.num_slots = opt.num_slots
        self.netEncoder = networks.init_net(Encoder(3, z_dim=z_dim, bottom=opt.bottom),
                                            gpu_ids=self.gpu_ids, init_type='normal')
        self.parameters.append(self.netEncoder.parameters())
        self.nets.append(self.netEncoder)
        self.model_names.append('Encoder')
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
                                                        use_ray_dir=self.opt.use_ray_dir), gpu_ids=self.gpu_ids, init_type='xavier')
            self.parameters.append(self.netDecoder.parameters())
            self.nets.append(self.netDecoder)
            self.model_names.append('Decoder')

        # if weight_pixelfeat, we need to make a separate slot decoder
        if self.opt.weight_pixelfeat:
            if self.opt.shared_pixel_slot_decoder and self.opt.no_concatenate:
                self.netSlotDecoder = self.netDecoder
            else:
                self.netSlotDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6 * opt.n_freq + 3 + z_dim, pixel_dim=None, z_dim=opt.z_dim, n_layers=opt.n_layer,
                            locality_ratio=opt.obj_scale / opt.nss_scale, fixed_locality=opt.fixed_locality, no_concatenate=self.opt.no_concatenate, bg_no_pixel=self.opt.bg_no_pixel,
                                                            use_ray_dir=self.opt.use_ray_dir, small_latent=False), gpu_ids=self.gpu_ids, init_type='xavier')
            self.parameters.append(self.netSlotDecoder.parameters())
            self.nets.append(self.netSlotDecoder)
            self.model_names.append('SlotDecoder')
        # self.netDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=opt.z_dim, n_layers=opt.n_layer, locality=False,
        #                                             locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality), gpu_ids=self.gpu_ids, init_type='xavier')
        self.L2_loss = torch.nn.MSELoss()
        self.LPIPS_loss = lpips.LPIPS().cuda()

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
            masks = F.interpolate(masks.float(), size=[64, 64], mode='nearest')
            self.masks = masks.flatten(2, 3)

        if self.opt.silhouette_loss:
            bg_masks = input['bg_mask'].to(self.device)
            obj_masks = input['obj_masks'].to(self.device)

            masks = torch.cat([bg_masks, obj_masks], dim=1)
            masks = F.interpolate(masks.float(), size=[64, 64], mode='nearest')

            self.silhouette_masks = masks.flatten(2, 3) # NxKx(HxW)


    def forward(self, epoch=0):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
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
            feature_map_pixel = self.netPixelEncoder(F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False), masks=masks)

        # Get rays and coordinates
        cam2world = self.cam2world
        N = cam2world.shape[0]

        W, H, D = self.projection.frustum_size
        scale = H // self.opt.render_size # size of rendering in one index of for loop.
        frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world, partitioned=True)
        # 4x(NxDx(H/2)x(W/2))x3, 4x(Nx(H/2)x(W/2))xD, 4x(Nx(H/2)x(W/2))x3
        self.cam2spixel = self.projection.cam2spixel
        self.world2nss = self.projection.world2nss
        frustum_size = torch.Tensor(self.projection.frustum_size).to(self.device)

        x = self.x
        x_recon, rendered, masked_raws, unmasked_raws = \
            torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev)
        if self.opt.weight_pixelfeat:
            transmittance_cam0 = torch.zeros([1, 1, D, H, W], device=dev)
        for (j, (frus_nss_coor_, z_vals_, ray_dir_)) in enumerate(zip(frus_nss_coor, z_vals, ray_dir)):
            h, w = divmod(j, scale)
            H_, W_ = H // scale, W // scale
            sampling_coor_fg_ = frus_nss_coor_[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
            sampling_coor_bg_ = frus_nss_coor_  # Px3

            if self.opt.use_ray_dir:
                ray_dir_input = ray_dir_.view([N, H_, W_, 3]).unsqueeze(1).expand(-1, D, -1, -1, -1)
                ray_dir_input_ = ray_dir_input.flatten(0, 3)
            else:
                ray_dir_input_ = None

            # raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
            # Get pixel feature if using [pixel Encoder]
            if self.opt.pixel_encoder:
                W, H, D = self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp
                self.cam2spixel = self.projection.cam2spixel
                self.world2nss = self.projection.world2nss
                frustum_size = torch.Tensor(self.projection.frustum_size).to(self.device)
                # construct uv in the first image coordinates
                cam02world = cam2world[0:1]  # 1x4x4
                world2cam0 = cam02world.inverse()  # 1x4x4
                nss2world = self.world2nss.inverse()  # 1x4x4
                frus_nss_coor = torch.cat([frus_nss_coor_, torch.ones_like(frus_nss_coor_[:, 0].unsqueeze(1))],
                                          dim=-1)  # Px4
                frus_world_coor = torch.matmul(nss2world[None, ...], frus_nss_coor[None, ..., None])  # 1xPx4x1
                frus_cam0_coor = torch.matmul(world2cam0[None, ...],
                                              frus_world_coor)  # 1x1x4x4, 1x(NxDxHxW)x4x1 -> 1x(NxDxHxW)x4x1
                pixel_cam0_coor = torch.matmul(self.cam2spixel[None, ...], frus_cam0_coor)  # 1x1x4x4, 1x(NxDxHxW)x4x1
                pixel_cam0_coor = pixel_cam0_coor.squeeze(-1)  # 1x(NxDxHxW)x4
                uv = pixel_cam0_coor[:, :, 0:2] / pixel_cam0_coor[:, :, 2].unsqueeze(-1)  # 1x(NxDxHxW)x2
                uv = (uv / frustum_size[0:2][None, None, :] - 0.5) * 2  # nomalize to fit in with [-1, 1] grid #
                if self.opt.weight_pixelfeat:
                    w = pixel_cam0_coor[:, :, 2].unsqueeze(-1)
                    w = (w - self.opt.near_plane) / (self.opt.far_plane - self.opt.near_plane)
                    wuv = torch.cat([w, uv], dim=-1)  # 1x(NxDxHxW)x3
                else:
                    wuv = None

                if self.opt.mask_image or self.opt.mask_image_feature:
                    uv = uv.expand(K, -1, -1).clone()  # 1x(NxDxWxH)x2 -> Kx(NxDxWxH)x2
                    pixel_feat = self.netPixelEncoder.index(uv)  # Kx(NxDxWxH)x2 -> KxCx(NxDxWxH)
                    pixel_feat = pixel_feat.transpose(1, 2)
                else:
                    pixel_feat = self.netPixelEncoder.index(uv)  # 1x(NxDxWxH)x2 -> 1xCx(NxDxWxH)
                    pixel_feat = pixel_feat.transpose(1, 2)  # 1x(NxDxWxH)xC
                    pixel_feat = pixel_feat.expand(K, -1, -1).clone()  # Kx(NxDxWxH)xC
            else:
                pixel_feat = None
                wuv = None

            # Run [uORF Decoder] or [pixel Decoder]
            if self.opt.pixel_decoder:
                raws_, masked_raws_, unmasked_raws_, masks_ = \
                    self.netPixelDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0, pixel_feat,
                                         slot_repeat=self.opt.slot_repeat)
                # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1, Kx(NxDxHxW)xC

            else:
                raws_, masked_raws_, unmasked_raws_, masks_ = \
                    self.netDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0, pixel_feat,
                                    no_concatenate=self.opt.no_concatenate, ray_dir_input=ray_dir_input_)
                # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1, Kx(NxDxHxW)xC

            raws_ = raws_.view([N, D, H_, W_, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
            masked_raws_ = masked_raws_.view([K, N, D, H_, W_, 4])
            unmasked_raws_ = unmasked_raws_.view([K, N, D, H_, W_, 4])
            masked_raws[..., h::scale, w::scale, :] = masked_raws_
            unmasked_raws[..., h::scale, w::scale, :] = unmasked_raws_
            # rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_) # original

            if self.opt.weight_pixelfeat:
                raws_slot, masked_raws_slot, unmasked_raws_slot, masks_slot = \
                    self.netSlotDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0, pixel_feat=None,
                                        no_concatenate=self.opt.no_concatenate, ray_dir_input=ray_dir_input_)
                # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1, Kx(NxDxHxW)xC

                raws_slot = raws_slot.view([N, D, H, W, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0,
                                                                                             end_dim=2)  # (NxHxW)xDx4
                masked_raws_slot = masked_raws_slot.view([K, N, D, H, W, 4])
                unmasked_raws_slot = unmasked_raws_slot.view([K, N, D, H, W, 4])

                self.masked_raws_slot = masked_raws_slot
                self.wuv = wuv
                self.KNDHW = (K, N, D, H, W)  # not sure the location of this part

            else:
                raws_slot = None

            if self.opt.weight_pixelfeat:
                rgb_map_, _, _, mask_not_slotted_so_cannot_use, transmittance_cam0_ = raw2outputs(raws_, z_vals_, ray_dir_, render_mask=True,
                                                                            weight_pixelfeat=self.opt.weight_pixelfeat,
                                                                            raws_slot=raws_slot, wuv=wuv,
                                                                            KNDHW=(K, N, D, H, W),
                                                                            div_by_max=self.opt.div_by_max, return_transmittance=True, input_transmittance=None)
                self.transmittance_cam0[..., h::scale, w::scale] = transmittance_cam0_
            else:
                rgb_map_, _, _ = raw2outputs(raws_, z_vals_, ray_dir_, weight_pixelfeat=self.opt.weight_pixelfeat,
                                            raws_slot=raws_slot, wuv=wuv, KNDHW=(K, N, D, H, W),
                                            div_by_max=self.opt.div_by_max,  return_transmittance=False, input_transmittance=None)
            # (NxHxW)x3, (NxHxW)
            rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
            rendered[..., h::scale, w::scale] = rendered_
            x_recon_ = rendered_ * 2 - 1
            x_recon[..., h::scale, w::scale] = x_recon_

        x_recon_novel, x_novel = x_recon[1:], x[1:]
        self.loss_recon = self.L2_loss(x_recon_novel, x_novel)
        self.loss_lpips = self.LPIPS_loss(x_recon_novel, x_novel).mean()
        self.loss_psnr = compute_psnr(x_recon_novel/2+0.5, x_novel/2+0.5, data_range=1.)
        self.loss_ssim = compute_ssim(x_recon_novel/2+0.5, x_novel/2+0.5, data_range=1.)

        with torch.no_grad():
            for i in range(self.opt.n_img_each_scene):
                setattr(self, 'x_rec{}'.format(i), x_recon[i])
                if i == 0:
                    setattr(self, 'input_image', x[i])
                else:
                    setattr(self, 'gt_novel_view{}'.format(i), x[i])
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
                # rgb_map, depth_map, _, mask_map = raw2outputs(raws, z_vals, ray_dir, render_mask=True)
                if self.opt.weight_pixelfeat:
                    raws_slot = self.masked_raws_slot[k]
                    raws_slot = raws_slot.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                    rgb_map, depth_map, _, mask_map = raw2outputs(raws, z_vals, ray_dir, render_mask=True, weight_pixelfeat=self.opt.weight_pixelfeat, raws_slot=raws_slot, wuv=self.wuv, KNDHW=self.KNDHW, div_by_max=self.opt.div_by_max, input_transmittance=self.transmittance_cam0)
                else:
                    rgb_map, depth_map, _, mask_map = raw2outputs(raws, z_vals, ray_dir, render_mask=True, input_transmittance=self.transmittance_cam0)
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