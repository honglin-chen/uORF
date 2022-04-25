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

class pixelnerfNoGanModel(BaseModel):

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
        parser.add_argument('--no_use_background', action='store_true', help='do not use backbround')
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
        self.loss_names = ['recon', 'perc']
        n = opt.n_img_each_scene
        num_slots = opt.num_slots
        if opt.no_use_background:
            num_slots = opt.num_slots - 1
        self.visual_names = ['x{}'.format(i) for i in range(n)] + \
                            ['x_rec{}'.format(i) for i in range(n)] + \
                            ['slot{}_view{}'.format(k, i) for k in range(num_slots) for i in range(n)] + \
                            ['unmasked_slot{}_view{}'.format(k, i) for k in range(num_slots) for i in range(n)] + \
                            ['slot{}_attn'.format(k) for k in range(num_slots)]
        self.model_names = ['PixelEncoder', 'PixelDecoder']
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
        parameters = []
        self.nets = []
        self.num_slots = opt.num_slots
        # self.netEncoder = networks.init_net(Encoder(3, z_dim=z_dim, bottom=opt.bottom),
        #                                     gpu_ids=self.gpu_ids, init_type='normal')
        self.netSlotAttention = networks.init_net(
            SlotAttention(num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter, gt_seg=opt.gt_seg), gpu_ids=self.gpu_ids, init_type='normal')
        parameters.append(self.netSlotAttention.parameters())
        # self.nets.append(self.netSlotAttention)
        # parameters.extend([self.netEncoder.parameters(), self.netSlotAttention.parameters()])
        if self.opt.pixel_encoder:
            self.netPixelEncoder = networks.init_net(PixelEncoder(reduce_latent_size=False), gpu_ids=self.gpu_ids, init_type='None')
            parameters.append(self.netPixelEncoder.parameters())
            self.nets.append(self.netPixelEncoder)
            self.netDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, pixel_dim=512, z_dim=opt.z_dim, n_layers=opt.n_layer,
                                                        locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality), gpu_ids=self.gpu_ids, init_type='xavier')
            # parameters.append(self.netDecoder.parameters())
            if self.opt.pixel_decoder:
                self.netPixelDecoder = networks.init_net(PixelDecoder(n_freq=opt.n_freq, input_dim=512, z_dim=opt.z_dim, n_layers=opt.n_layer,
                                                    locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality, pixel_positional_encoding=True, use_ray_dir=True), gpu_ids=self.gpu_ids, init_type='None')
                parameters.append(self.netPixelDecoder.parameters())
                self.nets.append(self.netPixelDecoder)

        else:
            pass
            # self.netDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=opt.z_dim, n_layers=opt.n_layer,
            #                                             locality_ratio=opt.obj_scale / opt.nss_scale, fixed_locality=opt.fixed_locality), gpu_ids=self.gpu_ids, init_type='xavier')
            # parameters.append(self.netDecoder.parameters())

        self.parameters = parameters
        if self.isTrain:  # only defined during training time
            self.optimizer = optim.Adam(chain(*self.parameters), lr=opt.lr)
            self.optimizers = [self.optimizer]

        self.L2_loss = nn.MSELoss()

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
            masks = F.interpolate(masks.float(), size=[64, 64], mode='nearest')
            self.masks = masks.flatten(2, 3)

    def forward(self, epoch=0):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.weight_percept = self.opt.weight_percept if epoch >= self.opt.percept_in else 0
        self.loss_recon = 0
        self.loss_perc = 0
        dev = self.x[0:1].device
        nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()

        # Encoding images
        # feature_map = self.netEncoder(F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False))  # BxCxHxW
        if self.opt.pixel_encoder:
            feature_map_pixel = self.netPixelEncoder(F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False)) # BxCxHxW
        # feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC

        # Slot Attention
        # z_slots, attn = self.netSlotAttention(feat, masks=self.masks)  # 1xKxC, 1xKxN
        z_slots, attn = torch.ones([1, 2, feature_map_pixel.shape[1]]), torch.ones([1, 2, feature_map_pixel.shape[2]*feature_map_pixel.shape[3]])
        z_slots, attn = z_slots.squeeze(0), attn.squeeze(0)  # KxC, KxN
        # K = attn.shape[0]
        K = 2
        z_slots = None

        cam2world = self.cam2world
        N = cam2world.shape[0]
        if self.opt.stage == 'coarse':
            W, H, D = self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp
            frus_nss_coor, z_vals, ray_dir, frus_world_coor = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            self.cam2spixel = self.projection.cam2spixel
            self.world2nss = self.projection.world2nss
            frustum_size = torch.Tensor(self.projection.frustum_size).to(self.device)
            self.netDecoder.frustum_size = frustum_size
            x = F.interpolate(self.x, size=self.opt.supervision_size, mode='bilinear', align_corners=False)
            self.z_vals, self.ray_dir = z_vals, ray_dir
            ray_dir_input = ray_dir.view([N, H, W, 3])
            ray_dir_input = ray_dir_input[:, None, ...].expand(-1, D, -1, -1, -1).flatten(0, 3)
        else:
            W, H, D = self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp
            start_range = self.opt.frustum_size_fine - self.opt.render_size
            rs = self.opt.render_size
            frus_nss_coor, z_vals, ray_dir, frus_world_coor = self.projection_fine.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            self.cam2spixel = self.projection_fine.cam2spixel
            self.world2nss = self.projection_fine.world2nss
            frustum_size = torch.Tensor(self.projection_fine.frustum_size).to(self.device)
            self.netDecoder.frustum_size = frustum_size
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor.view([N, D, H, W, 3]), z_vals.view([N, H, W, D]), ray_dir.view([N, H, W, 3])
            H_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            W_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            frus_nss_coor_, z_vals_, ray_dir_ = frus_nss_coor[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], z_vals[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], ray_dir[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor_.flatten(0, 3), z_vals_.flatten(0, 2), ray_dir_.flatten(0, 2)
            x = self.x[:, :, H_idx:H_idx + rs, W_idx:W_idx + rs]
            self.z_vals, self.ray_dir = z_vals, ray_dir
            ray_dir_input = ray_dir.view([N, H, W, 3])
            ray_dir_input = ray_dir_input[:, None, ...].expand(-1, D, -1, -1, -1).flatten(0, 3)

        sampling_coor_fg = frus_nss_coor[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
        sampling_coor_bg = frus_nss_coor  # Px3

        W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp
        if self.opt.pixel_encoder:
            # construct uv in the first image coordinates
            cam02world = cam2world[0:1] # 1x4x4
            world2cam0 = cam02world.inverse() # 1x4x4
            nss2world = self.world2nss.inverse() # 1x4x4
            frus_nss_coor = torch.cat([frus_nss_coor, torch.ones_like(frus_nss_coor[:, 0].unsqueeze(1))], dim=-1) # Px4
            frus_world_coor = torch.matmul(nss2world[None, ...], frus_nss_coor[None, ..., None]) # 1xPx4x1
            frus_cam0_coor = torch.matmul(world2cam0[None, ...], frus_world_coor) #1x1x4x4, 1x(NxWxHxD)x4x1 -> 1x(NxWxHxD)x4x1
            pixel_cam0_coor = torch.matmul(self.cam2spixel[None, ...], frus_cam0_coor) # 1x1x4x4, 1x(NxWxHxD)x4x1
            pixel_cam0_coor = pixel_cam0_coor.squeeze(-1) # 1x(NxWxHxD)x4
            uv = pixel_cam0_coor[:, :, 0:2]/pixel_cam0_coor[:, :, 2].unsqueeze(-1) # 1x(NxWxHxD)x2
            uv = (uv/frustum_size[0:2][None, None, :] - 0.5) * 2 # nomalize to fit in with [-1, 1] grid #
            '''
            tensor(3.4756, device='cuda:0') tensor(-4.4286, device='cuda:0') tensor(-0.0233, device='cuda:0') tensor(-0.0241, device='cuda:0') tensor(0.6644, device='cuda:0') u max min median mean std
            tensor(12.8928, device='cuda:0') tensor(-1.7501, device='cuda:0') tensor(1.1921e-07, device='cuda:0') tensor(0.1183, device='cuda:0') tensor(0.9658, device='cuda:0') v max min median mean std
            '''
            pixel_feat = self.netPixelEncoder.index(uv) # 1x(NxWxHxD)x2 -> 1xCx(NxWxHxD)
            pixel_feat = pixel_feat.transpose(1, 2) # 1x(NxWxHxD)xC
            pixel_feat = pixel_feat.expand(K, -1, -1) # Kx(NxWxHxD)xC


            if self.opt.pixel_decoder:
                use_background = False if self.opt.no_use_background else True
                raws, masked_raws, unmasked_raws, masks = \
                    self.netPixelDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0, pixel_feat, ray_dir_input, use_background)
                # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1, Kx(NxDxHxW)xC
            # else:
            #     raws, masked_raws, unmasked_raws, masks = \
            #         self.netDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0, pixel_feat)
            #                     # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1, Kx(NxDxHxW)xC
        # else:
        #     raws, masked_raws, unmasked_raws, masks = \
        #         self.netDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0)
            # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1, Kx(NxDxHxW)xC

        if self.opt.no_use_background:
            K = K-1
            attn = attn[0, ...]

        raws = raws.view([N, D, H, W, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
        masked_raws = masked_raws.view([K, N, D, H, W, 4])
        unmasked_raws = unmasked_raws.view([K, N, D, H, W, 4])
        rgb_map, _, _ = raw2outputs(raws, z_vals, ray_dir)
        # (NxHxW)x3, (NxHxW)
        rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
        x_recon = rendered * 2 - 1

        self.loss_recon = self.L2_loss(x_recon, x)
        x_norm, rendered_norm = self.vgg_norm((x + 1) / 2), self.vgg_norm(rendered)
        rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
        self.loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)

        with torch.no_grad():
            attn = attn.detach().cpu()  # KxN
            H_, W_ = feature_map_pixel.shape[2], feature_map_pixel.shape[3]
            # H_, W_ = pixel_feat.shape[2], pixel_feat.shape[3]
            num_slots = self.opt.num_slots
            if self.opt.no_use_background:
                num_slots += -1
            attn = attn.view(num_slots, 1, H_, W_)
            if H_ != H:
                pass
                attn = F.interpolate(attn, size=[H, W], mode='bilinear')
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
            num_slots = self.num_slots
            if self.opt.no_use_background:
                num_slots += -1
            for k in range(num_slots):
                raws = masked_raws[k]  # NxDxHxWx4
                z_vals, ray_dir = self.z_vals, self.ray_dir
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i])

                raws = unmasked_raws[k]  # (NxDxHxW)x4
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'unmasked_slot{}_view{}'.format(k, i), x_recon[i])

                    setattr(self, 'slot{}_attn'.format(k), self.attn[k] * 2 - 1)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        loss = self.loss_recon + self.loss_perc
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