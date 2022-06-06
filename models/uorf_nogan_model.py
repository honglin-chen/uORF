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
from .model import Encoder, Decoder, SlotAttention, get_perceptual_net, raw2outputs


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
        parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
        parser.add_argument('--input_size', type=int, default=64)
        parser.add_argument('--frustum_size', type=int, default=64)
        parser.add_argument('--frustum_size_fine', type=int, default=128)
        parser.add_argument('--attn_decay_steps', type=int, default=2e5)
        parser.add_argument('--coarse_epoch', type=int, default=600)
        parser.add_argument('--near_plane', type=float, default=6)
        parser.add_argument('--far_plane', type=float, default=20)
        parser.add_argument('--fixed_locality', action='store_true', help='enforce locality in world space instead of transformed view space')
        parser.add_argument('--gt_seg', action='store_true', help='use GT segments')

        parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
                            dataset_mode='multiscenes', niter=1200, custom_lr=True, lr_policy='warmup')

        parser.set_defaults(exp_id='run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))

        return parser

    def __init__(self, opt):
        super(uorfNoGanModel).__init__()
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
        self.visual_names = ['x{}'.format(i) for i in range(n)] + \
                            ['x_rec{}'.format(i) for i in range(n)] + \
                            ['slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                            ['unmasked_slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                            ['slot{}_attn'.format(k) for k in range(opt.num_slots)]
        self.model_names = ['Encoder', 'SlotAttention', 'Decoder']
        self.perceptual_net = get_perceptual_net().cuda()
        self.vgg_norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
        self.projection = Projection(device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        frustum_size_fine = [self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp]
        self.projection_fine = Projection(device=self.device, nss_scale=opt.nss_scale,
                                          frustum_size=frustum_size_fine, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        z_dim = opt.z_dim
        self.num_slots = opt.num_slots
        self.netEncoder = networks.init_net(Encoder(3, z_dim=z_dim, bottom=opt.bottom),
                                            gpu_ids=self.gpu_ids, init_type='normal')
        self.netSlotAttention = networks.init_net(
            SlotAttention(num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter, gt_seg=opt.gt_seg), gpu_ids=self.gpu_ids, init_type='normal')
        self.netDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=opt.z_dim, n_layers=opt.n_layer,
                                                    locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality), gpu_ids=self.gpu_ids, init_type='xavier')

        if self.isTrain:  # only defined during training time
            self.optimizer = optim.Adam(chain(
                self.netEncoder.parameters(), self.netSlotAttention.parameters(), self.netDecoder.parameters()
            ), lr=opt.lr)
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
        NS, NV = self.opt.batch_size, self.opt.n_img_each_scene
        H = W = self.opt.load_size
        K = self.opt.num_slots
        ## Process Input image and camera parameters
        x = input['img_data'].to(self.device).view(NS, NV, 3, H, W)
        cam2world = input['cam2world'].to(self.device).view(NS, NV, 4, 4)

        if not self.opt.fixed_locality:
            cam2world_azi = input['azi_rot'].to(self.device).view(NS, NV, 3, 3)

        ## Process segmentation masks of the input view
        if 'masks' in input.keys():
            bg_masks = input['bg_mask'].to(self.device) # [NSxNV, 1, h, w]
            obj_masks = input['obj_masks'].to(self.device) # [NSxNV, K-1, h, w]
            bg_masks = bg_masks.view(NS, NV, 1, H, W)[:, 0]  # [NS, 1, h, w]
            obj_masks = obj_masks.view(NS, NV, K-1, H, W)[:, 0] # [NS, K-1, h, w]
            masks = torch.cat([bg_masks, obj_masks], dim=1) # [NS, K, h, w]
            masks = F.interpolate(masks.float(), size=[self.opt.input_size, self.opt.input_size], mode='nearest')
        return x, cam2world, cam2world_azi, masks

    def forward(self, x, cam2world, cam2world_azi, masks=None, epoch=0, iter=0):
        B, NV, C, H, W = x.shape
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        vis_dict = None
        self.weight_percept = self.opt.weight_percept if epoch >= self.opt.percept_in else 0
        self.loss_recon = 0
        self.loss_perc = 0
        dev = x[:, 0:1].device
        nss2cam0 = cam2world[:, 0:1].inverse() if self.opt.fixed_locality else cam2world_azi[:, 0:1].inverse()

        # Encoding images
        feature_map = self.netEncoder(F.interpolate(x[:, 0:1].flatten(0, 1), size=self.opt.input_size,
                                                    mode='bilinear', align_corners=False))  # BxCxHxW
        feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC

        # Slot Attention
        z_slots, attn = self.netSlotAttention(feat, masks=masks)  # BxKxC, BxKxN
        K = attn.shape[1]
        N = cam2world.shape[1]
        if self.opt.stage == 'coarse':
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            x = F.interpolate(x.flatten(0, 1), size=self.opt.supervision_size, mode='bilinear', align_corners=False)
            x = x.reshape(B, NV, 3, self.opt.supervision_size, self.opt.supervision_size)
            self.z_vals, self.ray_dir = z_vals, ray_dir
        else:
            W, H, D = self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp
            start_range = self.opt.frustum_size_fine - self.opt.render_size
            rs = self.opt.render_size
            frus_nss_coor, z_vals, ray_dir = self.projection_fine.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor.view([B, N, D, H, W, 3]), z_vals.view([B, N, H, W, D]), ray_dir.view([B, N, H, W, 3])
            H_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            W_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            frus_nss_coor_, z_vals_, ray_dir_ = frus_nss_coor[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], z_vals[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], ray_dir[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor_.flatten(1, 4), z_vals_.flatten(1, 3), ray_dir_.flatten(1, 3)
            x = x[:, :, :, H_idx:H_idx + rs, W_idx:W_idx + rs]
            self.z_vals, self.ray_dir = z_vals, ray_dir

        sampling_coor_fg = frus_nss_coor[:, None, ...].expand(-1, K - 1, -1, -1)  # (K-1)xPx3
        sampling_coor_bg = frus_nss_coor  # BxPx3

        W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp
        raws, masked_raws, unmasked_raws, masks = self.netDecoder(sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0)  # Bx(NxDxHxW)x4, BxKx(NxDxHxW)x4, BxKx(NxDxHxW)x4, BxKx(NxDxHxW)x1
        raws = raws.view([B, N, D, H, W, 4]).permute([0, 1, 3, 4, 2, 5]).flatten(start_dim=1, end_dim=3)  # Bx(NxHxW)xDx4
        masked_raws = masked_raws.view([B, K, N, D, H, W, 4])
        unmasked_raws = unmasked_raws.view([B, K, N, D, H, W, 4])
        rgb_map, _, _ = raw2outputs(raws, z_vals, ray_dir)
        # Bx(NxHxW)x3, Bx(NxHxW)

        rendered = rgb_map.view(B, N, H, W, 3).permute([0, 1, 4, 2, 3])  # BxNx3xHxW
        x_recon = rendered * 2 - 1

        self.loss_recon = self.L2_loss(x_recon, x)
        x_norm, rendered_norm = self.vgg_norm((x.flatten(0, 1) + 1) / 2), self.vgg_norm(rendered.flatten(0, 1))
        rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
        self.loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)

        with torch.no_grad():
            attn = attn[0].detach()  # KxN, only the first in batch
            H_, W_ = feature_map[0:1].shape[2], feature_map.shape[3]
            attn = attn.view(self.opt.num_slots, 1, H_, W_)
            if H_ != H:
                attn = F.interpolate(attn, size=[H, W], mode='bilinear')
            for i in range(self.opt.n_img_each_scene):
                setattr(self, 'x_rec{}'.format(i), x_recon[0:1, i]) # only the first in batch
                setattr(self, 'x{}'.format(i), x[0:1, i]) # only the first in batch
            setattr(self, 'masked_raws', masked_raws[0].detach()) # only the first in batch
            setattr(self, 'unmasked_raws', unmasked_raws[0].detach()) # only the first in batch
            setattr(self, 'attn', attn)

        if iter % self.opt.display_freq == 0:
            self.compute_visuals()
            vis_dict = self.get_current_visuals()

        return self.loss_recon, self.loss_perc, vis_dict

    def compute_visuals(self):
        with torch.no_grad():
            _, N, D, H, W, _ = self.masked_raws.shape
            masked_raws = self.masked_raws  # KxNxDxHxWx4
            unmasked_raws = self.unmasked_raws  # KxNxDxHxWx4
            for k in range(self.num_slots):
                raws = masked_raws[k]  # NxDxHxWx4
                z_vals, ray_dir = self.z_vals[0:1], self.ray_dir[0:1]
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _ = raw2outputs(raws.unsqueeze(0), z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i].unsqueeze(0))

                raws = unmasked_raws[k]  # (NxDxHxW)x4
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _ = raw2outputs(raws.unsqueeze(0), z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'unmasked_slot{}_view{}'.format(k, i), x_recon[i].unsqueeze(0))

                setattr(self, 'slot{}_attn'.format(k), self.attn[k].unsqueeze(0) * 2 - 1)

    def optimize_parameters(self, loss, ret_grad=False, epoch=0):
        """Update network weights; it will be called in every training iteration."""
        for opm in self.optimizers:
            opm.zero_grad()
        loss.backward()
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