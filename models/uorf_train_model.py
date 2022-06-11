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
from .model import get_perceptual_net
from models.morf.end2end import MorfEnd2end


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

        parser.add_argument('--mask_image', action='store_true')
        parser.add_argument('--mask_image_feature', action='store_true')
        parser.add_argument('--use_slot_feat', action='store_true')
        parser.add_argument('--use_pixel_feat', action='store_true')
        parser.add_argument('--gt_seg', action='store_true', help='use GT segments')
        parser.add_argument('--focal_ratio', nargs='+', default=(350. / 320., 350. / 240.), help='set the focal ratio in projection.py')

        parser.add_argument('--use_ray_dir_world', action='store_true')

        parser.add_argument('--unisurf_render_eq', action='store_true',
                            help='use unisurf style rendering equation, and treat the output of decoder as alpha, not density')
        parser.add_argument('--visualize_occl_silhouette', action='store_true',
                            help='occluded silhouette is computed at visualization epoch using rendering equation (making white color)')
        parser.add_argument('--visualize_unoccl_silhouette', action='store_true',
                            help='unoccluded silhouette is computed at visualization epoch using uorf raw2output')
        parser.add_argument('--visualize_mask', action='store_true',
                            help='visualize loaded masks')
        parser.add_argument('--visualize_attn', action='store_true',
                            help='visualize attn of uORF')

        parser.add_argument('--use_occl_silhouette_loss', action='store_true',
                            help='This will compute the occluded silhouette every epoch, and apply silhouette loss on it')
        parser.add_argument('--use_unoccl_silhouette_loss', action='store_true',
                            help='This will compute the unoccluded silhouette every epoch, and apply silhouette loss on it')

        parser.add_argument('--use_voxel_feat', action='store_true',
                            help='Not implemented yet. This might be preliminary step for instant ngp')

        parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
                            dataset_mode='multiscenes', niter=1200, custom_lr=True, lr_policy='warmup')

        parser.set_defaults(exp_id='run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))

        return parser

    def __init__(self, opt):
        super().__init__(opt)
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['recon', 'perc']
        if self.opt.use_occl_silhouette_loss:
            self.loss_names += ['silhouette']
        n = opt.n_img_each_scene
        self.visual_names = ['x{}'.format(i) for i in range(n)] + \
                            ['x_rec{}'.format(i) for i in range(n)] + \
                            ['slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)] + \
                            ['unmasked_slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)]
        if self.opt.visualize_occl_silhouette:
            self.visual_names += ['occl_silhouette_slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)]
        if self.opt.visualize_unoccl_silhouette:
            self.visual_names += ['unoccl_silhouette_slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)]
        self.visual_names += ['depth_map{}'.format(i) for i in range(n)]
        if self.opt.visualize_mask:
            self.visual_names += ['mask_slot{}_view{}'.format(k, i) for k in range(opt.num_slots) for i in range(n)]
        if self.opt.visualize_attn:
            raise NotImplementedError # this is for uORF
        # print('You will visualize these images \n', self.visual_names)
        self.model_names = ['End2end']
        self.perceptual_net = get_perceptual_net().cuda()
        self.vgg_norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.netEnd2end = networks.init_net(MorfEnd2end(opt), gpu_ids=self.gpu_ids, init_type=None)

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        if self.isTrain:  # only defined during training time
            self.optimizer = optim.Adam(chain(
                self.netEnd2end.parameters(),
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
        load_size = self.opt.load_size
        x = input['img_data'].to(self.device).view(NS, NV, 3, load_size, load_size)
        cam2world = input['cam2world'].to(self.device).view(NS, NV, 4, 4)
        if not self.opt.fixed_locality:
            cam2world_azi = input['azi_rot'].to(self.device).view(NS, NV, 3, 3)

        ## Process segmentation masks of the input view
        if 'masks' in input.keys():
            bg_masks = input['bg_mask'].to(self.device)  # [NSxNV, 1, h, w]
            obj_masks = input['obj_masks'].to(self.device)  # [NSxNV, K-1, h, w]
            # bg_masks = bg_masks.view(NS, NV, 1, H, W)[:, 0]  # [NS, 1, h, w]
            # obj_masks = obj_masks.view(NS, NV, K - 1, H, W)[:, 0]  # [NS, K-1, h, w]
            masks = torch.cat([bg_masks, obj_masks], dim=1)  # [NS, K, h, w]
            masks = F.interpolate(masks.float(), size=[self.opt.input_size, self.opt.input_size], mode='nearest')
            masks = masks.view(NS, NV, K, H, W)
        return x, cam2world, cam2world_azi, masks

    def forward(self, x, cam2world, cam2world_azi, masks, epoch=0, iter=0):
        B, NV, C, H, W = x.shape
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        vis_dict = None
        self.weight_percept = self.opt.weight_percept if epoch >= self.opt.percept_in else 0

        input_end2end = {'input_img': x,
                         'input_mask': masks,
                         'cam2world': cam2world,
                         'cam2world_azi': cam2world_azi}

        output_end2end = self.netEnd2end(input_end2end)

        x_recon = output_end2end['x_recon']
        rendered = (x_recon + 1) / 2
        attn = output_end2end['output_mask']
        masked_raws = output_end2end['weighted_raws']
        unmasked_raws = output_end2end['unweighted_raws']
        depth_map = output_end2end['depth_map']
        x = output_end2end['x_supervision']
        occl_silhouettes = output_end2end['occl_silhouettes'] if self.opt.use_occl_silhouette_loss else None
        if self.opt.use_unoccl_silhouette_loss:
            raise NotImplementedError

        self.loss_recon = self.L2_loss(x_recon, x)
        x_norm, rendered_norm = self.vgg_norm((x.flatten(0, 1) + 1) / 2), self.vgg_norm(rendered.flatten(0, 1))
        rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
        self.loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)
        if self.opt.use_occl_silhouette_loss:
            assert self.opt.gt_seg
            self.loss_silhouette = self.L2_loss(occl_silhouettes, attn.view(B, NV, self.opt.num_slots, 1,
                                                                            self.opt.supervision_size,
                                                                            self.opt.supervision_size))
        else:
            self.loss_silhouette = 0

        with torch.no_grad():
            attn = attn[0].detach()  # KxN, only the first in batch
            H_, W_ = self.opt.input_size, self.opt.input_size
            attn = attn.view(self.opt.num_slots, 1, H_, W_)
            if H_ != H:
                attn = F.interpolate(attn, size=[H, W], mode='bilinear')
            for i in range(self.opt.n_img_each_scene):
                setattr(self, 'x_rec{}'.format(i), x_recon[0:1, i]) # only the first in batch
                setattr(self, 'x{}'.format(i), x[0:1, i]) # only the first in batch
                setattr(self, 'depth_map{}'.format(i), depth_map[0:1, i])
            setattr(self, 'masked_raws', masked_raws[0].detach()) # only the first in batch
            setattr(self, 'unmasked_raws', unmasked_raws[0].detach()) # only the first in batch
            setattr(self, 'attn', attn[0].detach())
            setattr(self, 'masks', F.interpolate(masks[0].detach(),
                                                 size=[self.opt.supervision_size, self.opt.supervision_size],
                                                 mode='nearest')) # NVxKxHxW
            if self.opt.visualize_occl_silhouette:
                assert self.opt.use_occl_silhouette_loss
                setattr(self, 'occl_silhouettes', occl_silhouettes[0].detach())

        if iter % self.opt.display_freq == 0:
            self.compute_visuals()
            vis_dict = self.get_current_visuals()

        output = {'loss_recon': self.loss_recon,
                  'loss_perc': self.loss_perc,
                  'loss_silhouette': self.loss_silhouette,
                  'vis_dict': vis_dict}
        return output

    def compute_visuals(self):
        with torch.no_grad():
            _, N, D, H, W, _ = self.masked_raws.shape
            masked_raws = self.masked_raws  # KxNxDxHxWx4
            unmasked_raws = self.unmasked_raws  # KxNxDxHxWx4

            for k in range(self.opt.num_slots):
                output_visual = self.netEnd2end.renderer.compute_visual(masked_raws[k])
                x_recon = output_visual['x_recon']
                mask_map = output_visual['unoccl_silhouette']
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i].unsqueeze(0))
                    if self.opt.visualize_unoccl_silhouette:
                        setattr(self, 'unoccl_silhouette_slot{}_view{}'.format(k, i), mask_map.view(N, 1, H, W)[i].unsqueeze(0))
                    if self.opt.visualize_occl_silhouette:
                        setattr(self, 'occl_silhouette_slot{}_view{}'.format(k, i), self.occl_silhouettes[k][i].unsqueeze(0))
                    if self.opt.visualize_mask:
                        setattr(self, 'mask_slot{}_view{}'.format(k, i), self.masks[i][k].unsqueeze(0).unsqueeze(0))

            for k in range(self.opt.num_slots):
                output_visual = self.netEnd2end.renderer.compute_visual(unmasked_raws[k])
                x_recon = output_visual['x_recon']
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'unmasked_slot{}_view{}'.format(k, i), x_recon[i].unsqueeze(0))

            if self.opt.visualize_attn:
                raise NotImplementedError
                # for k in range(self.opt.num_slots):
                #     print(self.attn[k].unsqueeze(0).shape, 'attn')
                #     setattr(self, 'slot{}_attn'.format(k), self.attn[k].unsqueeze(0) * 2 - 1)

    def optimize_parameters(self, loss, ret_grad=False, epoch=0):
        """Update network weights; it will be called in every training iteration."""
        for opm in self.optimizers:
            opm.zero_grad()
        loss.backward()
        avg_grads = []
        layers = []
        if ret_grad:
            for n, p in chain(self.netEnd2end.named_parameters(),):
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