import os

import torchvision.transforms.functional as TF

from data.base_dataset import BaseDataset
from PIL import Image
import torch
import glob
import numpy as np
import random

from torchvision import transforms


class MultiscenesDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=3, output_nc=3)
        parser.add_argument('--n_scenes', type=int, default=100, help='dataset length is #scenes')
        parser.add_argument('--n_img_each_scene', type=int, default=2, help='for each scene, how many images to load in a batch')
        parser.add_argument('--no_shuffle', action='store_true')
        parser.add_argument('--mask_size', type=int, default=128)
        parser.add_argument('--frame5', action='store_true')
        parser.add_argument('--skip', type=int, default=0)
        parser.add_argument('--dataset_nearest_interp', action='store_true')
        parser.add_argument('--dataset_combine_masks', action='store_true')
        parser.add_argument('--color_jitter', action='store_true')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.n_scenes = opt.n_scenes
        self.n_img_each_scene = opt.n_img_each_scene
        self.min_num_masks = self.opt.num_slots if not self.opt.dataset_combine_masks else 4
        image_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*.png')))  # root/00000_sc000_az00_el00.png
        mask_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_mask.png')))
        fg_mask_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_mask_for_moving.png')))
        moved_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_moved.png')))
        bg_mask_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_mask_for_bg.png')))
        bg_in_mask_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_mask_for_providing_bg.png')))
        changed_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_changed.png')))
        bg_in_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_providing_bg.png')))
        changed_filenames_set, bg_in_filenames_set = set(changed_filenames), set(bg_in_filenames)
        bg_mask_filenames_set, bg_in_mask_filenames_set = set(bg_mask_filenames), set(bg_in_mask_filenames)
        image_filenames_set, mask_filenames_set = set(image_filenames), set(mask_filenames)
        fg_mask_filenames_set, moved_filenames_set = set(fg_mask_filenames), set(moved_filenames)
        filenames_set = image_filenames_set - mask_filenames_set - fg_mask_filenames_set - moved_filenames_set - changed_filenames_set - bg_in_filenames_set - bg_mask_filenames_set - bg_in_mask_filenames_set
        filenames = sorted(list(filenames_set))
        self.scenes = []
        skip = self.opt.skip
        for i in range(skip, skip+self.n_scenes):
            if self.opt.frame5:
                scene_filenames = [x for x in filenames if 'sc{:04d}_frame5'.format(i) in x]
            else:
                scene_filenames = [x for x in filenames if 'sc{:04d}'.format(i) in x]
            # print(scene_filenames)
            self.scenes.append(scene_filenames)

        if self.opt.color_jitter:
            self.color_transform = transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)

    def _transform(self, img):
        if self.opt.dataset_nearest_interp:
            img = TF.resize(img, (self.opt.load_size, self.opt.load_size), Image.NEAREST)
        else:
            img = TF.resize(img, (self.opt.load_size, self.opt.load_size))

        if self.opt.color_jitter:

            if self.reset_color_jitter:
                self.reset_color_jitter = False
                self.prev_brightness = float(torch.empty(1).uniform_(0.6, 1.4))
                self.prev_contrast = float(torch.empty(1).uniform_(0.6, 1.4))
                self.prev_saturation = float(torch.empty(1).uniform_(0.6, 1.4))
                self.prev_hue = float(torch.empty(1).uniform_(-0.4, 0.4))

            img = TF.adjust_brightness(img, self.prev_brightness)
            img = TF.adjust_contrast(img, self.prev_contrast)
            img = TF.adjust_saturation(img, self.prev_saturation)
            img = TF.adjust_hue(img, self.prev_hue)

        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5] * img.shape[0], [0.5] * img.shape[0])  # [0,1] -> [-1,1]
        return img

    def _transform_mask(self, img):
        img = TF.resize(img, (self.opt.mask_size, self.opt.mask_size), Image.NEAREST)
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5] * img.shape[0], [0.5] * img.shape[0])  # [0,1] -> [-1,1]
        return img

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        ''' Hash RGB values into unique integer indices '''
        objects = torch.tensor(np.array(objects)).permute(2, 0, 1) # [3, H, W]

        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]

        _, out = torch.unique(out, return_inverse=True)
        out -= out.min()

        out = Image.fromarray(out.squeeze(0).numpy().astype(np.uint8))
        return out

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing, here it is scene_idx
        """
        scene_idx = index
        scene_filenames = self.scenes[scene_idx]
        if self.opt.isTrain and not self.opt.no_shuffle:
            filenames = random.sample(scene_filenames, self.n_img_each_scene)
        else:
            filenames = scene_filenames[:self.n_img_each_scene]
        rets = []
        self.reset_color_jitter = True
        for path in filenames:
            img = Image.open(path).convert('RGB')
            img_data = self._transform(img)
            pose_path = path.replace('.png', '_RT.txt')
            try:
                pose = np.loadtxt(pose_path)
            except FileNotFoundError:
            # except:
                print('filenotfound error: {}'.format(pose_path))
                # pass # it was not here before
                assert False
            pose = torch.tensor(pose, dtype=torch.float32)
            azi_path = pose_path.replace('_RT.txt', '_azi_rot.txt')
            # azi_path = pose_path
            if self.opt.fixed_locality:
                azi_rot = np.eye(3)  # not used; placeholder
            else:
                azi_rot = np.loadtxt(azi_path)
            azi_rot = torch.tensor(azi_rot, dtype=torch.float32)
            depth_path = path.replace('.png', '_depth.npy')
            if os.path.isfile(depth_path):
                depth = np.load(depth_path)  # HxWx1
                depth = torch.from_numpy(depth)  # HxWx1
                depth = depth.permute([2, 0, 1])  # 1xHxW
                ret = {'img_data': img_data, 'path': path, 'cam2world': pose, 'azi_rot': azi_rot, 'depth': depth}
            else:
                ret = {'img_data': img_data, 'path': path, 'cam2world': pose, 'azi_rot': azi_rot}
            mask_path = path.replace('.png', '_mask.png')
            if os.path.isfile(mask_path):
                mask = Image.open(mask_path).convert('RGB')
                # mask_l = mask.convert('L')
                mask_l = self._object_id_hash(mask)
                mask = self._transform_mask(mask)
                ret['mask'] = mask
                mask_l = self._transform_mask(mask_l)
                mask_flat = mask_l.flatten(start_dim=0)  # HW,
                greyscale_dict = mask_flat.unique(sorted=True)  # 8,
                onehot_labels = mask_flat[:, None] == greyscale_dict  # HWx8, one-hot
                onehot_labels = onehot_labels.type(torch.uint8)
                mask_idx = onehot_labels.argmax(dim=1)  # HW
                bg_color = greyscale_dict[0] if 'tdw' in mask_path else greyscale_dict[1]
                fg_idx = mask_flat != bg_color  # HW
                ret['mask_idx'] = mask_idx
                ret['fg_idx'] = fg_idx
                obj_idxs = []
                for i in range(len(greyscale_dict)):
                    obj_idx = mask_l == greyscale_dict[i]  # 1xHxW
                    obj_idxs.append(obj_idx)
                obj_idxs = torch.stack(obj_idxs)  # Kx1xHxW
                ret['obj_idxs'] = obj_idxs  # KxHxW

                # additional attributes: GT background mask and object masks
                ret['bg_mask'] = mask_l == bg_color
                obj_masks = []
                for i in range(len(greyscale_dict)):
                    if greyscale_dict[i] == bg_color:
                        continue
                    obj_mask = mask_l == greyscale_dict[i]  # 1xHxW
                    obj_masks.append(obj_mask)
                obj_masks = torch.stack(obj_masks)  # Kx1xHxW

                # if the number of masks is too small, pad with empty masks
                if obj_masks.shape[0] < self.min_num_masks:
                    obj_masks = torch.cat([obj_masks, torch.zeros_like(obj_masks[0:(self.min_num_masks-obj_masks.shape[0])])], dim=0)

                if obj_masks.shape[0] > self.min_num_masks:
                    print('Error reading file: ', path)
                    return self.buffer_rets

                ret['obj_masks'] = obj_masks  # KxHxW

            rets.append(ret)
        self.buffer_rets = rets
        return rets

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_scenes


def collate_fn(batch):
    # "batch" is a list (len=batch_size) of list (len=n_img_each_scene) of dict
    flat_batch = [item for sublist in batch for item in sublist]
    img_data = torch.stack([x['img_data'] for x in flat_batch])
    paths = [x['path'] for x in flat_batch]
    cam2world = torch.stack([x['cam2world'] for x in flat_batch])
    azi_rot = torch.stack([x['azi_rot'] for x in flat_batch])
    if 'depth' in flat_batch[0]:
        depths = torch.stack([x['depth'] for x in flat_batch])  # Bx1xHxW
    else:
        depths = None
    ret = {
        'img_data': img_data,
        'paths': paths,
        'cam2world': cam2world,
        'azi_rot': azi_rot,
        'depths': depths
    }
    if 'mask' in flat_batch[0]:
        masks = torch.stack([x['mask'] for x in flat_batch])
        ret['masks'] = masks
        mask_idx = torch.stack([x['mask_idx'] for x in flat_batch])
        ret['mask_idx'] = mask_idx
        fg_idx = torch.stack([x['fg_idx'] for x in flat_batch])
        ret['fg_idx'] = fg_idx
        obj_idxs = flat_batch[0]['obj_idxs']  # Kx1xHxW
        ret['obj_idxs'] = obj_idxs
        bg_mask = torch.stack([x['bg_mask'] for x in flat_batch])
        ret['bg_mask'] = bg_mask
        obj_masks = torch.stack([x['obj_masks'].squeeze(1) for x in flat_batch])
        ret['obj_masks'] = obj_masks # [BxKxHXW]

    return ret