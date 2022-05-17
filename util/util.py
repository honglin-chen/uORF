"""This module contains simple helper functions """
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
import itertools

import six
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import skimage.transform
import random
import colorsys
import torch.nn.functional as F
from matplotlib import patches,  lines
from scipy.optimize import linear_sum_assignment



def resize_masks(masks, image_size):
    """
    Resize masks size
    :param masks: tensor of shape (n, 1, h, w)
    :param image_size: H, W
    :return: numpy array of shape (n, H, W)
    """
    masks_n = masks.squeeze()
    masks_resize = np.zeros((masks_n.shape[0], image_size[0], image_size[1]))

    for i in range(masks_n.shape[0]):
        masks_resize[i] = skimage.transform.resize(masks_n[i], image_size, order=3)
        masks_resize[i] = (masks_resize[i]>=0.75).astype('uint8')
    return masks_resize

def mask2bbox(mask):
    inds = (np.where(mask==1))
    xmin, xmax = inds[1].min(), inds[1].max()
    ymin, ymax = inds[0].min(), inds[0].max()
    width = xmax - xmin
    height = ymax - ymin
    rect = patches.Rectangle((xmin, ymin),xmax - xmin, ymax - ymin,linewidth=1,edgecolor='r',facecolor='none')
    return (xmin + width / 2, ymin + height / 2, width, height), rect


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + \
                                  alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_image(image, masks, display_index=False):
    image_mask = image
    colors = random_colors(masks.shape[0])
    for i in range(masks.shape[0]):
        image_mask = apply_mask(image, masks[i], colors[i])
    if display_index:
        image_mask = Image.fromarray(image_mask)
        draw = ImageDraw.Draw(image_mask)
        for i in range(masks.shape[0]):
            min_pixel = 30
            if masks[i].sum() > min_pixel:
                mask = masks[i]
                mask_eroded = np.array(Image.fromarray(mask).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3)))
                if mask_eroded.sum() <= min_pixel - 20:
                    continue
                (x_center, y_center, _, _), rect = mask2bbox(mask_eroded)
                # draw.text((x_center, y_center), str(i), (0, 0, 0))
                # import ipdb; ipdb.set_trace()
                draw.rectangle(list(rect.get_bbox().get_points().reshape(-1)))
        image_mask = np.array(image_mask)


    return image_mask

def get_colormap(rgb=True):
    color_list = np.array(
        [
            0, 0, 0.5625
            , 0, 0, 0.6250
            , 0, 0, 0.6875
            , 0, 0, 0.7500
            , 0, 0, 0.8125
            , 0, 0, 0.8750
            , 0, 0, 0.9375
            , 0, 0, 1.0000
            , 0, 0.0625, 1.0000
            , 0, 0.1250, 1.0000
            , 0, 0.1875, 1.0000
            , 0, 0.2500, 1.0000
            , 0, 0.3125, 1.0000
            , 0, 0.3750, 1.0000
            , 0, 0.4375, 1.0000
            , 0, 0.5000, 1.0000
            , 0, 0.5625, 1.0000
            , 0, 0.6250, 1.0000
            , 0, 0.6875, 1.0000
            , 0, 0.7500, 1.0000
            , 0, 0.8125, 1.0000
            , 0, 0.8750, 1.0000
            , 0, 0.9375, 1.0000
            , 0, 1.0000, 1.0000
            , 0.0625, 1.0000, 0.9375
            , 0.1250, 1.0000, 0.8750
            , 0.1875, 1.0000, 0.8125
            , 0.2500, 1.0000, 0.7500
            , 0.3125, 1.0000, 0.6875
            , 0.3750, 1.0000, 0.6250
            , 0.4375, 1.0000, 0.5625
            , 0.5000, 1.0000, 0.5000
            , 0.5625, 1.0000, 0.4375
            , 0.6250, 1.0000, 0.3750
            , 0.6875, 1.0000, 0.3125
            , 0.7500, 1.0000, 0.2500
            , 0.8125, 1.0000, 0.1875
            , 0.8750, 1.0000, 0.1250
            , 0.9375, 1.0000, 0.0625
            , 1.0000, 1.0000, 0
            , 1.0000, 0.9375, 0
            , 1.0000, 0.8750, 0
            , 1.0000, 0.8125, 0
            , 1.0000, 0.7500, 0
            , 1.0000, 0.6875, 0
            , 1.0000, 0.6250, 0
            , 1.0000, 0.5625, 0
            , 1.0000, 0.5000, 0
            , 1.0000, 0.4375, 0
            , 1.0000, 0.3750, 0
            , 1.0000, 0.3125, 0
            , 1.0000, 0.2500, 0
            , 1.0000, 0.1875, 0
            , 1.0000, 0.1250, 0
            , 1.0000, 0.0625, 0
            , 1.0000, 0, 0
            , 0.9375, 0, 0
            , 0.8750, 0, 0
            , 0.8125, 0, 0
            , 0.7500, 0, 0
            , 0.6875, 0, 0
            , 0.6250, 0, 0
            , 0.5625, 0, 0
            , 0.5000, 0, 0
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3))
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

def tensor2im(input_image, imtype=np.uint8, use_color_map=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array, range=[-1,1], CxHxW
        imtype (type)        --  the desired type of the converted numpy array
        use_color_map: if True, when inputting grayscale (n_ch==1), do color mapping
    output:
        image_numpy: HxWx3
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0  # now HxWx3, [0,1]
        if image_numpy.shape[2] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (1, 1, 3))
            if use_color_map:
                mapped = image_numpy.copy()
                colormap = get_colormap()  # 64-bin color map, 64x3
                img = image_numpy[:, :, 0]
                grid = np.linspace(0, 1, 65)
                for i in range(64):
                    lower, upper = grid[i], grid[i + 1]
                    img_pos = (img <= upper) & (img >= lower)
                    mapped[img_pos, :] = colormap[i, :]
                image_numpy = mapped
        image_numpy *= 255.0
        image_numpy = image_numpy.astype(imtype)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image.astype(imtype)
    return image_numpy

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def print_tensor(x):
    print('mean={}, min={}, max={}, median={}, std={}'.format(x.mean(), x.min(), x.max(), x.median(), x.std()))

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    val = 0
    avg = 0
    sum = 0
    count = 0
    tot_count = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count

class GroupMeters(object):
    def __init__(self):
        self._meters = collections.defaultdict(AverageMeter)

    def reset(self):
        list(map((AverageMeter.reset, self._meters.values())))

    def update(self, updates=None, value=None, n=1, **kwargs):
        """
        Example:
            >>> meters.update(key, value)
            >>> meters.update({key1: value1, key2: value2})
            >>> meters.update(key1=value1, key2=value2)
        """
        if updates is None:
            updates = {}
        if updates is not None and value is not None:
            updates = {updates: value}
        updates.update(kwargs)
        for k, v in updates.items():
            self._meters[k].update(v, n=n)

    def __getitem__(self, name):
        return self._meters[name]

    def items(self):
        return self._meters.items()

    @property
    def sum(self):
        return {k: m.sum for k, m in self._meters.items() if m.count > 0}

    @property
    def avg(self):
        return {k: m.avg for k, m in self._meters.items() if m.count > 0}

    @property
    def val(self):
        return {k: m.val for k, m in self._meters.items() if m.count > 0}

    def format(self, caption, values, kv_format, glue):
        meters_kv = self._canonize_values(values)
        log_str = [caption]
        log_str.extend(itertools.starmap(kv_format.format, sorted(meters_kv.items())))
        return glue.join(log_str)

    def format_simple(self, caption, values='avg', compressed=True):
        if compressed:
            return self.format(caption, values, '{}={:4f}', ' ')
        else:
            return self.format(caption, values, '\t{} = {:4f}', '\n')

    def _canonize_values(self, values):
        if isinstance(values, six.string_types):
            assert values in ('avg', 'val', 'sum')
            meters_kv = getattr(self, values)
        else:
            meters_kv = values
        return meters_kv

def compute_dice_cost(inputs, targets):
    """
    Compute the DICE cost, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of shape [B, K, H, W]
        targets: A float tensor with shape [B, O, H, W].
    """
    inputs = inputs.flatten(2, 3).unsqueeze(2)  # [B, K, 1, N]
    targets = targets.flatten(2, 3).unsqueeze(1)  # [B, 1, O, N]

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def silhouette_hungarian_matching(mask_a, mask_b, eps=1e-9):
    """
        Compute the hungarian between two masks based on dice cost
        Args:
            mask_a: A float tensor of shape [B, K, H, W]
            mask_b: A float tensor with shape [B, O, H, W]. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
    """

    B, K, H, W = mask_a.shape
    _, O, _, _ = mask_b.shape

    dice_cost = compute_dice_cost(mask_a, mask_b)
    matched_a, matched_b = [], []

    for batch_idx in range(dice_cost.shape[0]):
        indices = linear_sum_assignment(dice_cost[batch_idx].detach().cpu())
        matched_a.append(mask_a[batch_idx, indices[0]].reshape(1, K, H, W))
        matched_b.append(mask_b[batch_idx, indices[1]].reshape(1, O, H, W))

    matched_a = torch.cat(matched_a, 0)
    matched_b = torch.cat(matched_b, 0)

    # # Visualization
    # for i in range(B):
    #     fig, axs = plt.subplots(2, K)
    #     for k in range(K):
    #         axs[0, k].imshow(matched_a[i, k].detach().cpu())
    #         axs[1, k].imshow(matched_b[i, k].detach().cpu())
    #
    #     plt.show()
    #     plt.savefig('tmp%d.png' % i)
    #     plt.close()

    return matched_a, matched_b