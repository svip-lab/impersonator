from __future__ import print_function
from PIL import Image
import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import math
import pickle


class ImageTransformer(object):
    """
    Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        """
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is matched to output_size.
                            If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        images = sample['images']
        resized_images = []

        for image in images:
            image = cv2.resize(image, (self.output_size, self.output_size))
            image = image.astype(np.float32)
            image /= 255.0
            image = image * 2 - 1

            image = np.transpose(image, (2, 0, 1))

            resized_images.append(image)

        resized_images = np.stack(resized_images, axis=0)

        sample['images'] = resized_images
        return sample


class ImageNormalizeToTensor(object):
    """
    Rescale the image in a sample to a given size.
    """

    def __call__(self, image):
        # image = F.to_tensor(image)
        image = TF.to_tensor(image)
        image.mul_(2.0)
        image.sub_(1.0)
        return image


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        sample['images'] = torch.Tensor(sample['images']).float()
        sample['smpls'] = torch.Tensor(sample['smpls']).float()

        return sample


def morph(src_bg_mask, ks, mode='erode', kernel=None):
    n_ks = ks ** 2
    pad_s = ks // 2

    if kernel is None:
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32, device=src_bg_mask.device)

    if mode == 'erode':
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=1.0)
        out = F.conv2d(src_bg_mask_pad, kernel)
        out = (out == n_ks).float()
    else:
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=0.0)
        out = F.conv2d(src_bg_mask_pad, kernel)
        out = (out >= 1).float()

    return out


def cal_mask_bbox(head_mask, factor=1.3):
    """
    Args:
        head_mask (np.ndarray): (N, 1, 256, 256).
        factor (float): the factor to enlarge the bbox of head.

    Returns:
        bbox (np.ndarray.int32): (N, 4), hear, 4 = (left_top_x, right_top_x, left_top_y, right_top_y)

    """
    bs, _, height, width = head_mask.shape

    bbox = np.zeros((bs, 4), dtype=np.int32)
    valid = np.ones((bs,), dtype=np.float32)

    for i in range(bs):
        mask = head_mask[i, 0]
        ys, xs = np.where(mask == 1)

        if len(ys) == 0:
            valid[i] = 0.0
            bbox[i, 0] = 0
            bbox[i, 1] = width
            bbox[i, 2] = 0
            bbox[i, 3] = height
            continue

        lt_y = np.min(ys)   # left top of Y
        lt_x = np.min(xs)   # left top of X

        rt_y = np.max(ys)   # right top of Y
        rt_x = np.max(xs)   # right top of X

        h = rt_y - lt_y     # height of head
        w = rt_x - lt_x     # width of head

        cy = (lt_y + rt_y) // 2    # (center of y)
        cx = (lt_x + rt_x) // 2    # (center of x)

        _h = h * factor
        _w = w * factor

        _lt_y = max(0, int(cy - _h / 2))
        _lt_x = max(0, int(cx - _w / 2))

        _rt_y = min(height, int(cy + _h / 2))
        _rt_x = min(width, int(cx + _w / 2))

        if (_lt_x == _rt_x) or (_lt_y == _rt_y):
            valid[i] = 0.0
            bbox[i, 0] = 0
            bbox[i, 1] = width
            bbox[i, 2] = 0
            bbox[i, 3] = height
        else:
            bbox[i, 0] = _lt_x
            bbox[i, 1] = _rt_x
            bbox[i, 2] = _lt_y
            bbox[i, 3] = _rt_y

    return bbox, valid


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.FloatTensor(tensor)
    return tensor


def plot_fim_enc(fim_enc, map_name):
    # import matplotlib.pyplot as plt
    import utils.mesh as mesh
    if not isinstance(fim_enc, np.ndarray):
        fim_enc = fim_enc.cpu().numpy()

    if fim_enc.ndim != 4:
        fim_enc = fim_enc[np.newaxis, ...]

    fim_enc = np.transpose(fim_enc, axes=(0, 2, 3, 1))

    imgs = []
    for fim_i in fim_enc:
        img = mesh.cvt_fim_enc(fim_i, map_name)
        imgs.append(img)

    return np.stack(imgs, axis=0)


def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        img += 1.0
        img /= 2.0

    image_numpy = img.numpy()
    # image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy *= 255.0

    return image_numpy.astype(imtype)


def tensor2maskim(mask, imtype=np.uint8, idx=0, nrows=1):
    im = tensor2im(mask, imtype=imtype, idx=idx, unnormalize=False, nrows=nrows)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=-1)
    return im


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

    return paths


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def clear_dir(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)

    return mkdir(path)


def save_image(image_numpy, image_path):
    mkdir(os.path.dirname(image_path))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data


def write_pickle_file(pkl_path, data_dict):
    with open(pkl_path, 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=2)

