from __future__ import print_function
from PIL import Image
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision
import math
import pickle


def morph(src_bg_mask, ks, mode='erode', kernel=None):
    n_ks = ks ** 2
    pad_s = ks // 2

    if kernel is None:
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32, device=src_bg_mask.device)

    if mode == 'erode':
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=1.0)
        # print(src_bg_mask.shape, src_bg_mask_pad.shape)
        out = F.conv2d(src_bg_mask_pad, kernel)
        # print(out.shape)
        out = (out == n_ks).float()
    else:
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=0.0)
        # print(src_bg_mask.shape, src_bg_mask_pad.shape)
        out = F.conv2d(src_bg_mask_pad, kernel)
        # print(out.shape)
        out = (out >= 1).float()

    return out


def cal_head_bbox(head_mask, factor=1.3):
    """
    Args:
        head_mask (np.ndarray): (N, 1, 256, 256).
        factor (float): the factor to enlarge the bbox of head.

    Returns:
        bbox (np.ndarray.int32): (N, 4), hear, 4 = (left_top_x, left_top_y, right_top_x, right_top_y)

    """
    bs, _, height, width = head_mask.shape

    bbox = np.zeros((bs, 4), dtype=np.int32)
    valid = np.ones((bs,), dtype=np.float32)

    for i in range(bs):
        mask = head_mask[i, 0]
        ys, xs = np.where(mask == 1)

        if len(ys) == 0:
            valid[i] = 0.0
            bbox[i, 2] = width
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
            bbox[i, 2] = width
            bbox[i, 3] = height
        else:
            bbox[i, 0] = _lt_x
            bbox[i, 1] = _lt_y
            bbox[i, 2] = _rt_x
            bbox[i, 3] = _rt_y

    return bbox, valid


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor).float()
    return tensor


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


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(image_numpy, image_path):
    mkdir(os.path.dirname(image_path))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_str_data(data, path):
    mkdir(os.path.dirname(path))
    np.savetxt(path, data, delimiter=",", fmt="%s")


def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data


def write_pickle_file(pkl_path, data_dict):
    with open(pkl_path, 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=2)

