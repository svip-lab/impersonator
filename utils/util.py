from __future__ import print_function
from PIL import Image
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision
import math
import pickle

import utils.mesh as mesh


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


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.FloatTensor(tensor)
    return tensor


def plot_fim_enc(fim_enc, map_name):
    # import matplotlib.pyplot as plt

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

