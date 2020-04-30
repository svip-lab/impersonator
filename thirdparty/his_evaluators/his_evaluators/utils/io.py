import json
import cv2
import numpy as np
import pickle
import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data


def load_img(img_path, image_size):
    """
        load image from `img_path` and resize it to (image_size, image_size), convert to RGB color space.
    Args:
        img_path:
        image_size:

    Returns:
        img (np.ndarray): [3, image_size, image_size], np.float32, RGB channel, [0, 1] intensity.
    """
    img = cv2.imread(img_path)

    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32, copy=False)
    img /= 255
    return img
