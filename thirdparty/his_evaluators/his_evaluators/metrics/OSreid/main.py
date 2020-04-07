import numpy as np
import os

from .encoder import OsNetEncoder


dirpath = os.path.abspath(os.path.dirname(__file__))

# Declare an encoder object
encoder = OsNetEncoder(
    # input_width=704,
    # input_height=480,
    # weight_filepath="weights/model_weights.pth.tar-40",
    weight_filepath=os.path.join(dirpath, "osnet_ibn_x1_0_imagenet.pth"),
    input_width=512,
    input_height=512,
    batch_size=32,
    num_classes=2022,
    patch_height=256,
    patch_width=128,
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    GPU=True)


if __name__ == '__main__':
    pred_imgs = np.random.rand(5, 512, 512, 3)
    pred_imgs = pred_imgs * 255
    pred_imgs = pred_imgs.astype(np.uint8)
    features = encoder.get_features(pred_imgs)



