import numpy as np
import cv2
import os
import requests
from PIL import Image
from io import BytesIO


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '8'

from maskrcnn_benchmark.config import cfg
from temp.predictor import COCODemo

from utils.visualizer.demo_visualizer import MotionImitationVisualizer


def load_1(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    # image = np.transpose(image, (2, 0, 1))

    return image


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


config_file = "./thirdparty/detection/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=256,
    confidence_threshold=0.7,
)
visualizer = MotionImitationVisualizer(env='mask_rcnn', ip='http://10.19.126.34', port=10087)

src_img_path = '/home/piaozx/liuwen/p300/imgs/3.jpg'
image = load_1(src_img_path)
predictions = coco_demo.run_on_opencv_image(image)
print(image.shape, predictions.shape)
visualizer.vis_named_img('src', image[None, ...], normalize=True, transpose=True)
visualizer.vis_named_img('out', predictions[None, ...], normalize=True, transpose=True)


# # from http://cocodataset.org/#explore?id=345434
# image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
# # compute predictions
# print(image.shape)
# predictions = coco_demo.run_on_opencv_image(image)
# print(predictions.shape)
# visualizer.vis_named_img('src', image[None, ...], normalize=True)
# visualizer.vis_named_img('out', predictions[None, ...], normalize=True)
