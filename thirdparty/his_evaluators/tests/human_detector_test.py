import cv2
import os
import numpy as np
import torch

from his_evaluators.metrics.yolov3 import YoLov3HumanDetector, pad_to_square, resize


if __name__ == "__main__":

    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    sample_dir = "./data"
    img_names = os.listdir(sample_dir)

    detector = YoLov3HumanDetector(
        weights_path="../data/yolov3-spp.weights",
        device=device
    )

    original_imgs = []
    input_imgs = []
    input_shapes = []

    img_names = [
        "pred_00000000.jpg",
        "pred_00000114.jpg",
        "pred_00000175.jpg",
        "pred_00000423.jpg",
    ]
    for name in img_names:
        orig_img = cv2.imread(os.path.join(sample_dir, name))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        original_imgs.append(orig_img)
        input_shapes.append(orig_img.shape[0:2])

        img = orig_img.astype(np.float32)
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img).float()
        img, _ = pad_to_square(img, 0)
        img = resize(img, detector.img_size)
        input_imgs.append(img)

    input_imgs = torch.stack(input_imgs).to(device)

    boxes = detector.forward(input_imgs, input_shapes)

    for name, box in zip(img_names, boxes):
        if box is None:
            continue

        orig_img = cv2.imread(os.path.join(sample_dir, name))
        x1, y1, x2, y2 = box

        print(orig_img.shape)
        crop = orig_img[y1:y2, x1:x2, :]

        print(crop.shape)
        crop_path = "./data/crop_{}".format(name)
        print(crop_path)
        cv2.imwrite(crop_path, crop)

