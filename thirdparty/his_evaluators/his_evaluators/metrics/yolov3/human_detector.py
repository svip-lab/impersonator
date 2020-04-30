import torch
import torch.nn as nn
import os

from .models import Darknet
from .utils.utils import non_max_suppression, rescale_boxes


class YoLov3HumanDetector(nn.Module):
    def __init__(self, weights_path="weights/yolov3.weights",
                 conf_thres=0.8, nms_thres=0.4, img_size=416, device=torch.device("cpu")):
        super().__init__()

        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.img_size = img_size

        # Set up model
        model_def = os.path.abspath(os.path.dirname(__file__))
        # model_def = os.path.join(model_def, "config", "yolov3.cfg")
        model_def = os.path.join(model_def, "config", "yolov3-spp.cfg")
        model = Darknet(model_def, img_size=img_size).to(device)
        if weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path))

        model.eval()

        self.device = device
        self.model = model.to(device)

    def forward(self, input_imgs, input_shapes, factor=1.05):
        """
        Run YOLOv3 on input_imgs and return the largest bounding boxes of the person in input_imgs.

        Args:
            input_imgs (torch.tensor): (bs, 3, height, width) is in the range of [0, 1],
            input_shapes (list[tuple]): [(height, width), (height, width), ...],
            factor (float): the factor to enlarge the original boxes, e.g [x0, y0, x1, y1] -> [xx0, yy0, xx1, yy1],
                    here (xx1 - xx0) / (x1 - x0) = factor and (yy1 - yy0) / (y1 - y0) = factor.

        Returns:
            boxes_list (list[tuple or None]): (x1, y1, x2, y2) or None
        """

        # Get detections
        with torch.no_grad():
            # img, _ = pad_to_square(input_imgs, 0)
            # Resize
            img_detections = self.model(input_imgs)
            img_detections = non_max_suppression(img_detections, self.conf_thres, self.nms_thres)

        bs = len(img_detections)

        boxes_list = [None for _ in range(bs)]
        # Draw bounding boxes and labels of detections

        for i, (detections, img_shape) in enumerate(zip(img_detections, input_shapes)):
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, self.img_size, img_shape)

                max_area = 0
                boxes = None
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # if is `person`
                    if int(cls_pred) != 0:
                        continue

                    box_w = x2 - x1
                    box_h = y2 - y1
                    area = box_h * box_w

                    if area > max_area:
                        max_area = area
                        boxes = (x1, y1, x2, y2)

                if boxes is not None:
                    boxes_list[i] = self.enlarge_boxes(boxes, img_shape, factor=factor)

        return boxes_list

    @staticmethod
    def enlarge_boxes(boxes, orig_shape, factor=1.0):
        """

        Args:
            boxes (list or tuple): (x0, y0, x1, y1),
            orig_shape (tuple or list): (height, width),
            factor (float): the factor to enlarge the original boxes, e.g [x0, y0, x1, y1] -> [xx0, yy0, xx1, yy1],
                    here (xx1 - xx0) / (x1 - x0) = factor and (yy1 - yy0) / (y1 - y0) = factor.

        Returns:
            new_boxes (list of tuple): (xx0, yy0, xx1, yy1),
                here (xx1 - xx0) / (x1 - x0) = factor and (yy1 - yy0) / (y1 - y0) = factor.
        """

        height, width = orig_shape

        x0, y0, x1, y1 = boxes

        w = x1 - x0
        h = y1 - y0

        cx = (x1 + x0) / 2
        cy = (y1 + y0) / 2

        half_new_w = w * factor / 2
        half_new_h = h * factor / 2

        xx0 = int(max(0, cx - half_new_w))
        yy0 = int(max(0, cy - half_new_h))

        xx1 = int(min(width, cx + half_new_w))
        yy1 = int(min(height, cy + half_new_h))

        new_boxes = (xx0, yy0, xx1, yy1)
        return new_boxes



