from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transform_func

from .OSNet import osnet_ibn_x1_0
from .utils import load_pretrained_weights
from .image_handler import normalize, resize, ndarray_to_tensor


class OsNetEncoder(object):

    # Encoder constants
    PRETRAINED_MODEL = False
    LOSS = 'softmax'

    def __init__(self, input_width: int, input_height: int, weight_filepath: str, batch_size: int, num_classes: int, patch_height: int, patch_width: int, norm_mean: List[float], norm_std: List[float], GPU: bool):

        self._input_width = input_width
        self._input_height = input_height
        self._weight_filepath = weight_filepath
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.patch_size = (patch_height, patch_width)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.GPU = GPU
        self._model = osnet_ibn_x1_0(
            num_classes=self.num_classes,
            loss=OsNetEncoder.LOSS,
            pretrained=OsNetEncoder.PRETRAINED_MODEL,
            use_gpu=self.GPU
        )
        self._model.eval()  # Set the torch model for evaluation
        self.weights_loaded = load_pretrained_weights(
            model=self._model,
            weight_path=self._weight_filepath
        )
        if self.GPU:
            self._model = self._model.cuda()

    def load_image(self, patch: np.ndarray) -> torch.Tensor:
        ''' load image involves three processes: resizing, normalising and translating
        the np.ndarray into a torch.Tensor ready for GPU.

        :param patch: single detection patch, in np.ndarray format
        :return: resized and normalised single detection tensor
        '''

        if self.GPU:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        resized_patch = resize(
            img=patch,
            size=self.patch_size
        )
        torch_tensor = ndarray_to_tensor(pic=resized_patch)

        normalized_tensor = normalize(
            tensor=torch_tensor,
            mean=self.norm_mean,
            std=self.norm_std,
            inplace=False
        )

        # Transforms the normalised tensor to a cuda tensor or a cpu tensor wrt which device is available
        gpu_tensor = normalized_tensor.to(device)

        return gpu_tensor

    def __call__(self, images: torch.tensor, bboxes: Union[torch.tensor, np.ndarray, List] = None):
        """

        Args:
            images (torch.tensor): (bs, 3, height, width) is in the range [0, 255] with torch.uint8.
            bboxes (torch.tensor, np.ndarray, list or None): [(4,), (4,), ..., (4,)], (4,) = (x0, y0, x1, y1)

        Returns:
            feats (torch.tensor): (bs, dim)
        """
        images = images.float()
        images /= 255.0

        if bboxes is None:
            crop_imgs = F.interpolate(images, size=self.patch_size, mode="bilinear", align_corners=True)
        else:
            bs = images.shape[0]
            crop_imgs = []
            for i in range(bs):
                x = images[i:i+1]
                x0, y0, x1, y1 = bboxes[i]
                crop = x[:, :, y0:y1, x0:x1]
                crop = transform_func.normalize(crop, mean=self.norm_mean, std=self.norm_std)
                crop = F.interpolate(crop, size=self.patch_size, mode="bilinear", align_corners=True)
                crop_imgs.append(crop)
            crop_imgs = torch.cat(crop_imgs, dim=0)

        feats = self._model(crop_imgs)
        return feats

    def get_features(self, image_patches: List[np.ndarray]) -> List[np.ndarray]:

        ''' Extract the 512 features associated to each detection
        :param image_patches: List[np.ndarray] of detections
        :return features: List[np.ndarray] of features associated to each detection
        '''

        features = list()

        for patch in image_patches:
            if patch is not None:
                # initial_time = time.time()
                patch_gpu_tensor = self.load_image(patch)
                patch_features = self._model(patch_gpu_tensor)
                # Translating to np.ndarray avoids further issues with deepcopying torch.Tensors (in "tracker")
                numpy_features = patch_features.cpu().detach().numpy()
                features.append(numpy_features)

                # numpy_features returns a list of a list of features, so we get the first entry as an action of flattening
                # features.append(compress_feature_vector(numpy_features[0]))
                # print(list(numpy_features[0])) # Debugging console out
                # final_time = time.time()
                # print(f"[INFO] {numpy_features} appended")
                # print(f"[PERFORMANCE] Features extracted: {1/(final_time-initial_time)} Hz")
            else:
                features.append(None)
                # print("[INFO] None appended")

        return features

