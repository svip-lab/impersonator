from __future__ import division
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
import numpy as np
import skimage.metrics
from scipy import linalg


MODEL_ZOOS = dict()


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 4

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,    # First max pooling features
        192: 1,   # Second max pooling featurs
        768: 2,   # Pre-aux classifier features
        2048: 3,  # Final average pooling features
        1000: 4   # Final classifier score
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 4, \
            'Last possible output block index is 4'

        self.blocks = nn.ModuleList()

        inception = inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        # Block 4: 1000 classifier
        if self.last_needed_block >= 4:
            self.blocks.append(inception.fc)

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)

            if idx == 3:
                x = x[:, :, 0, 0]

            if idx in self.output_blocks:
                if len(x.shape) == 4 and (x.size(2) != 1 or x.size(3) != 1):
                    preds = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze_(-1).squeeze_(-1)
                else:
                    preds = x
                outp.append(preds)

            if idx == self.last_needed_block:
                break

        return outp


class BaseMetric(object):

    INCEPTION_V3 = 'inception_v3'
    PERCEPTUAL = 'perceptual'
    HMR = "hmr"
    FACE = "sphereface"
    FACE_DETECTOR = "MTCNN"
    FACE_FD = "inception_resnet_v1"
    FACE_RECOGNITION = "inception_resnet_v1"
    PERSON_DETECTOR = "YOLOv3"
    OSreID = 'OS-reid'
    PCBreID = 'PCB-reid'

    FACE_RECOGNITION_SIZE = 160

    MODEL_KEYS = [INCEPTION_V3, PERCEPTUAL, HMR, FACE, PERSON_DETECTOR,
                  OSreID, PCBreID, FACE_DETECTOR, FACE_RECOGNITION]

    LOWER = 'lower score is better.'
    HIGHER = 'higher score is better'

    def __init__(self, device=torch.device("cpu"), size=(512, 512)):
        global MODEL_ZOOS

        super(BaseMetric, self).__init__()

        self.device = device
        self.size = size
        self.model_zoos = MODEL_ZOOS

    def forward(self, *input):
        raise NotImplementedError

    def calculate_score(self, *input):
        raise NotImplementedError

    def register_model(self, key):
        assert key in self.MODEL_KEYS, '{} must in {}'.format(key, self.MODEL_KEYS)

        if key not in self.model_zoos.keys():
            if key == self.INCEPTION_V3:
                self.model_zoos[key] = InceptionV3(output_blocks=[3], resize_input=False,
                                                   normalize_input=False, requires_grad=False)
                self.model_zoos[key] = self.model_zoos[key].to(self.device)
                self.model_zoos[key].eval()

            elif key == self.PERCEPTUAL:
                from .lpips import PerceptualLoss

                use_gpu = self.device != "cpu"
                model = PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu)

                self.model_zoos[key] = model

            elif key == self.PERSON_DETECTOR:
                from .yolov3 import YoLov3HumanDetector

                data_dir = self.resource_dir

                detector = YoLov3HumanDetector(
                    weights_path=os.path.join(data_dir, "yolov3-spp.weights"),
                    device=self.device
                )
                self.model_zoos[key] = detector

            elif key == self.OSreID:
                from .OSreid import OsNetEncoder

                data_dir = self.resource_dir

                model = OsNetEncoder(
                    # input_width=704,
                    # input_height=480,
                    # weight_filepath="weights/model_weights.pth.tar-40",
                    weight_filepath=os.path.join(data_dir, "osnet_ibn_x1_0_imagenet.pth"),
                    input_height=self.size[0],
                    input_width=self.size[1],
                    batch_size=32,
                    num_classes=2022,
                    patch_height=256,
                    patch_width=128,
                    norm_mean=[0.485, 0.456, 0.406],
                    norm_std=[0.229, 0.224, 0.225],
                    GPU=True)
                self.model_zoos[key] = model

            elif key == self.PCBreID:
                from .PCBreid import PCBReIDMetric

                data_dir = self.resource_dir

                model = PCBReIDMetric(name="PCB", pretrain_path=os.path.join(data_dir, "pcb_net_last.pth"))
                model = model.to(self.device)

                self.model_zoos[key] = model

            elif key == self.HMR:
                from .bodynets import HumanModelRecovery

                data_dir = self.resource_dir

                model = HumanModelRecovery(smpl_pkl_path=os.path.join(data_dir, "smpl_model.pkl"))
                model_dict = torch.load(os.path.join(data_dir, "hmr_tf2pt.pth"))
                model.load_state_dict(model_dict)
                model = model.to(self.device)
                model.eval()

                self.model_zoos[key] = model

            elif key == self.FACE_DETECTOR:
                from .facenet_pytorch import MTCNN

                mtcnn = MTCNN(image_size=self.FACE_RECOGNITION_SIZE, device=self.device)
                self.model_zoos[key] = mtcnn

            elif key == self.FACE_RECOGNITION:
                from .facenet_pytorch import InceptionResnetV1

                resnet = InceptionResnetV1(pretrained='vggface2', classify=False, device=self.device).eval()
                self.model_zoos[key] = resnet

            else:
                raise ValueError(key)

    @property
    def resource_dir(self):
        dirpath = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.dirname(os.path.dirname(dirpath))
        data_dir = os.path.join(data_dir, "data")
        return data_dir

    def preprocess(self, x):
        raise NotImplementedError

    @staticmethod
    def to_numpy(x, transpose=False):
        if not isinstance(x, np.ndarray):
            x = x.cpu().numpy()

        if transpose:
            # (3, 256, 256) -> (256, 256, 3)
            x = np.transpose(x, (1, 2, 0))

        return x

    def quality(self):
        raise NotImplementedError

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    @staticmethod
    def fid_score_func(pred_feats, gt_feats):
        """

        Args:
            pred_feats:
            gt_feats:

        Returns:

        """
        if len(pred_feats) == 0 or len(gt_feats) == 0:
            return 0
        else:
            m1, s1 = np.mean(pred_feats, axis=0), np.cov(pred_feats, rowvar=False)
            m2, s2 = np.mean(gt_feats, axis=0), np.cov(gt_feats, rowvar=False)
            return BaseMetric.calculate_frechet_distance(m1, s1, m2, s2)

    @staticmethod
    def is_score_func(feats_softmax):
        """
            inception score function.

        Args:
            feats_softmax (np.ndarray): features after softmax

        Returns:
            score (float)
        """
        kl = feats_softmax * (np.log(feats_softmax) - np.log(np.expand_dims(np.mean(feats_softmax, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        score = np.exp(kl)
        return score

    @staticmethod
    def ssp_abs_err_score_func(src_smpls, ref_smpls):
        """
            The function of Scale-Shape-Pose absolution error score.
        Args:
            src_smpls (np.ndarray): (bs, 85),
            ref_smpls (np.ndarray): (bs, 85).

        Returns:
            error (float): the absolution Scale-Shape-Pose error between the source and the reference smpls .
        """

        src_scale = src_smpls[:, 0]
        ref_scale = ref_smpls[:, 0]

        scale_error = np.mean(np.abs(src_scale - ref_scale))
        shape_error = np.mean(np.sum(np.abs(src_smpls[:, -10:] - ref_smpls[:, -10:]), axis=1))
        pose_error = np.mean(np.sum(np.abs(src_smpls[:, 0:-10] - ref_smpls[:, 0:-10]), axis=1))

        error = scale_error + shape_error + pose_error

        return error

    @staticmethod
    def normalize_feats(feats):
        """

        Args:
            feats (np.ndarray): (bs, dim)

        Returns:
            feats (np.ndarray): (bs, dim)
        """

        norm = np.sqrt(np.sum(feats ** 2, axis=1, keepdims=True))
        feats = feats / (norm + 1e-6)
        return feats

    def cosine_similarity(self, pred_feats, ref_feats, norm_first=True):

        if len(pred_feats) == 0 or len(ref_feats) == 0:
            return 0
        else:
            if norm_first:
                pred_norm = self.normalize_feats(pred_feats)
                ref_norm = self.normalize_feats(ref_feats)
            else:
                pred_norm = pred_feats
                ref_norm = ref_feats

            return np.mean(np.sum(pred_norm * ref_norm, axis=1))


class SSIMMetric(BaseMetric):

    def __init__(self):
        super(SSIMMetric, self).__init__()
        BaseMetric.__init__(self)

    def preprocess(self, x):
        """
            normalize x from [0, 1] color intensity with np.float32 to [-1, 1] with np.float32,
        Args:
            x (np.ndarray or torch.tensor): [0, 1] color intensity, np.float32 (torch.tensor),
            shape = (3, image_size, image_size)

        Returns:
            out (np.ndarray): [-1, 1] color intensity, np.float32, (image_size, image_size, 3)
        """
        if isinstance(x, np.ndarray):
            out = x.astype(np.float32, copy=True)
        else:
            out = x.numpy().astype(np.float32, copy=True)

        out *= 2
        out -= 1

        out = np.transpose(out, (1, 2, 0))

        return out

    def forward(self, pred, ref):
        """

        Args:
            pred (np.ndarray): color intensity is [-1, 1]
            ref (np.ndarray): color intensity is [-1, 1]

        Returns:

        """
        # print(pred.shape, img.shape)
        # score = measure.compare_ssim(pred, ref, multichannel=True)
        pred = self.preprocess(pred)
        ref = self.preprocess(ref)
        score = skimage.metrics.structural_similarity(pred, ref, multichannel=True)
        return score

    def calculate_score(self, preds, gts):
        scores = []
        length = len(preds)

        assert length == len(gts)

        for i, (pred, ref) in enumerate(zip(preds, gts)):
            scores.append(self.forward(pred, ref))

        return np.mean(scores)

    def quality(self):
        return self.HIGHER


class PSNRMetric(BaseMetric):
    def __init__(self):
        super(PSNRMetric, self).__init__()
        BaseMetric.__init__(self)

    def preprocess(self, x):
        """
            normalize x from [0, 1] color intensity with np.uint8 to [-1, 1] with np.float32,
        Args:
            x (np.ndarray or torch.tensor): [0, 1] color intensity, np.float32 (torch.tensor),
            shape = (3, image_size, image_size)

        Returns:
            out (np.ndarray): [-1, 1] color intensity, np.float32, (image_size, image_size, 3)
        """
        if isinstance(x, np.ndarray):
            out = x.astype(np.float32, copy=True)
        else:
            out = x.numpy().astype(np.float32, copy=True)

        out *= 2
        out -= 1

        out = np.transpose(out, (1, 2, 0))

        return out

    def forward(self, pred, ref):
        """

        Args:
            pred (np.ndarray): color intensity is [-1, 1]
            ref (np.ndarray): color intensity is [-1, 1]

        Returns:
            score (np.float32): the ssim score, higher is better.
        """

        # score = measure.compare_psnr(pred, ref)
        pred = self.preprocess(pred)
        ref = self.preprocess(ref)
        score = skimage.metrics.peak_signal_noise_ratio(image_true=ref, image_test=pred)
        return score

    def calculate_score(self, preds, gts):
        scores = []
        length = len(preds)

        assert length == len(gts)

        for i, (pred, ref) in enumerate(zip(preds, gts)):
            scores.append(self.forward(pred, ref))

        return np.mean(scores)

    def quality(self):
        return self.HIGHER


class PerceptualMetric(BaseMetric):
    def __init__(self, device):

        BaseMetric.__init__(self, device=device)
        self.register_model(self.PERCEPTUAL)

    def preprocess(self, x):
        """

        Args:
            x (np.ndarray (or torch.tensor)): np.ndarray (or torch.tensor),
            each element is [bs, 3, image_size, image_size] wth np.float32 (torch.float32) and color intensity [0, 1];

        Returns:
            out (torch.tensor): (bs, 3, image_size, image_size), color intensity [-1, 1]
        """
        if isinstance(x, np.ndarray):
            out = torch.tensor(x).float()
        else:
            out = x.clone().float()

        out *= 2
        out -= 1
        # print(out.shape, out.max(), out.min(), out.device)
        # print(self.device)
        out = out.to(self.device)
        return out

    def forward(self, pred, ref):
        """

        Args:
            pred (torch.tensor): color intensity is [-1, 1]
            ref (torch.tensor): color intensity is [-1, 1]

        Returns:
            score (torch.tensor):
        """

        with torch.no_grad():
            pred = self.preprocess(pred)
            ref = self.preprocess(ref)
            # return 1.0 - torch.mean(self.model_zoos[self.PERCEPTUAL].forward(pred, target))
            return torch.mean(self.model_zoos[self.PERCEPTUAL].forward(pred, ref))

    def calculate_score(self, preds, gts, batch_size=32):
        assert len(preds) == len(gts)

        length = len(preds)

        scores = []

        for i in range(int(math.ceil((length / batch_size)))):
            pred_batch = preds[i * batch_size: (i + 1) * batch_size]
            gt_batch = gts[i * batch_size: (i + 1) * batch_size]

            s = self.forward(pred_batch, gt_batch)
            scores.append(s)
        scores = torch.stack(scores)
        return torch.mean(scores).cpu().numpy()

    def quality(self):
        return self.LOWER


class InceptionScoreMetric(BaseMetric):

    def __init__(self, device=torch.device("cpu")):
        BaseMetric.__init__(self, device)

        self.register_model(self.INCEPTION_V3)

        self.height, self.width = 299, 299
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).float().to(self.device).view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229, 0.224, 0.225]).float().to(self.device).view(1, 3, 1, 1)

    def preprocess(self, x):
        """

        Args:
            x (np.ndarray or torch.tensor): np.ndarray or torch.tensor, each element is [3, image_size, image_size] wth np.uint8 and color intensity [0, 255];

        Returns:
            out (torch.tensor): (bs, 3, 299, 299), color intensity [0, 1] and normalized using
            mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        """

        if isinstance(x, np.ndarray):
            out = torch.tensor(x)
        else:
            out = x.clone()

        out *= 2
        out -= 1
        out = out.to(self.device)

        with torch.no_grad():
            out = F.interpolate(out, size=(self.height, self.width), mode='bilinear', align_corners=False)
            # out = (out - self.mean) / self.std

        return out

    def forward(self, imgs):
        """

        Args:
            imgs (torch.tensor): (bs, 3, image_size, image_size), color intensity [0, 1] and normalized using
            mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

        Returns:
            probs (torch.tensor): (bs, number of feature outputs)
        """

        imgs = self.preprocess(imgs)
        with torch.no_grad():
            feats = self.model_zoos[self.INCEPTION_V3](imgs)[-1]
            probs = F.softmax(feats, dim=1)

        probs = probs.cpu().numpy()
        return probs

    def calculate_score(self, preds, batch_size=32):
        scores = []
        length = len(preds)
        for i in range(int(math.ceil((length / batch_size)))):
            pred_batch = preds[i * batch_size: (i + 1) * batch_size]

            part = self.forward(pred_batch)
            scores.append(self.is_score_func(part))
        return np.mean(scores), np.std(scores)

    def quality(self):
        return self.HIGHER


class FIDMetric(BaseMetric):

    def __init__(self, device=torch.device("cpu")):
        BaseMetric.__init__(self, device)

        self.register_model(self.INCEPTION_V3)

        self.height, self.width = 299, 299
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).float().to(self.device).view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229, 0.224, 0.225]).float().to(self.device).view(1, 3, 1, 1)

    def preprocess(self, x):
        """

        Args:
            x (np.ndarray or torch.tensor): np.ndarray or torch.tensor, each element is
                            [3, image_size, image_size] wth np.float32 and color intensity [0, 1];

        Returns:
            out (torch.tensor): (bs, 3, 299, 299), color intensity [0, 1] and normalized using
            mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        """

        if isinstance(x, np.ndarray):
            out = torch.tensor(x)
        else:
            out = x.clone()

        out *= 2
        out -= 1
        out = out.to(self.device)

        with torch.no_grad():
            out = F.interpolate(out, size=(self.height, self.width), mode='bilinear', align_corners=False)
            # out = (out - self.mean) / self.std

        return out

    def forward(self, imgs):
        """

        Args:
            imgs (torch.tensor): (bs, 3, image_size, image_size), color intensity [0, 1] and normalized using
            mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

        Returns:
            probs (torch.tensor): (bs, number of feature outputs)
        """

        imgs = self.preprocess(imgs)
        with torch.no_grad():
            feats = self.model_zoos[self.INCEPTION_V3](imgs)[-1]
            # print(feats.shape)
            feats = feats.cpu().numpy()
        return feats

    def calculate_score(self, preds, gts, batch_size=32):
        pred_feats = []
        gt_feats = []

        length = len(preds)
        for i in range(int(math.ceil((length / batch_size)))):
            pred_batch = preds[i * batch_size: (i + 1) * batch_size]
            gt_batch = gts[i * batch_size: (i + 1) * batch_size]

            pred_f = self.forward(pred_batch)
            gt_f = self.forward(gt_batch)

            pred_feats.append(pred_f)
            gt_feats.append(gt_f)

        pred_feats = np.concatenate(pred_feats, axis=0)
        gt_feats = np.concatenate(gt_feats, axis=0)

        return self.fid_score_func(pred_feats, gt_feats)

    def quality(self):
        return self.LOWER


class FreIDMetric(BaseMetric):
    def __init__(self, device=torch.device("cpu"), reid_name="PCB-reid", has_detector=True):

        BaseMetric.__init__(self, device)

        if reid_name == "PCB-reid":
            self.REID = self.PCBreID
        else:
            self.REID = self.OSreID
        self.register_model(self.REID)

        self.has_detector = has_detector
        if has_detector:
            self.register_model(self.PERSON_DETECTOR)

    def preprocess(self, x):
        """

        Args:
            x (torch.tensor): (bs, 3, height, width) is in the range of [0, 1] with torch.float32.

        Returns:
            x (torch.tensor): (bs, 3, height, width) is in the range of [0, 1] with torch.float32.
        """
        x = x.clone()
        x = x.to(self.device)
        return x

    def forward(self, pred):
        """

        Args:
            x (torch.tensor): np.ndarray or torch.tensor, each element is
                [bs, 3, image_size, image_size] wth torch.uint8 and color intensity [0, 255];

        Returns:
            feat (np.ndarray): [bs, C]
        """

        with torch.no_grad():
            pred = self.preprocess(pred)

            if self.has_detector:
                img_shapes = [pred.shape[2:]] * pred.shape[0]
                boxes = self.model_zoos[self.PERSON_DETECTOR](pred, img_shapes, factor=1.05)
            else:
                boxes = None
            feat = self.model_zoos[self.REID](pred, boxes)
            feat = feat.cpu().numpy()
        return feat

    def calculate_score(self, preds, gts, batch_size=32):
        pred_feats = []
        gt_feats = []

        length = len(preds)
        for i in range(int(math.ceil((length / batch_size)))):
            pred_batch = preds[i * batch_size: (i + 1) * batch_size]
            gt_batch = gts[i * batch_size: (i + 1) * batch_size]

            pred_f = self.forward(pred_batch)
            gt_f = self.forward(gt_batch)

            pred_feats.append(pred_f)
            gt_feats.append(gt_f)

        pred_feats = np.concatenate(pred_feats, axis=0)
        gt_feats = np.concatenate(gt_feats, axis=0)

        return self.fid_score_func(pred_feats, gt_feats)

    def quality(self):
        return self.LOWER


class ReIDScore(FreIDMetric):

    def __init__(self, device=torch.device("cpu"), reid_name="PCB-reid", has_detector=True):
        super().__init__(device, reid_name, has_detector)

    def calculate_score(self, preds, gts, batch_size=32):
        assert len(preds) == len(gts)

        length = len(preds)

        scores = []

        for i in range(int(math.ceil((length / batch_size)))):
            pred_batch = preds[i * batch_size: (i + 1) * batch_size]
            gt_batch = gts[i * batch_size: (i + 1) * batch_size]

            pred_f = self.forward(pred_batch)
            gt_f = self.forward(gt_batch)

            s = self.cosine_similarity(pred_f, gt_f)
            scores.append(s)

        return np.mean(scores)

    def quality(self):
        return self.HIGHER


class FaceSimilarityScore(BaseMetric):

    def __init__(self, device=torch.device("cpu"), has_detector=True):
        """

        Args:
            device (torch.device):
            has_detector (bool): use detector or not, default is True.
        """

        super().__init__(device=device)
        self.has_detector = has_detector

        if has_detector:
            self.register_model(self.FACE_DETECTOR)

        self.register_model(self.FACE_RECOGNITION)

    def preprocess(self, x):
        """

        Args:
            x (np.ndarray or torch.tensor): each element is [bs, 3, image_size, image_size]
                        with float32 and color intensity [0, 1];

        Returns:
            out (np.ndarray or torch.tensor): (bs, image_size, image_size, 3),
                 if self.has_detector is True, then it is in the range of [0, 255] with np.uint8;
                 otherwise, it is the torch.tensor and its color is in the range of [-1, 1] with torch.float32.
        """

        x = x.clone()
        if self.has_detector:
            if not isinstance(x, np.ndarray):
                x = x.numpy()
                x *= 255
                x = x.astype(np.uint8)
            x = np.transpose(x, (0, 2, 3, 1))
        else:
            x *= 2
            x -= 1
            x = x.to(self.device)
            x = F.interpolate(x, size=(self.FACE_RECOGNITION_SIZE, self.FACE_RECOGNITION_SIZE), mode="area")
        return x

    def detect_face(self, img):
        """

        Args:
            img (torch.tensor): (bs, 3, height, width) is in the range of [0, 1] with torch.float32.

        Returns:
            face_cropped (torch.tensor): (bs, 3, face_size, face_size) is in the range of [-1, 1] with torch.float32.
        """

        # Get cropped and prewhitened image tensor
        # img_cropped = mtcnn(img, save_path=<optional save path>)
        proc_img = img.clone()
        proc_img = proc_img.numpy()
        proc_img *= 255
        proc_img = proc_img.astype(np.uint8)
        proc_img = np.transpose(proc_img, (0, 2, 3, 1))

        img_cropped = self.model_zoos[self.FACE_DETECTOR](proc_img)

        face_cropped = []
        valid_ids = []
        for i, cropped in enumerate(img_cropped):
            if cropped is None:
                cropped = F.interpolate(
                    img[i:i + 1] * 2 - 1,
                    size=(self.FACE_RECOGNITION_SIZE, self.FACE_RECOGNITION_SIZE),
                    mode="area"
                )[0]
            else:
                valid_ids.append(i)

            # print(cropped.shape, cropped.max(), cropped.min(), img[i:i+1].max(), img[i:i+1].min())
            face_cropped.append(cropped)

        face_cropped = torch.stack(face_cropped).to(self.device)

        return face_cropped, valid_ids

    def forward(self, img):
        """

        Args:
            img:

        Returns:

        """

        with torch.no_grad():
            if self.has_detector:
                face_cropped, valid_ids = self.detect_face(img)
            else:
                face_cropped = self.preprocess(img)
                valid_ids = list(range(img.shape[0]))

            # print(face_cropped.shape, face_cropped.max(), face_cropped.min())
            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = self.model_zoos[self.FACE_RECOGNITION](face_cropped, normalize=False)

            img_embedding = img_embedding.cpu().numpy()

        return img_embedding, valid_ids

    def calculate_score(self, preds, gts, batch_size=32):
        pred_feats = []
        gt_feats = []

        length = len(preds)
        for i in range(int(math.ceil((length / batch_size)))):
            pred_batch = preds[i * batch_size: (i + 1) * batch_size]
            gt_batch = gts[i * batch_size: (i + 1) * batch_size]

            pred_f, _ = self.forward(pred_batch)
            gt_f, valid_ids = self.forward(gt_batch)

            pred_feats.append(pred_f[valid_ids])
            gt_feats.append(gt_f[valid_ids])

        pred_feats = np.concatenate(pred_feats, axis=0)
        gt_feats = np.concatenate(gt_feats, axis=0)

        return self.cosine_similarity(pred_feats, gt_feats, norm_first=True)

    def quality(self):
        return self.HIGHER


class FaceFrechetDistance(FaceSimilarityScore):
    def __init__(self, device=torch.device("cpu"), has_detector=True):
        super().__init__(device, has_detector)

    def calculate_score(self, preds, gts, batch_size=32):
        pred_feats = []
        gt_feats = []

        length = len(preds)
        for i in range(int(math.ceil((length / batch_size)))):
            pred_batch = preds[i * batch_size: (i + 1) * batch_size]
            gt_batch = gts[i * batch_size: (i + 1) * batch_size]

            pred_f = self.forward(pred_batch)
            gt_f = self.forward(gt_batch)

            pred_feats.append(pred_f)
            gt_feats.append(gt_f)

        pred_feats = np.concatenate(pred_feats, axis=0)
        gt_feats = np.concatenate(gt_feats, axis=0)

        return self.fid_score_func(pred_feats, gt_feats)

    def quality(self):
        return self.LOWER


class ScaleShapePoseError(BaseMetric):
    def __init__(self, device=torch.device("cpu")):
        super().__init__(device)
        self.register_model(self.HMR)

        self.height, self.width = 224, 224

    def quality(self):
        return self.LOWER

    def preprocess(self, x):
        """

        Args:
            x (np.ndarray or torch.tensor): np.ndarray or torch.tensor, each element is [3, image_size, image_size]
                        wth np.float32 and color intensity [0, 1];

        Returns:
            out (torch.tensor): (bs, 3, 224, 224) is in the range of [-1, 1].
        """

        x = x.clone()
        x *= 2
        x -= 1
        x = F.interpolate(x, (self.height, self.width), mode="bilinear", align_corners=False)
        x = x.to(self.device)
        return x

    def forward(self, pred):
        """

        Args:
            x (torch.tensor): np.ndarray or torch.tensor, each element is
                [bs, 3, image_size, image_size] wth torch.uint8 and color intensity [0, 255];

        Returns:
            smpls (np.ndarray): (bs, 85).
        """

        with torch.no_grad():
            pred = self.preprocess(pred)
            smpls = self.model_zoos[self.HMR](pred)
            smpls = smpls.cpu().numpy()
        return smpls

    def calculate_score(self, preds, gts, batch_size=32):
        pred_smpls = []
        gt_smpls = []

        length = len(preds)
        for i in range(int(math.ceil((length / batch_size)))):
            pred_batch = preds[i * batch_size: (i + 1) * batch_size]
            gt_batch = gts[i * batch_size: (i + 1) * batch_size]

            pred_f = self.forward(pred_batch)
            gt_f = self.forward(gt_batch)

            pred_smpls.append(pred_f)
            gt_smpls.append(gt_f)

        pred_smpls = np.concatenate(pred_smpls, axis=0)
        gt_smpls = np.concatenate(gt_smpls, axis=0)

        return self.ssp_abs_err_score_func(pred_smpls, gt_smpls)
