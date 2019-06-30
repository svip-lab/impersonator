import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import math
import ipdb


def read_cv2_img(path):
    """
    Read color images
    :param path: Path to image
    :return: Only returns color images
    """
    img = cv2.imread(path, -1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_cv2_img(img, path, image_size=None, normalize=False):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # print('normalize = {}'.format(normalize))

    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))

    if normalize:
        img = (img + 1) / 2.0 * 255
        img = img.astype(np.uint8)

    cv2.imwrite(path, img)
    return img


def transform_img(image, image_size, transpose=False):
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32)
    image /= 255.0

    if transpose:
        image = image.transpose((2, 0, 1))

    return image


def show_cv2_img(img, title='img'):
    '''
    Display cv2 image
    :param img: cv::mat
    :param title: title
    :return: None
    '''
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_images_row(imgs, titles, rows=1):
    """
       Display grid of cv2 images image
       :param img: list [cv::mat]
       :param title: titles
       :return: None
    """
    assert ((titles is None) or (len(imgs) == len(titles)))
    num_images = len(imgs)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, num_images + 1)]

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(rows, np.ceil(num_images / float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        ax.set_title(title)
        plt.axis('off')
    plt.show()


def intrinsic_mtx(f, c):
    """
    Obtain intrisic camera matrix.
    Args:
        f: np.array, 1 x 2, the focus lenth of camera, (fx, fy)
        c: np.array, 1 x 2, the center of camera, (px, py)
    Returns:
        - cam_mat: np.array, 3 x 3, the intrisic camera matrix.
    """
    return np.array([[f[1], 0, c[1]],
                     [0, f[0], c[0]],
                     [0, 0, 1]], dtype=np.float32)


def extrinsic_mtx(rt, t):
    """
    Obtain extrinsic matrix of camera.
    Args:
        rt: np.array, 1 x 3, the angle of rotations.
        t: np.array, 1 x 3, the translation of camera center.
    Returns:
        - ext_mat: np.array, 3 x 4, the extrinsic matrix of camera.
    """
    # R is (3, 3)
    R = cv2.Rodrigues(rt)[0]
    t = np.reshape(t, newshape=(3, 1))
    Rc = np.dot(R, t)
    ext_mat = np.hstack((R, -Rc))
    ext_mat = np.vstack((ext_mat, [0, 0, 0, 1]))
    ext_mat = ext_mat.astype(np.float32)
    return ext_mat


def extrinsic(rt, t):
    """
    Obtain extrinsic matrix of camera.
    Args:
        rt: np.array, 1 x 3, the angle of rotations.
        t: np.array, 1 x 3, or (3,) the translation of camera center.
    Returns:
        - R: np.ndarray, 3 x 3
        - t: np.ndarray, 1 x 3
    """
    R = cv2.Rodrigues(rt)[0]
    t = np.reshape(t, newshape=(1, 3))
    return R, t


def euler2matrix(rt):
    """
    Obtain rotation matrix from euler angles
    Args:
        rt: np.array, (3,)
    Returns:
        R: np.array, (3,3)
    """
    Rx = np.array([[1, 0,             0],
                   [0, np.cos(rt[0]), -np.sin(rt[0])],
                   [0, np.sin(rt[0]), np.cos(rt[0])]], dtype=np.float32)

    Ry = np.array([[np.cos(rt[1]),     0,       np.sin(rt[1])],
                   [0,                 1,       0],
                   [-np.sin(rt[1]),    0,       np.cos(rt[1])]], dtype=np.float32)

    Rz = np.array([[np.cos(rt[2]),     -np.sin(rt[2]),       0],
                   [np.sin(rt[2]),      np.cos(rt[2]),       0],
                   [0,                              0,       1]], dtype=np.float32)

    return np.dot(Rz, np.dot(Ry, Rx))


def get_rotated_smpl_pose(pose, theta):
    """
    :param pose: (72,)
    :param theta: rotation angle of y axis
    :return:
    """
    global_pose = pose[:3]
    R, _ = cv2.Rodrigues(global_pose)
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    new_R = np.matmul(R, Ry)
    new_global_pose, _ = cv2.Rodrigues(new_R)
    new_global_pose = new_global_pose.reshape(3)

    rotated_pose = pose.copy()
    rotated_pose[:3] = new_global_pose

    return rotated_pose


class ImageTransformer(object):
    """
    Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        """
        :type output_size: tuple or int
        :param output_size: Desired output size. If tuple, output is matched to output_size.
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
        image = F.to_tensor(image)
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


if __name__ == '__main__':
    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    R = euler2matrix(np.array([0, 90, 0], dtype=np.float32))

    print(isRotationMatrix(R))

