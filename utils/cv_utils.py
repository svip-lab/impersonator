import cv2
from matplotlib import pyplot as plt
import numpy as np


HMR_IMG_SIZE = 224
IMG_SIZE = 256


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


def resize_img_with_scale(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def kp_to_bbox_param(kp, vis_thresh=0, diag_len=150.0):
    """
    Finds the bounding box parameters from the 2D keypoints.

    Args:
        kp (Kx3): 2D Keypoints.
        vis_thresh (float): Threshold for visibility.
        diag_len(float): diagonal length of bbox of each person

    Returns:
        [center_x, center_y, scale]
    """
    if kp is None:
        return

    if kp.shape[1] == 3:
        vis = kp[:, 2] > vis_thresh
        if not np.any(vis):
            return
        min_pt = np.min(kp[vis, :2], axis=0)
        max_pt = np.max(kp[vis, :2], axis=0)
    else:
        min_pt = np.min(kp, axis=0)
        max_pt = np.max(kp, axis=0)

    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height < 0.5:
        return
    center = (min_pt + max_pt) / 2.
    scale = diag_len / person_height

    return np.append(center, scale)


def cal_process_params(im_path, bbox_param, rescale=None, image=None, image_size=IMG_SIZE, proc=False):
    """
    Args:
        im_path (str): the path of image.
        image (np.ndarray or None): if it is None, then loading the im_path, else use image.
        bbox_param (3,) : [cx, cy, scale].
        rescale (float, np.ndarray or None): rescale factor.
        proc (bool): the flag to return processed image or not.
        image_size (int): the cropped image.

    Returns:
        proc_img (np.ndarray): if proc is True, return the process image, else return the original image.
    """
    if image is None:
        image = read_cv2_img(im_path)

    orig_h, orig_w = image.shape[0:2]
    center = bbox_param[:2]
    scale = bbox_param[2]
    if rescale is not None:
        scale = rescale

    if proc:
        image_scaled, scale_factors = resize_img_with_scale(image, scale)
        resized_h, resized_w = image_scaled.shape[:2]
    else:
        scale_factors = [scale, scale]
        resized_h = orig_h * scale
        resized_w = orig_w * scale

    center_scaled = np.round(center * scale_factors).astype(np.int)

    if proc:
        # Make sure there is enough space to crop image_size x image_size.
        image_padded = np.pad(
            array=image_scaled,
            pad_width=((image_size,), (image_size,), (0,)),
            mode='edge'
        )
        padded_h, padded_w = image_padded.shape[0:2]
    else:
        padded_h = resized_h + image_size * 2
        padded_w = resized_w + image_size * 2

    center_scaled += image_size

    # Crop image_size x image_size around the center.
    margin = image_size // 2
    start_pt = (center_scaled - margin).astype(int)
    end_pt = (center_scaled + margin).astype(int)
    end_pt[0] = min(end_pt[0], padded_w)
    end_pt[1] = min(end_pt[1], padded_h)

    if proc:
        proc_img = image_padded[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
        height, width = image_scaled.shape[:2]
    else:
        height, width = end_pt[1] - start_pt[1], end_pt[0] - start_pt[0]
        proc_img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # proc_img = None

    center_scaled -= start_pt
    im_shape = [height, width]

    return {
        # return original too with info.
        'image': proc_img,
        'im_path': im_path,
        'im_shape': im_shape,
        'orig_im_shape': [orig_h, orig_w],
        'center': center_scaled,
        'scale': scale,
        'start_pt': start_pt,
    }


def cam_denormalize(cam, N):
    # This is camera in crop image coord.
    new_cam = np.hstack([N * cam[0] * 0.5, cam[1:] + (2. / cam[0]) * 0.5])
    return new_cam


def cam_init2orig(cam, scale, start_pt, N=HMR_IMG_SIZE):
    """
    Args:
        cam (3,): (s, tx, ty)
        scale (float): scale = resize_h / orig_h
        start_pt (2,): (lt_x, lt_y)
        N (int): hmr_image_size (224) or IMG_SIZE

    Returns:
        cam_orig (3,): (s, tx, ty), camera in original image coordinates.

    """
    # This is camera in crop image coord.
    cam_crop = np.hstack([N * cam[0] * 0.5, cam[1:] + (2. / cam[0]) * 0.5])

    print('cam_init', cam)
    print('cam_crop', cam_crop)

    # This is camera in orig image coord
    cam_orig = np.hstack([
        cam_crop[0] / scale,
        cam_crop[1:] + (start_pt - N) / cam_crop[0]
    ])
    print('cam_orig', cam_orig)
    return cam_orig


def cam_orig2crop(cam, scale, start_pt, N=IMG_SIZE, normalize=True):
    """
    Args:
        cam (3,): (s, tx, ty), camera in orginal image coordinates.
        scale (float): scale = resize_h / orig_h or (resize_w / orig_w)
        start_pt (2,): (lt_x, lt_y)
        N (int): hmr_image_size (224) or IMG_SIZE
        normalize (bool)

    Returns:

    """
    cam_recrop = np.hstack([
        cam[0] * scale,
        cam[1:] + (N - start_pt) / (scale * cam[0])
    ])
    if normalize:
        cam_norm = np.hstack([
            cam_recrop[0] * (2. / N),
            cam_recrop[1:] - N / (2 * cam_recrop[0])
        ])
    else:
        cam_norm = cam_recrop
    return cam_norm


def cam_process(cam_init, scale_150, start_pt_150, scale_proc, start_pt_proc, image_size):
    """
    Args:
        cam_init:
        scale_150:
        start_pt_150:
        scale_proc:
        start_pt_proc:
        image_size

    Returns:

    """
    cam_orig = cam_init2orig(cam_init, scale=scale_150, start_pt=start_pt_150, N=HMR_IMG_SIZE)
    cam_crop = cam_orig2crop(cam_orig, scale=scale_proc, start_pt=start_pt_proc, N=image_size, normalize=True)

    return cam_crop


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

