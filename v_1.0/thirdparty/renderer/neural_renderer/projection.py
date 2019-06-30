from __future__ import division

import torch
import numpy as np


def batch_skew(vec, batch_size=None, device="cpu"):
    """
    vec is N x 3, batch_size is int.

    e.g. r = [rx, ry, rz]
        skew(r) = [[ 0,    -rz,      ry],
                   [ rz,     0,     -rx],
                   [-ry,    rx,       0]]

    returns N x 3 x 3. Skew_sym version of each matrix.
    """

    if batch_size is None:
        batch_size = vec.shape[0]

    col_inds = np.array([1, 2, 3, 5, 6, 7], dtype=np.int32)

    indices = torch.from_numpy(np.reshape(
        np.reshape(np.arange(0, batch_size) * 9, [-1, 1]) + col_inds,
        newshape=(-1,))).to(device)

    updates = torch.stack(
        [
            -vec[:, 2], vec[:, 1], vec[:, 2],
            -vec[:, 0], -vec[:, 1], vec[:, 0]
        ],
        dim=1
    ).view(-1).to(device)

    res = torch.zeros(batch_size * 9, dtype=vec.dtype).to(device)
    res[indices] = updates
    res = res.view(batch_size, 3, 3)

    return res


def batch_rodrigues(theta, device="cpu"):
    """
    Theta is N x 3

    rodrigues (from cv2.rodrigues):
    source: https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    input: r (3 x 1)
    output: R (3 x 3)

        angle = norm(r)
        r = r / angle

        skew(r) = [[ 0,    -rz,      ry],
                   [ rz,     0,     -rx],
                   [-ry,    rx,       0]]

        R = cos(theta * eye(3) + (1 - cos(theta)) * r * r.T + sin(theta) *  skew(r)
    """
    batch_size = theta.shape[0]

    # angle (batch_size, 1), r (batch_size, 3)
    angle = torch.norm(theta + 1e-8, p=2, dim=1, keepdim=True)
    r = torch.div(theta, angle)

    # angle (batch_size, 1, 1), r (batch_size, 3, 1)
    angle = angle.unsqueeze(-1)
    r = r.unsqueeze(-1)

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # outer (batch_size, 3, 3)
    outer = torch.matmul(r, r.permute(0, 2, 1))
    eyes = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    R = cos * eyes + (1 - cos) * outer + sin * batch_skew(r, batch_size=batch_size, device=device)

    return R


def projection_by_params(vertices, camera_f, camera_c, rt, t, dist_coeffs, orig_size, get_image_points=False):
    """
    Calculate projective transformation of vertices given a projection matrix

    Args:
        vertices: [batch_size, N, 3]
        camera_f: focal length of camera, [batch_size, 2]
        camera_c: camera center, [batch_size, 2]
        rt: rotation vector of (each-axis), [batch_size, 3]
        t: translation vector, [batch_size, 3]
        dist_coeffs: vector of distortion coefficients, [batch_size, 5]
        orig_size: original size of image captured by the camera
        get_image_points: the flag to control whether return image points or not, if True, it will return image points.

    Returns:
        projected vertices: [batch_size, N, 3]
        image points (optional): [batch_size, N, 2]
    """
    # R is (batch_size, 3, 3)
    R = batch_rodrigues(rt, device=vertices.device)

    vertices = torch.bmm(vertices, R) + t[:, None, :]

    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + 1e-5)
    y_ = y / (z + 1e-5)

    # Get distortion coefficients from vector
    k1 = dist_coeffs[:, None, 0]
    k2 = dist_coeffs[:, None, 1]
    p1 = dist_coeffs[:, None, 2]
    p2 = dist_coeffs[:, None, 3]
    k3 = dist_coeffs[:, None, 4]

    # we use x_ for x' and x__ for x'' etc.
    r = torch.sqrt(x_ ** 2 + y_ ** 2)
    x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
    y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
    
    u = camera_f[:, 0] * x__ + camera_c[:, 0]
    v = camera_f[:, 1] * y__ + camera_c[:, 1]

    x__ = 2 * (u - orig_size / 2.) / orig_size
    y__ = 2 * (v - orig_size / 2.) / orig_size

    vertices = torch.stack([x__, y__, z], dim=-1)
    if get_image_points:
        points = torch.stack([u, v], dim=-1)
        return vertices, points
    else:
        return vertices


def projection(vertices, P, dist_coeffs, orig_size):
    '''
    Calculate projective transformation of vertices given a projection matrix
    P: 3x4 projection matrix
    dist_coeffs: vector of distortion coefficients
    orig_size: original size of image captured by the camera
    '''
    vertices = torch.cat([vertices, torch.ones_like(vertices[:, :, None, 0])], dim=-1)
    vertices = torch.bmm(vertices, P.transpose(2,1))
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + 1e-5)
    y_ = y / (z + 1e-5)

    # Get distortion coefficients from vector
    k1 = dist_coeffs[:, None, 0]
    k2 = dist_coeffs[:, None, 1]
    p1 = dist_coeffs[:, None, 2]
    p2 = dist_coeffs[:, None, 3]
    k3 = dist_coeffs[:, None, 4]

    # we use x_ for x' and x__ for x'' etc.
    r = torch.sqrt(x_ ** 2 + y_ ** 2)
    x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
    y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
    x__ = 2 * (x__ - orig_size / 2.) / orig_size
    y__ = 2 * (y__ - orig_size / 2.) / orig_size
    vertices = torch.stack([x__,y__,z], dim=-1)
    return vertices


