import unittest
import os

import torch
import numpy as np
from skimage.io import imread

import neural_renderer as nr
import utils

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


class TestRasterizeSilhouettes(unittest.TestCase):
    def test_case1(self):
        """Whether a silhouette by neural renderer matches that by Blender."""

        # load teapot
        vertices, faces, _ = utils.load_teapot_batch()

        # create renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.image_size = 256
        renderer.anti_aliasing = False

        images = renderer(vertices, faces, mode='silhouettes')
        images = images.detach().cpu().numpy()
        image = images[2]

        # load reference image by blender
        ref = imread(os.path.join(data_dir, 'teapot_blender.png'))
        ref = (ref.min(-1) != 255).astype(np.float32)

        assert(np.allclose(ref, image))

    def test_backward_case1(self):
        """Backward if non-zero gradient is out of a face."""

        vertices = [
            [0.8, 0.8, 1.],
            [0.0, -0.5, 1.],
            [0.2, -0.4, 1.]]
        faces = [[0, 1, 2]]
        pxi = 35
        pyi = 25
        grad_ref = [
            [1.6725862, -0.26021874, 0.],
            [1.41986704, -1.64284933, 0.],
            [0., 0., 0.],
        ]

        renderer = nr.Renderer(camera_mode='look_at')
        renderer.image_size = 64
        renderer.anti_aliasing = False
        renderer.perspective = False

        vertices = torch.from_numpy(np.array(vertices, np.float32)).cuda()
        faces = torch.from_numpy(np.array(faces, np.int32)).cuda()
        grad_ref = torch.from_numpy(np.array(grad_ref, np.float32)).cuda()
        vertices, faces, grad_ref = utils.to_minibatch((vertices, faces, grad_ref))
        vertices.requires_grad = True
        images = renderer(vertices, faces, mode='silhouettes')
        loss = torch.sum(torch.abs(images[:, pyi, pxi] - 1))
        loss.backward()

        assert(np.allclose(vertices.grad, grad_ref, rtol=1e-2))

    def test_backward_case2(self):
        """Backward if non-zero gradient is on a face."""

        vertices = [
            [0.8, 0.8, 1.],
            [-0.5, -0.8, 1.],
            [0.8, -0.8, 1.]]
        faces = [[0, 1, 2]]
        pyi = 40
        pxi = 50
        grad_ref = [
            [0.98646867, 1.04628897, 0.],
            [-1.03415668, - 0.10403691, 0.],
            [3.00094461, - 1.55173182, 0.],
        ]

        renderer = nr.Renderer(camera_mode='look_at')
        renderer.image_size = 64
        renderer.anti_aliasing = False
        renderer.perspective = False

        vertices = torch.from_numpy(np.array(vertices, np.float32)).cuda()
        faces = torch.from_numpy(np.array(faces, np.int32)).cuda()
        grad_ref = torch.from_numpy(np.array(grad_ref, np.float32)).cuda()
        vertices, faces, grad_ref = utils.to_minibatch((vertices, faces, grad_ref))
        vertices.requires_grad = True
        images = renderer(vertices, faces, mode='silhouettes')
        loss = torch.sum(torch.abs(images[:, pyi, pxi]))
        loss.backward()

        assert(np.allclose(vertices.grad, grad_ref, rtol=1e-2))


if __name__ == '__main__':
    unittest.main()
