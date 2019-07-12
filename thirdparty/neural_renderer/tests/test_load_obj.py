import unittest
import os

import numpy as np
from skimage.io import imsave

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class TestCore(unittest.TestCase):
    def test_tetrahedron(self):
        vertices_ref = np.array(
            [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 0., 0.]],
            'float32')
        faces_ref = np.array(
            [
                [1, 3, 2],
                [3, 1, 0],
                [2, 0, 1],
                [0, 2, 3]],
            'int32')

        obj_file = os.path.join(data_dir, 'tetrahedron.obj')
        vertices, faces = nr.load_obj(obj_file, False)
        assert (np.allclose(vertices_ref, vertices))
        assert (np.allclose(faces_ref, faces))
        vertices, faces = nr.load_obj(obj_file, True)
        assert (np.allclose(vertices_ref * 2 - 1.0, vertices))
        assert (np.allclose(faces_ref, faces))

    def test_teapot(self):
        obj_file = os.path.join(data_dir, 'teapot.obj')
        vertices, faces = nr.load_obj(obj_file)
        assert (faces.shape[0] == 2464)
        assert (vertices.shape[0] == 1292)

    def test_texture(self):
        renderer = nr.Renderer(camera_mode='look_at')

        vertices, faces, textures = nr.load_obj(
            os.path.join(data_dir, '1cde62b063e14777c9152a706245d48/model.obj'), load_texture=True)

        renderer.eye = nr.get_points_from_angles(2, 15, 30)
        images = renderer.render(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :]).permute(0,2,3,1).detach().cpu().numpy()
        imsave(os.path.join(data_dir, 'car.png'), images[0])

        vertices, faces, textures = nr.load_obj(
            os.path.join(data_dir, '4e49873292196f02574b5684eaec43e9/model.obj'), load_texture=True, texture_size=16)
        renderer.eye = nr.get_points_from_angles(2, 15, -90)
        images = renderer.render(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :]).permute(0,2,3,1).detach().cpu().numpy()
        imsave(os.path.join(data_dir, 'display.png'), images[0])


if __name__ == '__main__':
    unittest.main()
