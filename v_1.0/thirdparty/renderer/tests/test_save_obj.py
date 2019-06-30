import unittest
import os

import numpy as np

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class TestCore(unittest.TestCase):
    def test_save_obj(self):
        teapot = os.path.join(data_dir, 'teapot.obj')
        teapot2 = os.path.join(data_dir, 'teapot2.obj')
        vertices, faces = nr.load_obj(teapot)
        nr.save_obj(teapot2, vertices, faces)
        vertices2, faces2 = nr.load_obj(teapot2)
        os.remove(teapot2)
        assert np.allclose(vertices, vertices2)
        assert np.allclose(faces, faces2)

    def test_texture(self):
        pass

if __name__ == '__main__':
    unittest.main()
