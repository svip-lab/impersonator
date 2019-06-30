import unittest

import torch
import numpy as np

import neural_renderer as nr

class TestPerspective(unittest.TestCase):
    def test_case1(self):
        vertices = torch.from_numpy(np.array([1,2,10], np.float32))
        v_out = np.array([np.sqrt(3) / 10, 2 * np.sqrt(3) / 10, 10], np.float32)
        vertices = vertices[None, None, :]
        transformer = nr.perspective(vertices)
        assert(np.allclose(transformer.data.squeeze().numpy(), v_out))

if __name__ == '__main__':
    unittest.main()
