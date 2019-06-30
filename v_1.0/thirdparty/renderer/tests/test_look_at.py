import unittest

import torch
import numpy as np

import neural_renderer as nr

class TestLookAt(unittest.TestCase):
    def test_case1(self):
        eyes = [
            [1, 0, 1],
            [0, 0, -10],
            [-1, 1, 0],
        ]
        answers = [
            [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
            [1, 0, 10],
            [0, np.sqrt(2) / 2, 3. / 2. * np.sqrt(2)],
        ]
        vertices = torch.from_numpy(np.array([1, 0, 0], np.float32))
        vertices = vertices[None, None, :]
        for e, a in zip(eyes, answers):
            eye = np.array(e, np.float32)
            transformed = nr.look_at(vertices, eye)
            assert(np.allclose(transformed.data.squeeze().numpy(), np.array(a)))

if __name__ == '__main__':
    unittest.main()
