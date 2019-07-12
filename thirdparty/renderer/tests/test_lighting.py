import unittest

import torch

import neural_renderer as nr

class TestLighting(unittest.TestCase):
    
    def test_case1(self):
        """Test whether it is executable."""
        faces = torch.randn(64, 16, 3, 3, dtype=torch.float32)
        textures = torch.randn(64, 16, 8, 8, 8, 3, dtype=torch.float32)
        nr.lighting(faces, textures)

if __name__ == '__main__':
    unittest.main()



