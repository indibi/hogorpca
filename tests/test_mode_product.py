import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np
import torch
from src.multilinear_ops.mode_n_product import mode_n_product, multi_mode_product

class TestModeProduct(unittest.TestCase):

    def test_mode_product_shape(self):
        """Test if the mode_product function returns the correct shape."""
        tensor = np.random.rand(4, 5, 6)  # Example tensor
        matrix = np.random.rand(3, 5)     # Example matrix for mode 2
        result = mode_n_product(tensor, matrix, mode=2)
        expected_shape = (4, 3, 6)  # Expected shape after mode-2 product
        self.assertEqual(result.shape, expected_shape)

    def test_mode_product_values(self):
        """Test if the mode_product function produces expected values."""
        tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2x2x2 tensor
        matrix = np.array([[1, 0], [0, 1]])  # Identity matrix for mode 2
        result = mode_n_product(tensor, matrix, mode=2)
        expected_result = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Should be the same as tensor
        np.testing.assert_array_equal(result, expected_result)

    def test_mode_product_invalid_mode(self):
        """Test if the mode_product function raises an error for invalid mode."""
        tensor = np.random.rand(4, 5, 6)
        matrix = np.random.rand(5, 3)
        with self.assertRaises(ValueError):
            mode_n_product(tensor, matrix, mode=4)  # Mode out of range

    def test_mode_product_non_matching_dimensions(self):
        """Test if the mode_product function raises an error for non-matching dimensions."""
        tensor = np.random.rand(4, 5, 6)
        matrix = np.random.rand(4, 3)  # Incorrect dimension for mode 2
        with self.assertRaises(ValueError):
            mode_n_product(tensor, matrix, mode=2)
    
    def test_multimode_product_matches_dimensions(self):
        """Test if the multi_mode_product function returns correct shape after multiple mode products."""
        tensor = np.random.rand(4, 5, 6)
        matrices = [np.random.rand(3, 4), np.random.rand(2, 6)]
        modes = [1, 3]
        result = multi_mode_product(tensor, matrices, modes)
        expected_shape = (3, 5, 2)
        self.assertEqual(result.shape, expected_shape)
    
    def test_multimode_product_values(self):
        """Test if the multi_mode_product function produces expected values."""
        tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        matrices = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]
        modes = [1, 3]
        result = multi_mode_product(tensor, matrices, modes)
        expected_result = tensor
        np.testing.assert_array_equal(result, expected_result)

    def test_if_torch_tensor_works(self):
        """Test if the mode_n_product function works with torch tensors."""
        tensor = torch.rand(4, 5, 6)
        matrix = torch.rand(3, 5)
        result = mode_n_product(tensor, matrix, mode=2)
        expected_shape = (4, 3, 6)
        self.assertEqual(result.shape, expected_shape)
    
    def test_mode_product_values_mean_calculation(self):
        """Test if the mode_product function correctly computes mean along a mode."""
        tensor = torch.rand(4, 5, 6)
        matrix = torch.ones(1, 5) / 5
        result = mode_n_product(tensor, matrix, mode=2)
        expected_result = tensor.mean(dim=1, keepdim=True)
        torch.testing.assert_close(result, expected_result)
    
if __name__ == '__main__':
    unittest.main()