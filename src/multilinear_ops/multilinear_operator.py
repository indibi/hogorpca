"""Module for multi-linear operators on tensors

Author: Mert Indibi
Date: 9/9/2025
"""

import numpy as np
import torch

from src.multilinear_ops.graph_linear_operators import GraphLinearOperator
from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize

class MultiLinearOperator:
    """Helper class to represent the application of multiple linear operators in different modes of a tensor.
    
    Can represent operators of the form:
        A = A_1 ⊗ A_2 ⊗ ... ⊗ A_N
    where A_n is a linear operator applied in mode n of the tensor. In particular, when these linear
    operators are sparse matrices, it is possible to use this class to solve the linear system Ax=b 
    using conjugate gradient methods without explicitly forming the full Kronecker product.
    """
    def __init__(self, factor_matrices, matrix_modes=None,
                    device=None,
                    dtype=None):
        """Initialize the MultiLinearOperator with factor matrices and their corresponding modes.

        Args:
            factor_matrices (list of torch.Tensor): List of 2D tensors representing the linear operators
                to be applied in each mode. i.e. factor_matrices[n] @ X_(matrix_modes[n])
            matrix_modes (list of int, optional): Tensor modes to apply linear operators. Defaults to
            [1,...,N] where N is the number of factor matrices.
        """
        if matrix_modes is None:
            matrix_modes = list(range(1, len(factor_matrices) + 1))
        if len(factor_matrices) != len(matrix_modes):
            raise ValueError("Number of factor matrices must be equal to the number of matrix modes.")
        if device is None and len(factor_matrices) > 0:
            device = factor_matrices[0].device
        if dtype is None and len(factor_matrices) > 0:
            dtype = factor_matrices[0].dtype
        self.factor_matrices = [factor_matrix.to(device=device, dtype=dtype) for factor_matrix in factor_matrices]
        self.matrix_modes = matrix_modes
        self.device = device
        self.dtype = dtype
    
    def __call__(self, x):
        """Apply linear operators (from the left) in the factor matrices to the corresponding modes of x.

        Let [A_1, A_2, ..., A_N] be the factor matrices and [n_1, n_2, ..., n_N] be the corresponding modes.
        This computes the following operation:
            X_out = X x_1 A_1 x_2 A_2 ... x_N A_N
            where x_n denotes the mode-n product.
        Args:
            x (torch.Tensor): Input Tensor to apply the operator on.
        """
        N = len(self.factor_matrices)
        if isinstance(x, torch.Tensor):
            X_out = x
        else:
            X_out = torch.tensor(x, device=self.device, dtype=self.dtype)
        if X_out.ndim == 1:
            # If input is one dimensional, turn it into a column vector (-1,1)
            X_out = X_out.reshape(-1, 1)
        for n in range(N):
            A_n = self.factor_matrices[n]
            mode = self.matrix_modes[n]
            if not isinstance(mode, list):
                mode = [mode]
            
            if isinstance(A_n, GraphLinearOperator):
                X_out = tensorize(
                        A_n(matricize(X_out, mode)),
                        X_out.shape, mode)
            else:
                X_out = tensorize(
                    torch.matmul(A_n, matricize(X_out, mode)),
                    X_out.shape, mode)
        return X_out

class SumMultiLinearOperator:
    """Helper class to represent the sum of multiple MultiLinearOperators."""

    def __init__(self, multi_linear_operators):
        """Initialize the SumMultiLinearOperator with a list of MultiLinearOperators.

        Args:
            multi_linear_operators (list of MultiLinearOperator): List of MultiLinearOperators to be summed.
        """
        if len(multi_linear_operators) == 0:
            raise ValueError("List of MultiLinearOperators cannot be empty.")
        self.multi_linear_operators = multi_linear_operators
        self.device = multi_linear_operators[0].device
        self.dtype = multi_linear_operators[0].dtype
    
    @property
    def weights(self):
        """Return the weights of the MultiLinearOperators in the sum."""
        if self._weights is None:
            return [1.0]*len(self.multi_linear_operators)
        else:
            return self._weights
    
    @weights.setter
    def weights(self, weights):
        """Set the weights of the MultiLinearOperators in the sum."""
        if len(weights) != len(self.multi_linear_operators):
            raise ValueError("Number of weights must be equal to the number of MultiLinearOperators.")
        self._weights = weights

    def __call__(self, x, weights=None):
        """Apply the sum of MultiLinearOperators to the input tensor x.

        Args:
            x (torch.Tensor): Input Tensor to apply the operator on.
        """
        if isinstance(x, torch.Tensor):
            X_out = torch.zeros_like(x, device=self.device, dtype=self.dtype)
        else:
            X_out = torch.zeros_like(torch.tensor(x, device=self.device, dtype=self.dtype))
        if X_out.ndim == 1:
            # If input is a vector, turn it into a column vector (-1,1)
            X_out = X_out.reshape(-1, 1)
        if weights is None:
            weights = self.weights
        for i, op in enumerate(self.multi_linear_operators):
            X_out += op(x) * weights[i]
        return X_out