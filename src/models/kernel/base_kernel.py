import os,sys
from pathlib import Path
from abc import ABC, abstractmethod
import warnings

REPO_DIR = Path().cwd().parent.parent
sys.path.append(str(REPO_DIR))

import numpy as np
import numpy.linalg as la
import torch
import matplotlib.pyplot as plt

class Kernel(ABC):
    def __init__(self, kernel_type):
        self.type = kernel_type
    
    @property
    @abstractmethod
    def parameters(self):
        pass

    @property
    @abstractmethod
    def finite_dimensional(self):
        pass

    @property
    @abstractmethod
    def __call__(self, X, Y=None):
        pass

    @abstractmethod
    def _feature_map(self, X):
        pass
    
    def feature_map(self, X):
        if not self.finite_dimensional:
            raise NotImplementedError("Feature map not implemented for finite-dimensional kernels.")
        else:
            return self._feature_map(X)

class RBFKernel(Kernel):
    def __init__(self, gamma=1.0):
        super().__init__('RBF')
        self.gamma = gamma
    
    @property
    def parameters(self):
        return {'gamma': self.gamma}
    
    @property
    def finite_dimensional(self):
        return False
    
    def _feature_map(self, X):
        # For RBF kernel, the feature map is infinite-dimensional.
        raise NotImplementedError("RBF kernel has an infinite-dimensional feature map.")
    
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        sq_dists = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-self.gamma * sq_dists)


class QuadraticKernel(Kernel):
    def __init__(self, c=0.0):
        super().__init__('Quadratic')
        self.c = c

    @property
    def parameters(self):
        return {'c': self.c}
    
    @property
    def finite_dimensional(self):
        return True
    
    def _feature_map(self, X):
        """Compute the feature map for the quadratic kernel.

        Args:
            X (torch.Tensor or np.ndarray): Input tensor of shape (n_samples, n_features).

        Returns:
            Phi (torch.Tensor): (n_samples, n_features+(n_features*(n_features+1)) // 2 + 1)
        """
        n_samples, n_features = X.shape
        # Number of features in the quadratic feature map
        n_quad_features = n_features + (n_features * (n_features + 1)) // 2 + 1
        Phi = torch.zeros((n_samples, n_quad_features), device=X.device)
        
        # First feature is the bias term
        Phi[:, 0] = self.c
        # Next n_features are the original features
        Phi[:, 1:n_features+1] =  X * (2 * self.c)**0.5
        
        # Remaining features are the pairwise products
        idx = n_features + 1
        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    Phi[:, idx] = X[:, i] * X[:, j]
                else:
                    Phi[:, idx] =  X[:, i] * X[:, j] *(2**0.5)
                idx += 1
        return Phi

    def __call__(self, X, Y=None):
        """Compute the quadratic kernel matrix. K(x, y) = (x^T y + c)^2

        Args:
            X (torch.Tensor): First input tensor of shape (n_samples_X, n_features).
            Y (torch.Tensor, optional): Second input tensor of shape (n_samples_Y, n_features).
                Defaults to None, in which case Y = X.
        """
        if Y is None:
            Y = X
        return (X @ Y.T + self.c) ** 2