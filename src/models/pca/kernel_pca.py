"""Kernel PCA (Principal Component Analysis) using kernel methods.

References:
    1. Schölkopf, Bernhard, Alexander Smola, and Klaus-Robert Müller. 
        "Nonlinear component analysis as a kernel eigenvalue problem."
        Neural computation 10.5 (1998): 1299-1319.
"""


import os,sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_DIR))
from src.models.kernel.base_kernel import QuadraticKernel

import numpy as np
import numpy.linalg as la
import torch
import matplotlib.pyplot as plt
import networkx as nx

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize

class KernelPCA:
    @torch.no_grad()
    def __init__(self, input_tensor, feature_modes=[1],
                    kernel=QuadraticKernel(c=1.0),
                    device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Principal Component Analysis (PCA).
        
        Args:
            input_tensor (torch.Tensor or np.ndarray): Input data tensor.
            feature_modes (list, optional): Modes to be treated as features. Defaults to [1].
            kernel
            device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cuda' if available else 'cpu'.
        """
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor, device=device)
        self.device = device
        self.kernel = kernel

        self.og_dim = input_tensor.shape
        self.feature_modes = feature_modes
        self._cov = None

        self.input = matricize(input_tensor, feature_modes).to(device)

    @torch.no_grad()
    def fit(self, n_components=None):
        
        """Fit the Kernel PCA model to the input data.

        Args:
            n_components (int, optional): Number of principal components to retain.
                If None, all components are retained. Defaults to None.
        """
        n_samples = self.input.shape[0]
        if n_components is None:
            n_components = n_samples

        # Compute the kernel matrix
        self.K = self.kernel(self.input).cpu().numpy()
        self.K_centered = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()

        # Eigen decomposition
        eigvals, eigvecs = torch.linalg.eigh(self.K_centered)
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        # Retain only the top n_components
        self.eigvals_ = eigvals[:n_components]
        self.eigvecs_ = eigvecs[:, :n_components]
        # Normalize eigenvectors
        self.alphas_ = self.eigvecs_ / (self.eigvals_ + 1e-10)**0.5

    @torch.no_grad()
    def eigen_functions(self):
        """Compute the eigen functions of the kernel PCA.
        
        Returns:
            phi_center (torch.Tensor): Centered feature map of shape (n_features, 1).
            phi ( torch.Tensor): Eigen functions of shape (n_features, n_components).
        """
        if self.kernel.finite_dimensional:
            Phi = self.kernel.feature_map(self.input.T).to(self.device, dtype=self.alphas_.dtype)
            Phi_center = Phi.mean(dim=0, keepdim=True).to(self.device, dtype=self.alphas_.dtype)
            return Phi_center.T, ((Phi-Phi_center).T @ self.alphas_).T
        else:
            raise NotImplementedError("Eigen functions not implemented for infinite-dimensional kernels.")

    def project(self, X_test):
        """Project new data points into the kernel PCA space.

        Args:
            X_test (torch.Tensor or np.ndarray): New data points to project.

        Returns:
            X_proj (torch.Tensor): Projected data points in the kernel PCA space.
        """
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, device=self.device)
        K_test = self.kernel(X_test, self.input)
        K_test_centered = K_test - K_test.mean(axis=1, keepdims=True) - self.K.mean(axis=0, keepdims=True) + self.K.mean()
        X_proj = K_test_centered @ self.alphas_
        return X_proj


    def transform(self, X):
        ## TODO implement
        pass