"""Singular Value Thresholding (SVT) implementation for matrix reconstruction.

References:
    1. Cai, Jian-Feng, Emmanuel J. Candès, and Zuowei Shen. "A 
        singular value thresholding algorithm for matrix completion."
        SIAM Journal on optimization 20.4 (2010): 1956-1982.
    2. E. J. Candès, C. A. Sing-Long and J. D. Trzasko, "Unbiased
        Risk Estimates for Singular Value Thresholding and Spectral
        Estimators," in IEEE Transactions on Signal Processing, vol. 61,
        no. 19, pp. 4643-4657, Oct.1, 2013, doi: 10.1109/TSP.2013.2270464
"""


import numpy as np
import numpy.linalg as la
import torch
import matplotlib.pyplot as plt
import networkx as nx

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize

# Not tested yet
class SVThresholding:
    """Singular Value Thresholding (SVT) for matrix reconstruction."""
    def __init__(self, Y,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    dtype=torch.float64):
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.Y = Y.to(device, dtype)
        self.m, self.n = Y.shape[-2:],
    
    def _decompose(self):
        """Analyze the singular values of the input matrix."""
        U, S, Vt = torch.linalg.svd(self.Y, full_matrices=False)
        self.U = U
        self.S = S
        self.Vt = Vt
    
    def _degrees_of_freedom(self, tau):
        if self.S is None:
            self._decompose()
        
        nonzero_S = self.S[self.S > 0]
        dof = abs(self.m - self.n) * torch.sum(
                    torch.clamp(1 - tau / nonzero_S, min=0)
                ) + min(self.m, self.n) * torch.sum(
                    (nonzero_S > tau).to(self.dtype)
                )
        gt_tau = nonzero_S > tau
        s_gt_tau = nonzero_S[gt_tau]
        for i in range(len(s_gt_tau)):
            for j in range(i, len(s_gt_tau)):
                if i != j:
                    dof += 2*(s_gt_tau[i]*(s_gt_tau[i]-tau)/(s_gt_tau[i]**2-s_gt_tau[j]**2)
        return dof

    def __call__(self, tau):
        """Apply singular value thresholding with threshold tau.

        Args:
            tau (float): Threshold value.

        Returns:
            X_denoised (torch.Tensor): Denoised matrix after applying SVT.
        """
        if self.S is None:
            self._decompose()
        S_thresholded = torch.clamp(self.S - tau, min=0)
        return self.U * S_thresholded @ self.Vt
    
    def steins_unbiased_risk_estimate(self, variance_estimate, tau):
        """Compute Stein's Unbiased Risk Estimate (SURE) for SVT.

        Args:
            variance_estimate (float): Estimate of the noise variance.
            tau (float): Threshold value.

        Returns:
            sure (float): SURE value.
        """
        if self.S is None:
            self._decompose()
        
        dof = self._degrees_of_freedom(tau)
        torch.sum(torch.minimum(self.S, torch.tensor(tau)))
        sure = -self.m * self.n * variance_estimate + frob_norm_sq + 2 * variance_estimate * dof
        return sure.item()
