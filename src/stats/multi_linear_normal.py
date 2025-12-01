"""Statistical Inference and sampling for the multi-linear normal model"""

from collections import defaultdict
from time import perf_counter
from pprint import pprint

import numpy as np
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize

class MultiLinearNormal:
    def __init__(self, dim, mean=None, covs=None, device=None, dtype=None):
        self.dim = dim
        self.mean = torch.zeros(dim, device=device, dtype=dtype
                                ) if mean is None else torch.tensor(mean, device=device, dtype=dtype)
        self.covs = [torch.eye(dim[i], device=device, dtype=dtype) for i in range(len(dim))
                     ] if covs is None else [
                    torch.tensor(covs[i], device=device, dtype=dtype) for i in range(len(dim))]
        
        self.order = len(dim)
        
        self.device = device
        self.dtype = dtype
        self.info = defaultdict(list)
        self.converged = False
        self.ground_truth = True if mean is not None and covs is not None else False
        self.it = 0
        if self.ground_truth:
            self._sqrt_covs = [torch.linalg.cholesky(cov) for cov in self.covs]
        else:
            self._sqrt_covs = None


    def sample(self, n_samples, seed=None):
        """Generate samples from the multi-linear normal distribution"""
        if seed is not None:
            torch.manual_seed(seed)
        samples = torch.zeros((n_samples, *self.dim), device=self.device, dtype=self.dtype)
        
        E = torch.randn(size=(n_samples, *self.dim), device=self.device, dtype=self.dtype)
        for i in range(len(self.dim)):
            E = matricize(E, [i+2])
            E = self._sqrt_covs[i] @ E
            E = tensorize(E, samples.shape, [i+2])
        samples = E + self.mean
        return samples


    def mle(self, samples, sample_mode=1, err_tol=1e-6, max_iter=100,
                verbose=False, method='eigh', report_freq=1, metric_tracker=None):
        self.err_tol = err_tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.report_freq = report_freq
        self.metric_tracker = metric_tracker
        samples = self._prepare_samples(samples) # Stack the samples if they are in a list
        # Calculate mean
        self.mean = torch.mean(samples, dim=sample_mode-1)
        self._modes = list(range(1, len(samples.shape)+1))
        self._modes.remove(sample_mode)

        E = samples - self.mean  # Center the samples
        self._initialize_method(method)

        while self.it < max_iter and not self.converged:
            iter_start = perf_counter()
            cov_diffs = []
            for i, m in enumerate(self._modes):
                mode_update_start = perf_counter()
                E_n_m = E.clone()
                for j in range(len(self._modes)):
                    if i!=j:
                        E_n_m = self._scale_deviations_for_mode(E_n_m, j)

                # E_n_m = self._scale_deviations_except_mode_m(E, i, modes)
                cov_new = torch.cov(matricize(E_n_m, [m]), correction=0)
                cov_diffs.append(torch.norm(cov_new - self.covs[i], p='fro').cpu().item())

                self.covs[i] = cov_new

                self._update_deviation_scaling(i)
                self.info[f'mode_{m}_update_time'].append(perf_counter() - mode_update_start)
            
            self.info['cov_diffs'].append(cov_diffs)
            self.info['iteration_time'].append(perf_counter() - iter_start)
            self.it += 1
            self._report_iteration()
            self._check_convergence()
            self._call_tracker()

        if self.converged and self.verbose > 0:
            print(f"Converged in {self.it} iterations.")
        elif self.verbose > 0:
            print(f"Did not converge in {self.max_iter} iterations.")
        return self.mean, self.covs        

    def center_and_scale_samples(self, X):
        """Recenter and scale the tensor(s) X with respect to the distribution.

        vec(Xn) = cov^{-1/2} @ vec(X - mean)
        Args:
            X (np.ndarray, torch.Tensor): Tensor(s) to be recentered and scaled.
        
        Returns:
            torch.Tensor: Recentered and scaled tensor(s).
        """
        X = self._prepare_samples(X)
        E_n_m = X - self.mean
        for i in range(len(self.dim)):
            E_n_m = matricize(E_n_m, [i+2])
            if self.method == 'eigh':
                E_n_m = self._mode_cov_sqrt_inv[i] @ E_n_m
            elif self.method == 'cholesky':
                # cov = A @ A^T <-- Cholesky factors
                # E_n = cov^{-1/2} @ E = A^{-1} @ E <-- A is lower triangular
                E_n_m = torch.linalg.solve(self._mode_cov_cholesky_factor[i], E_n_m)
            E_n_m = tensorize(E_n_m, X.shape, [i+2])
        return E_n_m

    def visualize(self, fig_args={}, plot_args={}, **kwargs):
        """Visualize the distributions of the covariance matrices."""
        fig, axes = plt.subplots(len(self.covs), 2, **fig_args)
        cov_names = kwargs.get('cov_names', [f'Covariance Matrix {i+1}' for i in range(self.order)])
        for i, cov in enumerate(self.covs):
            axes[i, 0].imshow(cov.cpu().numpy(), cmap='viridis', interpolation='nearest')
            
            axes[i, 0].set_title(cov_names[i])
            axes[i, 0].axis('off')

            eigvals, eigvecs = torch.linalg.eigh(cov)
            axes[i, 1].plot(eigvals.cpu().numpy(), **plot_args)
            axes[i, 1].set_title(f'Eigenvalues of Covariance Matrix {i+1}')
            axes[i, 1].set_xlabel('Index')
            axes[i, 1].set_ylabel('Eigenvalue')
            axes[i, 1].grid()
        
        fig.tight_layout()
        return fig, axes

    def _initialize_method(self, method):
        """Initialize the method for scaling the deviations from mean."""
        if method == 'eigh':
            self._eigvecs = [torch.eye(self.dim[i], device=self.device, dtype=self.dtype) for i in range(self.order)]
            self._eigvals = [torch.zeros(self.dim[i], device=self.device, dtype=self.dtype) for i in range(self.order)]
            self._mode_cov_sqrt_inv = [torch.eye(self.dim[i], device=self.device, dtype=self.dtype) for i in range(self.order)]
        elif method == 'cholesky':
            self._mode_cov_cholesky_factor = [torch.eye(self.dim[i], device=self.device, dtype=self.dtype) for i in range(self.order)]
        else:
            raise ValueError("Method must be 'eigh' or 'cholesky'")
        self.method = method


    def _scale_deviations_for_mode(self, E, i):
        m = self._modes[i]
        E_n_m = matricize(E, [m])
        if self.method == 'eigh':
            E_n_m = self._mode_cov_sqrt_inv[i] @ E_n_m
        elif self.method == 'cholesky':
            # cov = A @ A^T <-- Cholesky factors
            # E_n = cov^{-1/2} @ E = A^{-1} @ E <-- A is lower triangular
            E_n_m = torch.linalg.solve(self._mode_cov_cholesky_factor[i], E_n_m)
        return tensorize(E_n_m, E.shape, [self._modes[i]])


    def _scale_deviations_except_mode_m(self, E, i, modes):
        E_n_m = E.copy()
        m = modes[i]
        for j in modes[:i]+modes[i+1:]:
            E_n_m = matricize(E_n_m, [m])
            if self.method == 'eigh':
                E_n_m = self._mode_cov_sqrt_inv[i] @ E_n_m
            elif self.method == 'cholesky': 
                # cov = A @ A^T <-- Cholesky factors
                # E_n_m = cov^{-1/2} @ E = A^{-1} @ E <-- A is lower triangular
                E_n_m = torch.linalg.solve(self._mode_cov_cholesky_factor[i], E_n_m)
            E_n_m = tensorize(E_n_m, E.shape, [m])
        return E_n_m


    def _update_deviation_scaling(self, i):
        if self.method == 'eigh':
            eigvals, eigvecs = torch.linalg.eigh(self.covs[i])
            self._eigvecs[i] = eigvecs
            self._eigvals[i] = eigvals
            self._mode_cov_sqrt_inv[i] = eigvecs @ torch.diag(1/torch.sqrt(eigvals)) @ eigvecs.T
        elif self.method == 'cholesky':
            # cov = A @ A^T <-- Cholesky factors
            # E_n = cov^{-1/2} @ E = A^{-1} @ E <-- A is lower triangular
            self._mode_cov_cholesky_factor[i] = torch.linalg.cholesky(self.covs[i])
    
    def _normalize_covariances(self):
        scale = 1.0
        for i in range(len(self.covs)):
            cov_norm = torch.norm(self.covs[i], p=2)
            scale *= cov_norm
            self.covs[i] /= cov_norm
        self.covs[0]*= scale


    def _check_convergence(self):
        if all([diff < self.err_tol for diff in self.info['cov_diffs'][-1]]):
            self.converged = True


    def _report_iteration(self):
        if self.verbose>0 and self.it % self.report_freq == 0:
            print(f"Iteration {self.it}: Differences in covariance matrices")
            pprint(self.info['cov_diffs'][-1])


    def _call_tracker(self):
        """Tracks metrics of the state variables of the algorithm using a MetricTracker object.
        
        Please see the Tracker class for more information.
        """
        if self.metric_tracker is not None:
            if self.it % self.metric_tracker.tracker_frequency == 0 or self.converged:
                self.metric_tracker.track(self)
    

    def _prepare_samples(self, samples):
        if isinstance(samples, list):
            if isinstance(samples[0], torch.Tensor):
                samples = torch.stack(samples)
            elif isinstance(samples[0], np.ndarray):
                samples = torch.stack([torch.tensor(s, device=self.device, dtype=self.dtype) for s in samples])
            else:
                raise TypeError("Samples must be a list of tensors or numpy arrays.")
        elif isinstance(samples, np.ndarray):
            samples = torch.tensor(samples, device=self.device, dtype=self.dtype)
        elif not isinstance(samples, torch.Tensor):
            raise TypeError("Samples must be a tensor or numpy array.")
        return samples
    
    def tensor_outlyingness(self, X):
        """Calculate the outlyingness of a tensor X with respect to the distribution"""
        pass


    def __str__(self):
        return f"MultiLinearNormal(dim={self.dim})"
    

def generate_random_multilinear_normal_distribution(dimensions, seed, scales=1.0, spectrum_distribution='uniform'):
    # if not isinstance(dimensions, list) or isinstance(dimensions, tuple):
    #     raise ValueError("Dimensions must be a list or tuple of integers.")
    
    # rng = np.random.default_rng(seed)

    # mean = rng.normal(size=dimensions)*scales

    # covs = []
    pass





def generate_random_low_rank_multilinear_normal_distribution():
    pass