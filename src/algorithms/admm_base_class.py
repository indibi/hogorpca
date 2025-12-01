from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import torch

import matplotlib.pyplot as plt

class TwoBlockADMMBase(ABC):
    """Abstract base class for ADMM optimization algorithms."""

    @torch.no_grad()
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.double)
        self.it = 0
        self.converged = False
        self.times = defaultdict(list)
        self.rhos = defaultdict(list)
        self.obj = []                           # objective function
        self.lagrangian = []                    # lagrangian
        self.r = []
        self.s = []
        self.rs = defaultdict(list)
        self.ss = defaultdict(list)

        self.report_freq = kwargs.get('report_freq', 1)
        self.verbose = kwargs.get('verbose', 0)
        self.metric_tracker = kwargs.get('metric_tracker', None)
        self.err_tol = None
        self.rho_update = None
        self.rho_update_thr = None
        self.seed = None

    @torch.no_grad()
    def __call__(self, **kwargs):
        """Iterate over the algorithm until convergence or reaching maximum iterations."""
        self.converged = False
        self.err_tol = kwargs.get('err_tol', 1e-6)
        self.rho_update = kwargs.get('rho_update', 'domain_parametrization')
        self.rho_update_thr = kwargs.get('rho_update_thr', 100)
        self._initialize_rhos(kwargs.get('rho', 0.3))
        self.seed = kwargs.get('seed', None)
        self.max_iter = kwargs.get('max_iter', 100)

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            torch.use_deterministic_algorithms(True)

    def _initialize_rhos(self, rho):
        """Initialize the step sizes (rhos) for the algorithm."""
        # self.rhos = {dual_key: rho for dual_key in self.dual_variable_keys}
        for key in self.dual_variable_keys:
            self.rhos[key].append(rho)

    @property
    @abstractmethod
    def first_block_keys(self):
        """Names/keys of the first block of variables."""
        pass

    @property
    @abstractmethod
    def second_block_keys(self):
        """Names/keys of the second block of variables."""
        pass

    @property
    @abstractmethod
    def dual_variable_keys(self):
        """Names/keys of the dual variables."""
        pass

    @abstractmethod
    def  _calc_variable_norms(self):
        """Calculate the norms of algorithm variables"""
        pass

    @abstractmethod
    def _update_step_size_dependents(self, updated_rho_keys):
        """Update variables dependent on ADMM step sizes"""
        pass

    @torch.no_grad()
    def _update_step_size(self):
        updated_rhos = []
        if not isinstance(self.rho_update, str):
            if self.rho_update <1:
                raise ValueError("Step size growth factor must be larger than 1")
            if self.rho_update != 1.0:
                for key in self.ss.keys():
                    if self.rs[key][-1] > self.ss[key][-1]*self.rho_update_thr:
                        self.rhos[key].append(self.rhos[key][-1]*self.rho_update)
                        updated_rhos.append(key)
                    elif self.ss[key][-1] > self.rs[key][-1]*self.rho_update_thr:
                        self.rhos[key].append(self.rhos[key][-1]/self.rho_update)
                        updated_rhos.append(key)
                    else:
                        self.rhos[key].append(self.rhos[key][-1])
        elif self.rho_update == 'adaptive_spectral':
            raise NotImplementedError("Adaptive spectral step size update is not implemented yet.")
        elif self.rho_update == 'domain_parametrization' and self.it %3==0 and self.it<50:
            #"""Practical use implementation of the domain parametrization step size update.
            #
            #Taken from the paper:
            #'General Optimal Step-size for ADMM-type Algorithms:
            # Domain Parametrization and Optimal Rates',
            # - Yifan Ran 2024
            #"""
            norms = self._calc_variable_norms()
            first_b_norm = sum([norms[key]**2 for key in self.first_block_keys])**0.5
            dual_norm = sum([norms[key]**2 for key in self.dual_variable_keys])**0.5
            if dual_norm>0 and first_b_norm>0:
                new_rho = dual_norm / first_b_norm
                for key in self.rhos.keys():
                    self.rhos[key].append(new_rho)
                    updated_rhos.append(key)
        self._update_step_size_dependents(updated_rhos)


    def _call_tracker(self):
        """Tracks metrics of the state variables of the algorithm using a MetricTracker object.
        
        Please see the Tracker class for more information.
        """
        if self.metric_tracker is not None:
            if self.it % self.metric_tracker.tracker_frequency == 0 or self.converged:
                self.metric_tracker.track(self)

    def _check_convergence(self):
        if self.s[-1] < self.err_tol and self.r[-1] < self.err_tol:
            self.converged = True
            if self.verbose > 1:
                print(f"Converged in {self.it} iterations.")
        return self.converged

    def _report_iteration(self):
        if self.verbose > 0 and self.it % self.report_freq == 0:
            report_text = (f"It-{self.it} "
                           f"\t# |r| = {self.r[-1]:.4e} "
                           f"\t|s| = {self.s[-1]:.4e} "
                           f"\t obj = {self.obj[-1]:.4e} "
                           f"\t {self.times['iter'][-1]:.3f} sec.")
            print(report_text)
            if self.verbose>1:
                for key in self.rs.keys():
                    additional_text = (f"\t# |r_{key}| = {self.rs[key][-1]:.4e} "
                                        f"\t|s_{key}| = {self.ss[key][-1]:.4e} "
                                        f"\t rho_{key} = {self.rhos[key][-1]:.4e}")
                    print(additional_text)


    def plot_alg_run(self, figsize=(6,6)):
        """Plots the algorithm log in 2x2 subplots."""
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()
        axs[0].semilogy(self.obj)
        axs[0].set_title('Objective function')
        axs[1].semilogy(self.r)
        axs[1].set_title('Primal residual')
        axs[2].semilogy(self.s)
        axs[2].set_title('Dual residual')
        for key in self.rhos.keys():
            axs[3].semilogy(self.rhos[key], label=key)
        axs[3].legend()
        axs[3].set_title(r'Step sizes $(\rho)$')
        for ax in axs:
            ax.grid()
            ax.set_xlabel('Iteration')
        return fig, axs

    def move_metrics_to_cpu(self):
        """Move all logged metrics to CPU and convert to numpy arrays."""
        self.obj = torch.Tensor(self.obj).cpu().numpy()
        self.r = torch.Tensor(self.r).cpu().numpy()
        self.s = torch.Tensor(self.s).cpu().numpy()
        self.rhos = {key: torch.Tensor(self.rhos[key]).cpu().numpy() for key in self.rhos.keys()}
        for key in self.rs.keys():
            self.rs[key] = torch.Tensor(self.rs[key]).cpu().numpy()
        if self.metric_tracker is not None:
            self.metric_tracker.move_to_cpu()

    # @property
    # def model_parameters(self):
    #     pass

    # @property
    # def algorithm_parameters(self):
    #     pass

    # @property
    # def model_configuration(self):
    #     pass
