"""Implementation of the Algorithm in the paper 'Matrix Completion on Graphs'

Reference:
    "Matrix Completion on Graphs" by Kalofolias et al. (2014)
    https://arxiv.org/pdf/1408.1717
"""
import sys
from pathlib import Path
from collections import defaultdict
from time import perf_counter
from math import prod
import warnings

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_DIR))

import networkx as nx
import torch
import torch.linalg as la
from torch.nn.functional import softshrink
import matplotlib.pyplot as plt

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
from src.proximal_ops.soft_hosvd import soft_moden
from src.algorithms.admm_base_class import TwoBlockADMMBase
from src.multilinear_ops.graph_linear_operators import GraphLinearOperator
from src.multilinear_ops.multilinear_operator import MultiLinearOperator, SumMultiLinearOperator
from src.algorithms.conjugate_gradient import conjugate_gradient


class MCOnGraphs(TwoBlockADMMBase):
    
    @torch.no_grad()
    def __init__(self, M,
                        mask,
                        graphs,
                        conjugate_gradient_config={'tol': 1e-10, 'max_iter': 500},
                        **kwargs):
        """Initialize the Matrix Completion on Graphs model.

        Args:
            M (torch.Tensor or np.ndarray): Observed data matrix with missing entries.
            mask (torch.Tensor or np.ndarray): Binary mask indicating observed entries in Y.
            graphs (list of networkx.Graph): List containing two graphs for rows and columns.
            kwargs: Additional keyword arguments for the TwoBlockADMMBase class initialization.

        """
        super().__init__(**kwargs)
        if isinstance(mask, torch.Tensor):
            self.mask = mask.to(device=self.device, dtype=torch.bool)
        else:
            self.mask = torch.tensor(mask, device=self.device, dtype=torch.bool)
        if isinstance(M, torch.Tensor):
            self.M = M.to(device=self.device, dtype=self.dtype)
        else:
            self.M = torch.tensor(M, device=self.device, dtype=self.dtype)
        
        self.M = self.M * self.mask # Ensure missing entries are zeroed out
        self.X = torch.zeros_like(self.M, device=self.device, dtype=self.dtype) # Primal variable
        self.Y = torch.zeros_like(self.M, device=self.device, dtype=self.dtype)
        self.Z = torch.zeros_like(self.M, device=self.device, dtype=self.dtype)
        self.graphs = graphs
        self._fb_keys = ['X']
        self._sb_keys = ['Y']
        self._dv_keys = ['Z']
        self.conjugate_gradient_config = conjugate_gradient_config
        self.conjugate_gradient_results = defaultdict(list)

        self.diag_mask = torch.sparse.spdiags(
            self.mask.ravel().to('cpu',self.dtype),
            torch.zeros(1, device='cpu', dtype=torch.long),
            (self.M.numel(), self.M.numel()),
            layout=torch.sparse_csr,
            ).to(device=self.device)
        self.L_c = GraphLinearOperator(self.graphs[0],
                                            operator_type='laplacian', dtype=self.dtype, device=self.device)
        self.L_r = GraphLinearOperator(self.graphs[1],
                                    operator_type='laplacian', dtype=self.dtype, device=self.device)
        self.sum_ml_op = SumMultiLinearOperator([MultiLinearOperator([self.diag_mask],
                                                            [[1,2]], dtype=self.dtype, device=self.device
                                                            ), # Masking operator applied to vectorized matrix
                                        MultiLinearOperator([self.L_c], [1], # Mode-1 column Laplacian product
                                                            dtype=self.dtype, device=self.device),
                                        MultiLinearOperator([self.L_r], [2], # Mode-2 row Laplacian product
                                                            dtype=self.dtype, device=self.device),
                                        MultiLinearOperator([], [], # Identity operator
                                                            dtype=self.dtype, device=self.device)
                                                            ])

    
    def __call__(self, gamma_n, gamma_r, gamma_c, **kwargs):
        super().__call__(**kwargs)
        self.gamma_n = gamma_n
        self.gamma_r = gamma_r
        self.gamma_c = gamma_c

        self.ml_sum_weights = [1.0, gamma_c, gamma_r, self.rhos['Z'][-1]]
        self.sum_ml_op.weights = self.ml_sum_weights

        while not self.converged and self.it < self.max_iter:
            tic = perf_counter()
            self._update_X()
            self._update_Y()
            self._update_Z()

            self.times['iter'].append(perf_counter()-tic)
            self._report_iteration()
            self._check_convergence()
            self._call_tracker()
            if not self.converged:
                self._update_step_size()
                self.it += 1
        self.move_metrics_to_cpu()
        return self.X



    def _update_X(self):
        # X-update step
        tic = perf_counter()
        rho = self.rhos['Z'][-1]
        self.X, nuc_norm = soft_moden((self.Y - self.Z/rho), # Input
                            self.gamma_n/rho, # Threshold
                            1) # Mode-1 SVT
        self.obj.append(nuc_norm*self.gamma_n)
        self.times['X'].append(perf_counter() - tic)
    
    def _update_Y(self):
        # Y-update step
        tic = perf_counter()
        rho = self.rhos['Z'][-1]
        Ynew, info = conjugate_gradient(
            self.sum_ml_op,
            self.M + rho*(self.X + self.Z/rho),
            x0=self.Y,
            **self.conjugate_gradient_config
            )
        self.conjugate_gradient_results['num_iter'].append(info['num_iter'])
        self.conjugate_gradient_results['residual_norms'].append(info['residual_norms'][-1])

        # calculate dual residual norm
        norm = la.norm(Ynew - self.Y, 'fro')
        self.ss['Z'].append(norm*rho)
        self.Y = Ynew
        self.times['Y'].append(perf_counter() - tic)
        self.obj[-1] += 0.5 * self.gamma_r * torch.linalg.norm(self.Y - self.M, 'fro')**2
    
    def _update_Z(self):
        # Z-update step
        tic = perf_counter()
        r = (self.X - self.Y)
        self.Z += self.rhos['Z'][-1] * r

        self.rs['Z'].append(la.norm(r, 'fro'))
        self.r.append(self.rs['Z'][-1])
        self.s.append(self.ss['Z'][-1])
        self.times['Z'].append(perf_counter() - tic)

    @property
    def first_block_keys(self):
        return self._fb_keys
    
    @property
    def second_block_keys(self):
        return self._sb_keys
    
    @property
    def dual_variable_keys(self):
        return self._dv_keys

    @torch.no_grad()    
    def _calc_variable_norms(self):
        return {
            'X': la.norm(self.X, 'fro'),
            'Y': la.norm(self.Y, 'fro'),
            'Z': la.norm(self.Z, 'fro')
        }
        

    def _update_step_size_dependents(self, updated_rho_keys):
        if 'Z' in updated_rho_keys:
            self.ml_sum_weights[3] = self.rhos['Z'][-1]
            self.sum_ml_op.weights = self.ml_sum_weights


    @property
    def model_configuration(self):
        return {'Model': 'Matrix Completion on Graphs',
                'Soft Constrained': self.soft_constrained,
                'Product Graph Type': self.product_graph_type,
                'Reconstruction': True if self.mask is not None else False}

