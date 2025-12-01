"""Geometric PCA models for reconstruction and/or denoising of two-way data.

This module contains two implementations of Geometric PCA for matrix data:
- 'InverseGeoPCA': Inverse model,
- 'ForwardGeoPCA': Forward model.

Author: Mert Indibi 9/2/2025
"""


from collections import defaultdict
from time import perf_counter
from math import prod
import warnings

import networkx as nx
import torch
import torch.linalg as la
from torch.nn.functional import softshrink
import matplotlib.pyplot as plt


from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
from src.proximal_ops.soft_hosvd import soft_moden

# from src.metrics.estimate_rank import estimate_tucker_rank


class ProGPCA(TwoBlockADMMBase):
    """Inverse Geometric PCA model for matrix data"""
    def __init__(self, Y, graphs,
                    soft_constrained=True,
                    mask=None,
                    product_graph_type='cartesian',
                    schatten_p = 1,
                    **kwargs):
        super().__init__(**kwargs)
        if isinstance(Y, torch.Tensor):
            self.Y = Y.to(device=self.device, dtype=self.dtype)
        else:
            self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)

        self.mask = mask
        self.graphs = graphs
        self.product_graph_type = product_graph_type
        self.soft_constrained = soft_constrained
        self.schatten_p = schatten_p

    def __call__(self, **kwargs):
        """"""
        super().__call__(**kwargs)
    

    @property
    def model_configuration(self):
        return {'Model': 'Matrix ProGPCA',
                'Soft Constrained': self.soft_constrained,
                'Product Graph Type': self.product_graph_type,
                'Schatten p': self.schatten_p,
                'Reconstruction': True if self.mask is not None else False}

    
    