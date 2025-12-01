import os, sys
import warnings
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_DIR))

import networkx as nx
import numpy as np
import torch

from src.utils.sparse_array_conversion import convert_sp_sparse_to_sparse_torch


class GraphLinearOperator:
    """Class to represent a linear operator derived from a graph structure.
    
    This class can create various types of graph-based linear operators such as the graph Laplacian,
    adjacency matrix, and incidence matrix. It provides methods to apply the operator to graph signals,
    compute the graph Fourier transform, and handle connected components."""
    def __init__(self, graph, operator_type='laplacian',
                layout='csr',
                nodelist=None, edgelist=None,
                dtype=torch.float64,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                operator_creation_kwargs={},
                **kwargs):
        self.graph = graph
        self.operator_type = operator_type
        self.nodelist = nodelist if nodelist is not None else list(graph.nodes())
        self.edgelist = edgelist if edgelist is not None else list(graph.edges())
        self.node_index = {node: i for i, node in enumerate(self.nodelist)}
        self.dtype = dtype
        self.device = device
        self._create_linear_operator(operator_creation_kwargs)
        self._connected_components = None
        self._cc_mask = None
        self._cc_basis = None
        self._cc_basis_T = None
        self._basis = None
        self._spectrum = None
        self._edge_basis = None


    def _create_linear_operator(self, kwargs):
        # Adding Normalized Laplacian and Random Walk Laplacian options
        # Adding Half Smoother option
        if self.operator_type == 'laplacian':
            L = nx.laplacian_matrix(self.graph, nodelist=self.nodelist, **kwargs)
            self.A = convert_sp_sparse_to_sparse_torch(nx.laplacian_matrix(self.graph)
                        ).to_sparse_csr().to(dtype=self.dtype, device=self.device)
        elif self.operator_type == 'adjacency':
            A = nx.adjacency_matrix(self.graph, nodelist=self.nodelist, **kwargs)
            self.A = convert_sp_sparse_to_sparse_torch(A).to_sparse_csr(
                    ).to(dtype=self.dtype, device=self.device)
        elif self.operator_type == 'incidence':
            oriented = kwargs.pop('oriented', True)
            B = nx.incidence_matrix(self.graph, 
                                    nodelist=self.nodelist,
                                    edgelist=self.edgelist,
                                    oriented=oriented,
                                    **kwargs)
            self.A = convert_sp_sparse_to_sparse_torch(B).T.to_sparse_csr(
                ).to(dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Unknown operator type: {self.operator_type}")

    @property
    def spectrum(self):
        if self._spectrum is None:
            self._spectrum = torch.linalg.eigvalsh(self.A.to_dense().to(dtype=self.dtype))
            ind = self._spectrum.argsort(descending=True if self.operator_type == 'adjacency'
                                                        else False)
            self._spectrum = self._spectrum[ind]
            # Enforce numerical stability by zeroing out small values that correspond to connected components
            if self.operator_type in ['laplacian', 'incidence']:
                # Enforce numerical stability by zeroing out small values that correspond to connected components
                self._spectrum[:self.n_connected_components] = 0.0
        return self._spectrum

    @property
    def basis(self):
        if self._basis is None:
            if self.operator_type in ['laplacian', 'adjacency']:
                if self.graph.is_directed():
                    self._spectrum, self._basis = torch.linalg.eig(self.A.to_dense().to(dtype=self.dtype))
                else:
                    self._spectrum, self._basis = torch.linalg.eigh(self.A.to_dense().to(dtype=self.dtype))
            elif self.operator_type == 'incidence':
                U, S, Vh = torch.linalg.svd(self.A.to_dense(), full_matrices=False)
                self._basis = Vh.T
                self._spectrum = S
                self._edge_basis = U
            ind = self._spectrum.argsort(descending=True if self.operator_type == 'adjacency'
                                                        else False)
            self._spectrum = self._spectrum[ind]
            self._basis = self._basis[:, ind]
            if self._edge_basis is not None:
                self._edge_basis = self._edge_basis[:, ind]
            
            if self.operator_type in ['laplacian', 'incidence']:
                # Enforce numerical stability by zeroing out small values that correspond to connected components
                self._spectrum[:self.n_connected_components] = 0.0
        return self._basis

    def graph_fourier_transform(self, x):
        return self.basis.T @ x
    
    def inverse_graph_fourier_transform(self, x):
        return self.basis @ x

    def __call__(self, x):
        """Apply the linear operator to the input tensor x of shape (num_nodes, ...) by left-multiplication."""
        if x.dtype != self.dtype:
            warnings.warn((f"Input dtype {x.dtype} does not match operator dtype {self.dtype}." 
                            f"Changing operator dtype to input {self.dtype}."))
            self.A = self.A.to(dtype=x.dtype)
            self.dtype = x.dtype
        if self.A.layout == torch.sparse_csr:
            return torch.sparse.mm(self.A, x)
        else:
            return self.A @ x
    
    @property
    def connected_components(self):
        if self._connected_components is None:
            self._connected_components = list(nx.connected_components(self.graph))
        return self._connected_components

    @property
    def n_connected_components(self):
        return len(self.connected_components)

    @property
    def connected_component_mask(self):
        """Return a boolean mask of support for each connected component of the graph.
        
        Returns:
            torch.Tensor: A boolean tensor of shape (num_nodes, num_connected_components)
        """
        if self._cc_mask is None:
            self._cc_mask = torch.zeros((self.A.shape[0], self.n_connected_components),
                                            dtype=bool, device=self.device)
            for i, comp in enumerate(self.connected_components):
                for node in comp:
                    node_idx = self.node_index[node]
                    self._cc_mask[node_idx, i] = 1
        return self._cc_mask

    def connected_component_means(self, x):
        """Compute the mean of the input tensor over each connected component of the graph.
        
        Args:
            x (torch.Tensor): Input tensor of shape (num_nodes, ...)
        
        Returns:
            torch.Tensor: A tensor of shape (num_connected_components, 1, ...) containing the means.
        """
        mask = self.connected_component_mask
        means = []
        for i in range(mask.shape[1]):
            means.append(x[mask[:,i],...].mean(dim=0, keepdim=True))
        return torch.stack(means, dim=0)
    
    @property
    def cc_basis(self):
        """Constant basis for each connected component of the graph"""
        if self._cc_basis is None:
            self._cc_basis = self.connected_component_mask*torch.tensor(1.0,
                                                        device=self.device, dtype=self.dtype
                                    )/self.connected_component_mask.sum(dim=0, keepdim=True)**0.5

            self._cc_basis = self._cc_basis.to_sparse_csr()

            self._cc_basis_T = self._cc_basis.transpose(0,1).to_sparse_csr()
        return self._cc_basis

    @property
    def cc_basis_T(self):
        if self._cc_basis is None:
            _ = self.cc_basis
        return self._cc_basis_T
    
    def project_onto_constant_space(self, x):
        """Project the input tensor onto the space of constant signals over each connected component.
        
        Args:
            x (torch.Tensor): Input tensor of shape (num_nodes, ...)
        
        Returns:
            torch.Tensor: A tensor of shape (num_nodes, ...) projected onto the constant space.
        """
        return self.cc_basis @ (self.cc_basis_T @ x)
    
    def center_about_constant_space(self, x):
        """Center the graph signal by subtracting the mean over each connected component.
        
        Args:
            x (torch.Tensor): Input tensor of shape (num_nodes, ...)
        
        Returns:
            torch.Tensor: A tensor of shape (num_nodes, ...) centered about the constant space.
        """
        return x - self.project_onto_constant_space(x)
    
    def to(self, device=None, dtype=None):
        if device is not None and device != self.device:
            self.A = self.A.to(device=device)
            self.basis.to(device=device)
            self.spectrum.to(device=device)
            self.cc_basis.to(device=device)
            self.cc_basis_T.to(device=device)
            self.cc_mask = self.cc_mask.to(device=device)
            self.device = device
        if dtype is not None and dtype != self.dtype:
            self.A = self.A.to(dtype=dtype)
            self.basis.to(dtype=dtype)
            self.spectrum.to(dtype=dtype)
            self.cc_basis.to(dtype=dtype)
            self.cc_basis_T.to(dtype=dtype)
            self.cc_mask = self.cc_mask.to(dtype=dtype)
            self.dtype = dtype

        return self