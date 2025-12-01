"""Higher-order Singular Value Decomposition Algorithm.

Proposed in:
    L. De Lathauwer, B. De Moor, and J. Vandewalle, A multilinear singular value decomposition,
    SIAM J. Matrix Anal. Appl., 21 (2000), pp. 1253–1278.

"""
import numpy as np
import torch

# from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.mode_svd import mode_svd
from src.multilinear_ops.mode_product import mode_product

def hosvd(X, modes=None, core_dims=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float64):
    """Higher-order Singular Value Decomposition of a tensor.

    Args:
        X (np.ndarray or torch.Tensor): Tensor to be decomposed
        modes (list): List of modes to be decomposed. Indexing starts at 1, i.e [1,...,N].
            Defaults to all modes.
        core_dims (int or list of int): List of the truncated svd ranks. Defaults to dimensions
            of the decomposition modes.

    Returns:
        core (torch.Tensor): Core tensor of the decomposition
        factors (list of torch.Tensor): List of the factor matrices
        svals (list of torch.Tensor): List of singular values for each mode
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, device=device, dtype=dtype)
    elif isinstance(X, torch.Tensor):
        X = X.to(device=device, dtype=dtype)

    if modes is None:
        modes = list(range(1, len(X.shape) + 1))

    if core_dims is None:
        core_dims = [X.shape[modes[i]-1] for i in range(len(modes))]

    N = len(modes)
    Us = [None for _ in range(N)]
    svals = [None for _ in range(N)]
    for i, mode in enumerate(modes):
        Us[i], svals[i], _ = mode_svd(X, mode, core_dims[i])

    C = mode_product(X, Us[0].T, modes[0])
    for i in range(1, N):
        C = mode_product(C, Us[i].T, modes[i])
    return {'core': C, 'factors': Us, 'svals': svals}



class HoSVD:
    """Higher-order Singular Value Decomposition"""
    @torch.no_grad()
    def __init__(self, X, modes=None, core_dims=None,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                dtype=torch.float64
                ):
        """Initialize the algorithm.

        Args:
            X (np.ndarray or torch.Tensor): Tensor to be decomposed
            modes (list): List of modes to be decomposed. Indexing starts at 1, i.e [1,...,N].
                Defaults to all modes.
            core_dims (int or list of int): List of the truncated svd ranks. Defaults to dimensions
                of the decomposition modes.
            device (str, optional): Device to use for computation. Defaults to 'cuda' if
                available, else 'cpu'.
            dtype (torch.dtype, optional): Data type for the tensor. Defaults to torch.float64.
        """
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, device=device, dtype=dtype)
        elif isinstance(X, torch.Tensor):
            self.X = X.to(device=device, dtype=dtype)
        self.dims = self.X.shape

        if modes is None:
            self.modes = list(range(1, len(self.dims)+1))
        else:
            for mode in modes:
                if mode < 1 or mode > len(self.dims):
                    raise ValueError(f'Mode {mode} is out of range [1,{len(self.dims)}]')
            self.modes = modes
        self.N = len(self.modes)

        
        if core_dims is None:
            self.core_dims = [self.dims[self.modes[i]-1] for i in range(self.N)]
        elif isinstance(core_dims, int):
            if core_dims < 1:
                raise ValueError('Core dimension must be positive')
            elif core_dims > min(self.dims):
                raise ValueError(f'Core dimension {core_dims} is too large. Max dimension is {min(self.dims)}')
            self.core_dims = [core_dims for _ in range(self.N)]
        elif isinstance(core_dims, (list, tuple)):
            core_dims = list(core_dims)
            if len(core_dims)!=self.N:
                raise ValueError(f'Core dimension list must match the number of modes ({self.N})')
            for i, rank in enumerate(core_dims):
                if rank < 1:
                    raise ValueError('Core dimension must be positive')
                elif rank > self.dims[self.modes[i]-1]:
                    raise ValueError(f'Core dimension for mode-{self.modes[i]} is {rank}>{self.dims[self.modes[i]-1]}')
            self.core_dims = core_dims
        
        self.C = None
        self.Us = [None for i in range(self.N)]
        self.svals = [None for i in range(self.N)]

    @torch.no_grad()
    def __call__(self):
        """Decompose the tensor.

        Returns:
            core (torch.Tensor): Core tensor of the decomposition
            factors (list of torch.Tensor): List of the factor matrices
        """
        
        for i in range(self.N):
            self.Us[i], self.svals[i], _ = mode_svd(self.X, i+1, self.core_dims[i])
        
        self.C = mode_product(self.X, self.Us[0].T, 1)
        for i in range(1,self.N):
            self.C = mode_product(self.C, self.Us[i].T, i+1)
        return self.C, self.Us
    
    @torch.no_grad()
    def trank(self, r_tol=None, a_tol=None, method='GCV'):
        """Estimates the multilinear rank of the tensor.

        Args:
            method (str): Method to estimate the rank.
                'GCV' (Generalized Cross-Validation): the default method.
                    select the combination of mode ranks that minimize the GCV score
                    based on residual error and effective number of parameters.
                'SVD_tol': Threshold the mode unfolding singular values based on specified
                    absolute and relative tolerances.
            r_tol (float, optional): Relative tolerance for singular values.
                Defaults to pytorch matrix rank implementation.
            a_tol (float, optional): Absolute tolerance for singular values.
                Defaults to pytorch matrix rank implementation.
        Returns:
            trank (list): Multilinear rank of the tensor
        """
        pass
        # if method == 'GCV':
        
        # elif method == 'SVD_tol':

        #     for i in range(self.N):
        #         mX = matricize(self.X, i+1)
        #         torch.linalg.matrix_rank(self.X)
        #         torch.matrix_rank()
        
