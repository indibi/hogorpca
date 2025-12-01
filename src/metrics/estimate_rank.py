"""Functions to estimate the rank of a matrix and Tucker rank of a tensor"""

import warnings

import torch
import numpy as np

from src.models.tucker_decomp.hosvd import hosvd


# def estimate_matrix_rank(X, method='GCV'):
#     """Estimate the rank of a matrix
    
#     Args:
#         X (torch.Tensor): Input matrix
#         method (str): Method to estimate the rank. Options are 'GCV' or 'SVD'
        
#     Returns:
#         int: Estimated rank of the matrix
#     """


def estimate_tucker_rank(X, method='GCV',
                            abs_thr=None, rel_thr=None, r2_thr=None, zero_tensor_thr=1e-8,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            dtype=torch.float64):
    """Estimate the Tucker rank of a tensor with HoSVD

    This function estimates the Tucker ranks of a tensor using the Higher-order Singular Value
    Decomposition (HoSVD) and the Generalized Cross-Validation (GCV) criterion. The ranks are
    estimated for each mode of the tensor, and the number of parameters in the HoSVD decomposition
    is also calculated. The GCV criterion is used to find the optimal ranks that minimize the
    residual error of the decomposition. The GCV is computed as:
    .. math::
        GCV = \\frac{D \\cdot E}{(D - P + 1e-20)^2}

    where :math:`D` is the number of elements in the core tensor, :math:`E` is the residual
        frobenius norm, and :math:`P` is the number of parameters in the HoSVD decomposition.
        :math:`P` is computed as:
        .. math::
            P = \\sum_{k=1}^{N} n_k r_k - r_k^2 + \\prod_{k=1}^{N} r_k
        where :math:`n_k` is the size of the k-th mode of the tensor, and :math:`r_k` is the
        estimated rank for that mode.
    Args:
        X (torch.Tensor, np.ndarray): Input tensor
        method (str): Method to estimate the ranks. Defaults to 'GCV'. Other options are 'core_thresholding'
            and 'sv_thresholding'.
        rel_thr (float): Relative threshold for thresholding methods. Defaults to machine epsilon of dtype.
        abs_thr (float): Absolute threshold for thresholding singular values or core entries.
            Defaults to None.
        r2_thr (float): R^2 threshold for thresholding the proportion of energy/variance explained
            by the singular values or core entries.
        device (str): Device to perform the computation on. Defaults to 'cuda' if available, otherwise 'cpu'.
        dtype (torch.dtype): Data type for the tensor. Defaults to torch.float64.
    Returns:
        {'estimated_ranks' (list of int): Estimated Tucker ranks for each mode of the tensor
        'num_params' (float): Number of parameters in the Truncated HoSVD decomposition,
        'r2_value' (float): Proportion of energy explained by the estimated ranks
        'residual_energy' (float): Frobenius norm of the residual after truncation. 
            Can be used to estimate noise variance.}
    """
    # Decompose the tensor using HoSVD and obtain the core tensor and mode singular values
    hosvd_decomp = hosvd(X, device=device, dtype=dtype)
    C = hosvd_decomp['core']
    svals = hosvd_decomp['svals']
    
    # if abs_thr is not None:
    #     C[torch.abs(C) < abs_thr] = 0
    # Calculate the number of parameters in the HoSVD decomposition for possible ranks
    # P = \sum_{k=1}^{N} n_k r_k - r_k^2 + \prod_{k=1}^{N} r_k
    D = C.numel()
    rs = []
    for k, nk in enumerate(C.shape):
        dim_rs = [1]*k + [nk] + [1]*(len(C.shape)-k-1)
        rs.append(torch.linspace(1, nk, steps=nk,
                            device=C.device, dtype=C.dtype
                            ).reshape(dim_rs))    
    product = rs[0]
    for r in rs[1:]:
        product = product * r
    num_params_T = (sum([nk*rk - rk**2 for nk, rk in zip(C.shape, rs)]) + product)

    # Calculate the residual energy of the core tensor for all truncated ranks
    cumulative_energy = torch.cumsum(C**2, dim=0)
    for i in range(1, C.ndim):
        torch.cumsum(cumulative_energy, dim=i, out=cumulative_energy)
    total_energy = cumulative_energy.ravel()[-1]
    residual_energy = total_energy - cumulative_energy

    if zero_tensor_thr is not None:
        if torch.isclose(torch.zeros_like(total_energy), total_energy, atol=zero_tensor_thr):
            warnings.warn((f'Tensor frobenius norm is close to zero ({total_energy:.2e}) in tensor rank estimation.' 
                            ' Returning zeros for all ranks.'), UserWarning)
            return {'estimated_ranks': [0]*C.ndim,
                    'num_param': 0,
                    'r2_value': 0,
                    'residual_energy': 0}


    # Calculate the mode residual variances for truncated ranks for all modes
    if method.upper() == 'GCV':
        gcv = D*residual_energy / (D-num_params_T+1e-20)**2
        gcv.flatten()[-1]= gcv.flatten().max()+1 # Avoid trivial full solution
        index = torch.unravel_index(gcv.argmin(), gcv.shape)
        estimated_ranks = [r.item()+1 for r in index]
        num_param = num_params_T[index].item()
        r2_value = cumulative_energy[index].item() / total_energy.item()
    if method == 'core_thresholding':
        if r2_thr is None:
            r2_thr = 0.99
        r_thr = 1-r2_thr
        if rel_thr is not None:
            warn_msg = ('Relative thresholding is not used for core thresholding method. ',
                        f'Using r2_thr={r2_thr:.4f} instead of rel_thr={rel_thr:.4f} and',
                        f' abs_thr= {abs_thr}.')
            warnings.warn(' '.join(warn_msg), UserWarning)

        accepted_core_mask = residual_energy < r_thr * total_energy
        if abs_thr is not None:
            accepted_core_mask |= residual_energy < abs_thr
        # Find the truncation with the smallest number of parameters
        num_params_T[~accepted_core_mask] = num_params_T.max() + 1 # Set to a large number
        if accepted_core_mask.sum() == 0:
            r2_value = 0
            num_param = 0
            estimated_ranks = [0]*C.ndim
        else:
            index = torch.unravel_index(num_params_T.argmin(), num_params_T.shape)
            estimated_ranks = [r.item()+1 for r in index]
            num_param = num_params_T[index].item()
            r2_value = cumulative_energy[index].item() / total_energy.item()
    if method == 'sv_thresholding':
        raise(NotImplementedError('SV thresholding is not implemented yet'))
        estimated_ranks = []
        for i, sval in enumerate(svals):
            if rel_thr is None:
                r_thr = torch.finfo(C.dtype).eps* max(D/len(sval), len(sval))
            else:
                r_thr = rel_thr
            rejected_svals_mask = sval < r_thr * sval.max()
            
            if abs_thr is not None:
                rejected_svals_mask |=  sval < abs_thr
            if r2_thr is not None:
                cumulative_variance = torch.cumsum(sval.reshape((len(sval),1))**2, dim=0)
                total_variance = cumulative_variance[-1]
                cumulative_variance /= total_variance
                rejected_svals_mask |= ~(cumulative_variance <= r2_thr)
            estimated_ranks.append(int(torch.sum(~rejected_svals_mask).item()))
        num_param = num_params_T[*[r-1 for r in estimated_ranks]].item()
        r2_value = cumulative_energy[*[r-1 for r in estimated_ranks]].item() / total_energy.item()
    return {'estimated_ranks': estimated_ranks,
            'num_param': num_param,
            'r2_value': r2_value,
            'residual_energy': residual_energy[*[r-1 for r in estimated_ranks]].item()}

            
            



