"""Tensor mode product operations."""

# import torch

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
# from src.multilinear_ops.t2m import t2m
# from src.multilinear_ops.m2t import m2t

def mode_n_product(X, A, mode: int):
    """Compute mode-n product of tensor X with matrix A.
    
    Y = X ×_n A, Y_(mode) = A * X_(mode)
    Args:
        X (torch.Tensor): Input tensor of shape (I1, I2, ..., IN).
        A (torch.Tensor): Matrix of shape (J, In).
        mode (int): Mode along which to multiply (1-indexed).
    Returns:
        torch.Tensor: Resulting tensor of shape (I1, ..., In-1, J, In+1, ..., IN).
    """
    if mode < 1 or mode > X.ndim:
        raise ValueError(f"Invalid mode {mode} for tensor with {X.ndim} dimensions.")
    if A.shape[1] != X.shape[mode-1]:
        raise ValueError(f"Matrix A's number of columns {A.shape[1]} must match "
                         f"tensor X's size in mode {mode}, which is {X.shape[mode-1]}.")
    og_dims = X.shape
    result_dim = list(og_dims)
    result_dim[mode-1] = A.shape[0]

    X_mat = matricize(X, [mode])
    Y_mat = A @ X_mat
    return tensorize(Y_mat, result_dim, [mode])


def multi_mode_product(X, As, modes: list[int], skip_modes: list[int] = []):
    """Compute multiple mode-n products of tensor X with matrices in As along specified modes.

    Args:
        X (torch.Tensor): Input tensor of shape (I1, I2, ..., IN).
        As (torch.Tensor): Factor matrices of shapes [(J1, I_mode1), (J2, I_mode2), ...].
        modes (list[int]): Modes along which to multiply, 1-indexed.
        skip_modes (list[int], optional): Modes to skip during multiplication. Defaults to [].
    Returns:
        Y (torch.Tensor): Resulting tensor after multiple mode-n products.
    """
    for n, mode in enumerate(modes):
        if mode in skip_modes:
            continue
        X = mode_n_product(X, As[n], mode)

    # for A, mode in zip(As, modes):
    #     X = mode_n_product(X, A, mode)
    return X