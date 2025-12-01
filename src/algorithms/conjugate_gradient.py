"""Conjugate Gradient Algorithm

Implementation of the Conjugate Gradient algorithm for solving systems of linear equations.
Useful for large, sparse systems where direct methods are impractical.

References:
    'An Introduction to the Conjugate Gradient Method Without the Agonizing Pain' (http://www.cs.cmu.edu/%7Equake-papers/painless-conjugate-gradient.pdf) by Jonathan Richard Shewchuk.
    'Iterative methods for sparse linear systems' (http://www-users.cs.umn.edu/%7Esaad/books.html) by Yousef Saad
"""

from collections import defaultdict
import numpy as np
import torch


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Solve the system of linear equations Ax = b using the Conjugate Gradient method.

    Parameters:
        A (callable or torch.Tensor): A function that computes the matrix-vector product Ax or a symmetric positive-definite matrix.
        b (torch.Tensor): The right-hand side vector.
        x0 (torch.Tensor, optional): Initial guess for the solution. If None, defaults to a zero vector.
        tol (float, optional): Tolerance for convergence. The algorithm stops when the residual norm is below this value.
        max_iter (int, optional): Maximum number of iterations.

    Returns:
        x (torch.Tensor): The approximate solution vector.
        info (dict): Dictionary containing information about the convergence process.
    """
    if isinstance(A, torch.Tensor):
        A_mat = A
        A = lambda x: torch.matmul(A_mat, x)
    
    if not torch.is_tensor(b):
        b = torch.tensor(b, device=A_mat.device if 'A_mat' in locals() else None,
                            dtype=A_mat.dtype if 'A_mat' in locals() else None)
    dotprod = torch.dot if b.ndim == 1 else lambda x, y: torch.sum(x * y)

    n = b.numel()
    if x0 is None:
        x = torch.zeros(b.shape, device=b.device, dtype=b.dtype)
    else:
        x = x0.clone()

    max_iter = max_iter if max_iter is not None else n
    r = b - A(x)
    p = r.clone()
    rs_old = dotprod(r,r)#torch.dot(r, r)

    info = {'num_iter': 0, 'residual_norms': [torch.sqrt(rs_old)]}

    for i in range(max_iter):
        Ap = A(p)
        alpha = rs_old / dotprod(p,Ap)#torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = dotprod(r,r)#torch.dot(r, r)

        info['residual_norms'].append(torch.sqrt(rs_new))
        info['num_iter'] += 1

        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    info['residual_norms'] = torch.tensor(info['residual_norms']).cpu().numpy()
    return x, info


