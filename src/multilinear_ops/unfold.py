import numpy as np
import torch

@torch.jit.script
def __roll_2_dim(X, m:int):
    n = X.shape
    N = X.ndim
    dest = torch.arange(N)
    src = (torch.arange(N) + (m-1))%N
    return torch.moveaxis(X, src, dest)

@torch.jit.script
def unfold(X, m:int =1):
    """Matricize the tensor X in the m-th mode.
    
    It is done by stacking fibers of mode m as column vectors.
    Order of the other modes follow cyclic order.
    ie ( I_m x I_(m+1). ... .I_N x I_0. ... I_(m-1) ).
    Args:
        X (np.ndarray or torch.Tensor): Tensor to be matricized
        m (int, optional): The mode whose fibers are stacked as vectors. Defaults to 1.
    Returns:
        M (np.ndarray or torch.Tensor): Matricized tensor.
    """
    n = X.shape
    N = X.ndim
    if m>N or m<1:
        raise ValueError(f"Invalid unfolding mode provided. m={m}, X shape:{n}")
    Xm = __roll_2_dim(X,m).ravel().reshape(n[m-1],-1) #int(X.numel()//n[m-1])))
    return Xm



def convert_index(n,i,k):
    """Convert the index of an element of a tensor into it's k'th mode unfolding index

    Args:
        n (tuple): Tensor shape
        i (tuple): Tensor index
        k (int): Mode unfolding
    Returns:
        idx (tuple): Index of the element corresponding to the matricized tensor
    """
    if type(n) != tuple:
        raise TypeError("Dimension of the tensor, n is not a tuple")
    if type(i) != tuple:
        raise TypeError("index of the tensor element, i is not a tuple")
    if len(n) ==1:
        raise ValueError(f"The provided dimension n={n} is for the vector case")
    for j, i_ in enumerate(i):
        if i_>= n[j]:
            raise ValueError(f"Index i exceeds the dimension n in {j+1}'th mode")
    if k > len(n) or k<1:
        raise ValueError(f"Unfolding mode {k} is impossible for {len(n)}'th order tensor.")
    j=0
    idx = [i[k-1]]
    n_ = list(n)
    n_ = n_[k:] + n_[:k-1]
    i_ = list(i)
    i_ = i_[k:] + i_[:k-1]
    for p,i_k in enumerate(i_):
       j+= i_k*np.prod(n_[p+1:])
    idx.append(int(j))
    return tuple(idx)

@torch.jit.script
def m2t(Xm, dims, m=1):
    """Tensorize the matrix obtained by unfold to its original state.

    Args:
        Xm (np.ndarray, or torch.Tensor): Matrix
        dims (tuple): original dimensions of the tensor
        m (int): The mode for which the matricization was originally made with t2m. Defaults to 1.
    Returns:
        X (np.ndarray or torch.Tensor): Tensor
    """
    N = len(dims); 
    if m > N or m <1:
        raise ValueError(f"Invalid tensorization order m={m}, N={N}")
    
    old_dest = (np.arange(N) + (m-1))%N
    dims2 = tuple([dims[i] for i in old_dest])
    X = Xm.ravel().reshape(dims2)
    X = __unroll_from_dim(X, m)
    return X

@torch.jit.script
def __unroll_from_dim(X, m):
    n = X.shape
    N = len(n)
    dest = (np.arange(N) + (m-1))%N
    src = np.arange(N)
    if isinstance(X, np.ndarray):
        return np.moveaxis(X, src, dest)
    elif isinstance(X, torch.Tensor):
        return torch.moveaxis(X, tuple(src), tuple(dest))
