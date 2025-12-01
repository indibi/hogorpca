import numpy as np
import torch

def t2m(X, m=1):
    """Matricisez the tensor X in the m'th mode.
    
    It is done by stacking fibers of mode m as column vectors.
    Order of the other modes follow cyclic order.
    ie ( I_m x I_(m+1). ... .I_N x I_0. ... I_(m-1) ).
    Args:
        X (np.ndarray): Tensor to be matricized
        # m (int, optional): The mode whose fibers are stacked as vectors. Defaults to 1.
    Returns:
        M (np.ndarray): Matricized tensor.
    """
    n = X.shape
    if m>len(n) or m<1:
        raise ValueError(f"Invalid unfolding mode provided. m={m}, X shape:{n}")
    Xm = __roll_2_dim(X,m).ravel().reshape((n[m-1], int(np.prod(n)/n[m-1])))
    return Xm

def __roll_2_dim(X, m):
    n = X.shape; N = len(n)
    dest = np.arange(N)
    src = (np.arange(N) + (m-1))%N
    if isinstance(X, np.ndarray):
        return np.moveaxis(X, src, dest)
    elif isinstance(X, torch.Tensor):
        return torch.moveaxis(X, tuple(src), tuple(dest))

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


def m2t(Xm, dims, m=1):
    """Tensorizes the matrix obtained by t2m to its original state.
# 
    Args:
        Xm (np.ndarray): Matrix
        dims (tuple): original dimensions of the tensor
        m (int): The mode for which the matricization was originally made with t2m. Defaults to 1.
    Returns:  
        X (np.ndarray): Tensor
    """
    N = len(dims); 
    if m > N or m <1:
        raise ValueError(f"Invalid tensorization order m={m}, N={N}")
    
    old_dest = (np.arange(N) + (m-1))%N
    dims2 = tuple([dims[i] for i in old_dest])
    X = Xm.ravel().reshape(dims2)
    X = __unroll_from_dim(X, m)
    return X

def __unroll_from_dim(X, m):
    n = X.shape
    N = len(n)
    dest = (np.arange(N) + (m-1))%N
    src = np.arange(N)
    if isinstance(X, np.ndarray):
        return np.moveaxis(X, src, dest)
    elif isinstance(X, torch.Tensor):
        return torch.moveaxis(X, tuple(src), tuple(dest))

def bic_calculation(Y,S,X,time_mode, ranks, sigma, method , tol = 1e-5):
    
    ''' 
        Y: The observed tensor
        S: estimated sparse tensor or can be the auxiliary variable
        X: low rank tensor
        time_mode: temporal mode
        ranks: 1D torch tensor containing ranks of each mode 
        sigma: error standard deviation
        method: 'horpca', 'lrsts'
        tol: threshold for selecting non-zero entries, in case S is the auxiliary variable can be exactly chosen as zero
        
    '''
    dim = torch.tensor(Y.shape, dtype=torch.float32)  
    ranks = torch.tensor(ranks, dtype=torch.float32) 
    df_X = torch.prod(ranks) + torch.sum((dim * ranks) - (ranks * (1 + ranks) * 0.5))

    ## Df for HoRPCA
    df_S_horpca = torch.sum(S > tol)

    ## DF for LRSTS

    S_t = t2m(S, m = time_mode)
    S_current = S_t[:-1,:] ## except last row of S_t
    S_next = S_t[1:,:] ## except first row of S_t
    condition_nonzero = (S_current > tol) & (S_next > tol)
    condition_equality = (S_current - S_next <= tol)

    df_S_lrsts = df_S_horpca - torch.sum((condition_nonzero & condition_equality))

    ## DF

    df = df_X + (df_S_horpca if method == 'horpca' else df_S_lrsts)

    ## bic calculation 
    two_sigma2 = 2*torch.tensor(sigma**2)
    
    log_likelihood = -(0.5*torch.log(two_sigma2 * torch.pi)) - (two_sigma2*(torch.norm(Y-X-S, p='fro')**2))

    bic = (-2*log_likelihood) + (df*torch.log(torch.prod(dim)))

    return bic
    
    