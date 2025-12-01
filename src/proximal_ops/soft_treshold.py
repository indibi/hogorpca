import numpy as np
# import jax
# import jax.numpy as jnp

def soft_treshold(x, tau):
    """Soft tresholds any numpy array with element-wise with the treshold tau and returns it.
    Args:
        x (np.array): Vector/matrix/array to be tresholded
        tau (float): Treshold

    Returns:
        y: Tresholded array
    """
    if tau <0:
        raise ValueError("The threshold value tau is negative")
    if tau ==0:
        return x
    y = x.copy().astype('float64')
    y[x>tau] = x[x>tau] -tau
    y[x<-tau] = x[x<-tau] + tau
    y[(-tau<=x) & (x<=tau)]=0
    return y

def soft_threshold(x, tau):
    """Element-wise soft threshold an array, with thresholds in tau
    
    Args:
        x (np.array or torch.Tensor): Input array
        tau (float or np.array): Threshold value(s)
    """
    if isinstance(tau, torch.Tensor or np.ndarray):
        if (tau<0).any():
            raise ValueError("The threshold value tau has negative elements")
        y = x.clone().astype('float64') if isinstance(x, torch.Tensor) else x.copy().astype('float64')
        y[x>tau] = x[x>tau] - tau[x>tau]
        y[x<-tau] = x[x<-tau] + tau[x<-tau]
        y[(x>=-tau) & (x<=tau)] = 0
        return y
    else:
        return soft_treshold(x, tau)