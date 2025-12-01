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