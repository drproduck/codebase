import torch
from time import time
from .distance import batch_eudist_sq
import pdb


# utilities

# main funcs
def get_S(C, u, v, eta):
    n, m = C.shape
    K = - C + u + v.T
    return torch.exp(K / eta)

def sampling_sinkhorn_uot(r_sp, c_sp, eta=1.0, t1=1.0, t2=1.0, n_iter=100):
    dv = r_sp.device
    C = batch_eudist_sq(r_sp, c_sp)
    n_r, n_c = len(r_sp), len(c_sp)
    r = torch.ones(n_r, 1).to(dv) / n_r
    c = torch.ones(n_c, 1).to(dv) / n_c
    return sinkhorn_uot(C, r, c, eta=eta, t1=t1, t2=t2, n_iter=100)

    
def sampling_sinkhorn_uot(r_sp, c_sp, r=None, c=None, eta=0.1, t1=1., t2=1., method='sinkhorn', n_iter=100, gamma=1e-3):
    C = batch_eudist_sq(r_sp, c_sp)
    nr, nc = r_sp.shape[0], c_sp.shape[0]
    if r is None:
        r = torch.ones((nr, 1), dtype=torch.float32).cuda() / nr
    if c is None:
        c = torch.ones((nc, 1), dtype=torch.float32).cuda() / nc
    
    if method == 'sinkhorn':
        return sinkhorn_uot(C, r, c, eta=eta, t1=t1, t2=t2, n_iter=n_iter)
    elif method == 'gd':
        return grad_descent_uot(C, r, c, eta=eta, t1=t1, t2=t2, n_iter=n_iter, gamma=gamma)


def sinkhorn_uot(C, r, c, eta=0.1, t1=1.0, t2=1.0, n_iter=100):
    """
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    :arg eta: entropic-regularizer
    :arg t1: first KL regularizer
    :arg t2: second Kl regularizer
    :n_iter: number of Sinkhorn iterations
    """
    # initial solution
    u = torch.zeros_like(r)
    v = torch.zeros_like(c)
    

    for i in range(n_iter):
        S = get_S(C, u, v, eta)
        a = S.sum(dim=1).reshape(-1, 1)
        u = (u / eta + torch.log(r) - torch.log(a)) * (t1 * eta / (eta + t1))

        S = get_S(C, u, v, eta)
        b = S.sum(dim=0).reshape(-1, 1)
        v = (v / eta + torch.log(c) - torch.log(b)) * (t2 * eta / (eta + t2))

    S = get_S(C, u, v, eta)

    return S

def grad_descent_uot(C, r, c, eta=1.0, t1=1.0, t2=1.0, gamma=1e-3, n_iter=100):
 
    """
    :arg C: cost matrix shape = [r_dim, c_dim]
    :arg r: first marginal shape = [r_dim, 1]
    :arg c: second marginal shape = [c_dim, 1]
    :arg eta: entropic-regularizer
    :arg t1: first KL regularizer
    :arg t2: second Kl regularizer
    :arg gamma: step size
    :n_iter: number of Sinkhorn iterations
    """

    log_r = torch.log(r)
    log_c = torch.log(c).reshape(1, -1)

    X = torch.rand_like(C)

    for it in range(n_iter):
        log_sum_r = X.sum(dim=-1, keepdim=True).log() # [r_dim, 1]
        log_sum_c = X.sum(dim=0, keepdim=True).log() # [1, c_dim]
        delta = C + t1 * log_sum_r + t2 * log_sum_c - t1 * log_r - t2 * log_c + eta * torch.log(X)

        X = torch.max(1e-10, X - gamma * delta)

    return X
