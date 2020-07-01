import torch
import pdb
import numpy as np
import time
from functools import partial
import sys

from .distance import *


    
def pavel_clip(r, eps, n):
    """
    make sure that ||r - r'||_1 \le eps / 4 and min r \ge eps / (8n)
    from Dvurechensky et al.
    """
    r = (1 - eps / 8) * (r + eps / n / (8 - eps))
    return r


def batch_sinkhorn_feydy(r, c, M, eta=0.1, max_iter=50):
    """
    log-stable balanced OT, this formulation is best associated with Francis Bach
    :arg r: first marginal [b, r_dim, 1]
    :arg c: second marginal [b, c_dim, 1]
    :arg M: cost matrix [b, r_dim, c_dim]
    
    for symmetric sinkhorn, the # iters actually doubles
    """
    r_log = torch.log(r)
    c_log = torch.log(c)
    M_eta = M / eta

    # initial solutions
    u = torch.zeros_like(r)
    v = torch.zeros_like(c)
    
    # update
    # torch supports logsumexp. how nice!
    for i in range(max_iter):
        u = - eta * torch.logsumexp(transpose(v) / eta + transpose(c_log) - M_eta, dim=-1, keepdim=True)
        v = - eta * torch.logsumexp(transpose(u) / eta + transpose(r_log) - transpose(M_eta), dim=-1, keepdim=True)

    P = get_P(r_log, c_log, u / eta, v / eta, M_eta)
    unreg_val = torch.sum(P * M, dim=[-1, -2])
    reg_val = torch.sum(u * r, dim=[-1, -2]) + torch.sum(v * c, dim=[-1, -2])
    return unreg_val, reg_val, u, v
    
    
def get_P(r_log, c_log, u_eta, v_eta, M_eta):
    return torch.exp(u_eta + transpose(v_eta) - M_eta + r_log + transpose(c_log))


def sampling_sinkhorn_divergence(r_sp, c_sp, r=None, c=None, eta=0.1, max_iter=100, batch_cost_fn=batch_eudist_sq, ret_plan=False):
    batch1 = r_sp.shape[0]
    batch2 = c_sp.shape[0]
    if r is None:
        r = torch.ones((batch1, 1), dtype=r_sp.dtype, device=r_sp.device) / batch1
    if c is None:
        c = torch.ones((batch2, 1), dtype=c_sp.dtype, device=c_sp.device) / batch2
    reg_val, P, unreg_val = batch_sinkhorn_divergence(r.unsqueeze(0), c.unsqueeze(0), r_sp.unsqueeze(0), c_sp.unsqueeze(0), 
                    max_iter=max_iter, eta=eta, batch_cost_fn=batch_cost_fn, ret_plan=ret_plan)

    if ret_plan:
        return reg_val.squeeze(0), P.squeeze(0)
    else:
        return reg_val.squeeze(0)
        

def batch_sinkhorn_divergence(r, c, r_sp, c_sp, eta=0.1, max_iter=100, batch_cost_fn=None, ret_plan=False, ret_unreg_val=False, ret_potentials=False):
    """
    r: shape = [b, m, 1]
    c: shape = [b, n, 1]
    r_sp: shape = [b, m, d] # the supports of r
    c_sp: shape = [b, n, d]
    """
    if batch_cost_fn is None:
        batch_cost_fn = batch_eudist_sq
    M_rc = batch_cost_fn(r_sp, c_sp) / eta
#     M_rr = batch_cost_fn(r_sp, r_sp.detach()) / eta
#     M_cc = batch_cost_fn(c_sp, c_sp.detach()) / eta 
    M_rr = batch_cost_fn(r_sp, r_sp) / eta
    M_cc = batch_cost_fn(c_sp, c_sp) / eta 
    check_nan_inf(M_rc, stop=True)
    check_nan_inf(M_rr, stop=True)
    check_nan_inf(M_cc, stop=True)
    
    return batch_sinkhorn_divergence_M(r, c, M_rc, M_rr, M_cc, eta=0.1, max_iter=100, 
                                       ret_plan=ret_plan, ret_unreg_val=ret_unreg_val, ret_potentials=ret_potentials)


def batch_sinkhorn_divergence_M(r, c, M_rc, M_rr, M_cc, eta=0.1, max_iter=100, ret_plan=False, ret_unreg_val=False, ret_potentials=False):
    
#     r = pavel_clip(r, 1e-7, r.shape[-2])
#     c = pavel_clip(c, 1e-7, c.shape[-2])
    r_log = torch.log(r)
    c_log = torch.log(c)

    # initial solutions
    f = torch.zeros_like(r)
    g = torch.zeros_like(c)
    p = torch.zeros_like(r)
    q = torch.zeros_like(c)
     
    with torch.no_grad():
        for i in range(max_iter - 1):
            f = - eta * torch.logsumexp(transpose(g) / eta + transpose(c_log) - M_rc, dim=-1, keepdim=True)
            g = - eta * torch.logsumexp(transpose(f) / eta + transpose(r_log) - transpose(M_rc), dim=-1, keepdim=True)
            
            """p = (p_last + p_curr) / 2. Recall that the optimal potentials are unique up to constant factor.
            """
            p = 0.5 * (p - eta * torch.logsumexp(transpose(p) / eta + transpose(r_log) - M_rr, dim=-1, keepdim=True))
            q = 0.5 * (q - eta * torch.logsumexp(transpose(q) / eta + transpose(c_log) - M_cc, dim=-1, keepdim=True))
        
            
#     print(torch.sum(r * f, dim=[-1, -2]) + torch.sum(c * g, dim=[-1, -2]))
#     print(torch.sum(r * p * 2, dim=[-1, -2]))
#     print(torch.sum(c * q * 2, dim=[-1, -2]))

    # last iteration to enable gradient
    f = - eta * torch.logsumexp(transpose(g).detach() / eta + transpose(c_log).detach() - M_rc, dim=-1, keepdim=True)
    g = - eta * torch.logsumexp(transpose(f).detach() / eta + transpose(r_log).detach() - transpose(M_rc), dim=-1, keepdim=True)

    p = 0.5 * (p - eta * torch.logsumexp(transpose(p).detach() / eta + transpose(r_log).detach() - M_rr, dim=-1, keepdim=True))
    q = 0.5 * (q - eta * torch.logsumexp(transpose(q).detach() / eta + transpose(c_log).detach() - M_cc, dim=-1, keepdim=True))

    
    u = f - p
    v = g - q
    check_nan_inf(u)
    check_nan_inf(v)
    reg_val = torch.sum(r * u, dim=[-1, -2]) + torch.sum(c * v, dim=[-1, -2])
    
    P = get_P(r_log, c_log, u / eta, v / eta, M_rc) if ret_plan or ret_unreg_val else None
    unreg_val = torch.sum(P * M_rc, dim=[-1, -2]) if ret_unreg_val else None
    return reg_val, P, unreg_val
