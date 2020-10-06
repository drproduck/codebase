import torch
from torchutils import *
from torch.nn.functional import softplus
from functools import partial


def batch_eudist(A, B, sq=True, clip=0.):
    """
    A: shape = [..., m, d]
    B: shape = [..., n, d]
    """
    m = A.shape[-2]
    n = B.shape[-2]
    A_norm = torch.sum(A**2, dim=-1, keepdim=True) # [..., m, 1]
    B_norm = torch.sum(B**2, dim=-1, keepdim=True) # [..., n, 1]
    A_xp = expand_newdim(A, n, -2) # [..., m, n, d]
    B_xp = expand_newdim(B, m, -3) # [..., m, n, d]
    AdotB = torch.sum(A_xp * B_xp, dim=-1) # [..., m, n]
    
    res_sq = A_norm + transpose(B_norm) - 2 * AdotB
    if sq:
        return res_sq
    else:
        res = res_sq.clone()
        pos_idx = res > 0
        res[pos_idx] = torch.sqrt(res_sq[pos_idx])
        return res
    
batch_eudist_sq = partial(batch_eudist, sq=True)

def batch_l1(A, B):
    """
    A: shape = [..., m, d]
    B: shape = [..., n, d]
    """
    m = A.shape[-2]
    n = B.shape[-2]
    A_xp = expand_newdim(A, n, -2) # [..., m, n, d]
    B_xp = expand_newdim(B, m, -3) # [..., m, n, d]
    
    return torch.sum(torch.abs(A_xp - B_xp), dim=-1)
    
    
def norm2sq(X):
    """shape = [..., d]
    """
    return torch.sum(X**2, dim=-1)
    
    
def batch_diagnormal_w2(LocScale0, LocScale1, clip_0=True, clip_1=True):
    """LocScale0: shape = [ ..., 2  m, d]
        LocScale1: shape = [..., 2, n, d]
        clip: whether to clip scale by softplus (in case input contains negative)
    """
    m = LocScale0.shape[-2]
    n = LocScale1.shape[-2]
    LocScale0 = expand_newdim(LocScale0, n, -2) # shape = [..., 2, m, n, d]
    Loc0 = LocScale0.select(-4, 0) # [..., m, n, d]
    if clip_0:
        Scale0 = softplus(LocScale0.select(-4, 1)) + 1e-5
    else:
        Scale0 = LocScale0.select(-4, 1)
    
    LocScale1 = expand_newdim(LocScale1, m, -3)
    Loc1 = LocScale1.select(-4, 0)
    if clip_1:
        Scale1 = softplus(LocScale1.select(-4, 1)) + 1e-5
    else:
        Scale1 = LocScale1.select(-4, 1)
    
    M = norm2sq(Loc0 - Loc1) + norm2sq(Scale0 - Scale1) # shape = [..., m, n]
    return M