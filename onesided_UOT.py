import torch
from time import time
from distance import batch_eudist_sq
import pdb


# utilities

# main funcs
def get_S(C, u, v, eta):
    n, m = C.shape
    K = - C + u + v.T
    return torch.exp(K / eta)


def onesided_sinkhorn_uot(C, r, c, eta=0.1, t=10.0, n_iter=100):
    """
    \min_{X, X1 = a} <C, X> + \tau KL(X^T 1, b) - H(X)
    
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    :arg eta: entropic-regularizer
    :arg t2: Kl regularizer
    :n_iter: number of Sinkhorn iterations
    """
    # initial solution
    u = torch.zeros_like(r)
    v = torch.zeros_like(c)

    for i in range(n_iter):
        S = get_S(C, u, v, eta)
        a = S.sum(dim=1).reshape(-1, 1)
        u = (u / eta + torch.log(r) - torch.log(a)) * eta

        S = get_S(C, u, v, eta)
        b = S.sum(dim=0).reshape(-1, 1)
        v = (v / eta + torch.log(c) - torch.log(b)) * (t * eta / (eta + t))

    S = get_S(C, u, v, eta)

    return S

if __name__ == '__main__':
    r = torch.rand(10, 1).cuda()
    r = r / r.sum()
    c = torch.rand(20, 1).cuda()
    
    C = torch.rand(10, 20).cuda()
    T = onesided_sinkhorn_uot(C, r, c, eta=0.01, n_iter=10000)
    pdb.set_trace()