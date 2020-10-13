import torch
from time import time
from codebase.distance import batch_eudist_sq
import pdb


# utilities

# main funcs
def get_S(C, u, v, eta):
    K = - C + u + v.T
    return torch.exp(K / eta)


def onesided_sinkhorn_uot(C, r, c, eta=0.1, tau=10., n_iter=100):
    """
    \min_{X, X1 = a} <C, X> + \tau KL(X^T 1, b) - H(X)
    
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    :arg eta: entropic-regularizer
    :arg t2: Kl regularizer
    :n_iter: number of Sinkhorn iterations
    """

    with torch.no_grad():
        log_r = torch.log(r + 1e-16)
        log_c = torch.log(c + 1e-16)
    
        # initial solution
        u = torch.zeros_like(r)
        v = torch.zeros_like(c)
    
        for i in range(n_iter):
#             S = get_S(C, u, v, eta)
#             b = S.sum(dim=0).reshape(-1, 1)
            K = - C + u + v.T
            log_b = torch.logsumexp(K.t() / eta, dim=-1, keepdim=True)
            v = (v / eta + log_c - log_b) * (tau * eta / (eta + tau))

            # we end the loop with update of a so that row sum constraint is satisfied.
#             S = get_S(C, u, v, eta)
#             a = S.sum(dim=1).reshape(-1, 1)
            K = - C + u + v.T
            log_a = torch.logsumexp(K / eta, dim=-1, keepdim=True)
            u = (u / eta + log_r - log_a) * eta

        S = get_S(C, u, v, eta), u, v

    return S

if __name__ == '__main__':
    r = torch.rand(10, 1).cuda()
    r = r / r.sum()
    c = torch.rand(20, 1).cuda()
    
    C = torch.rand(10, 20).cuda()
    T = onesided_sinkhorn_uot(C, r, c, eta=0.01, n_iter=10000)
    pdb.set_trace()