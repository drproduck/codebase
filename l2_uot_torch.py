import torch
import numpy as np
import pdb
from math import sqrt

def projection_simplex(v, z):
    """
    projection of x onto the simplex, scaled by z:
        p(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        if array, len(z) must be compatible with v
    """
    n_features = v.shape[1]
#     pdb.set_trace()
    u = torch.sort(v, dim=1, descending=True)[0]
    cssv = torch.cumsum(u, dim=1) - z
    ind = torch.arange(n_features).to(v) + 1
    cond = u - cssv / ind > 0
    rho = cond.sum(dim=1)
    theta = cssv[torch.arange(len(v)), rho - 1] / rho
    return torch.max(v - theta.reshape(-1, 1), torch.Tensor([0.]).to(v))


def norm2sq(X):
    return torch.sum(X**2)

def fval(C, a, b, tau, X):
    r, c = C.shape
    return torch.sum(C * X) + tau / 2 * norm2sq(X - b.T)


def grad(C, b, X, tau):
    """
    C: [n_a, n_b]
    X: [n_a, n_b]
    a: [n_a, 1]
    b: [n_b, 1]
    """

    r, c = C.shape
    return C + tau * (X - b.T)


def prox(X, G, L, a):
    """
    X: [n_a, n_b]
    G: [n_a, n_b]
    """
    Z = X - (0.9 / L) * G
    P = projection_simplex(Z, a)
    return P


def fista(C, a, b, tau, n_iter=100, X=None):
    h, w = C.shape
#     Xs = []
#     Gs = []
    if X is None:
        X = torch.rand(*C.shape).to(C)
        X = projection_simplex(X, a)
    t = 1.
    Y = X
#     Xs.append(X)
    for t in range(n_iter):
        G = grad(C, b, Y, tau)
        XX = prox(Y, G, tau, a)
        tt = (1 + sqrt(1 + 4 * t**2)) / 2
        Y = XX + ((t - 1) / tt) * (XX - X)
        
        t = tt
        X = XX
#         Xs.append(X)
#         Gs.append(G)

#     return X, Xs, Gs
    return X

if __name__ == '__main__':
    a = torch.rand(2, 5)
    b = torch.rand(2, 1)
    
    c = projection_simplex(a, b)
    print(c)
    print(b)
    print(c.sum(1))