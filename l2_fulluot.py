import torch
import pdb
from math import sqrt


def norm2sq(X):
    return torch.sum(X**2)


def fval(C, a, b, tau1, tau2, X):
    r, c = C.shape
    return torch.sum(C * X) + tau1 / 2 * norm2sq(X - a) \
                            + tau2 / 2 * norm2sq(X - b.T)

def fdual(C, a, b, tau1, tau2, X):
    alpha = -tau1 * (X.sum(dim=-1, keepdim=True) - a)
    beta = -tau2 * (X.t().sum(dim=-1, keepdim=True) - b)
    d = torch.sum(alpha * a) + torch.sum(beta * b) - 1. / (4 * tau1) * norm2sq(alpha) - 1. / (4 * tau2) * norm2sq(beta)
    
    return d

def grad(C, a, b, X, tau1, tau2):
    """
    C: [n_a, n_b]
    X: [n_a, n_b]
    a: [n_a, 1]
    b: [n_b, 1]
    """

    r, c = C.shape
    return C + tau1 * (X - a) + tau2 * (X - b.T)


def prox(X):
    X[X < 0] = 0
    return X


def fista(C, a, b, tau1, tau2, n_iter=100, X=None):
    h, w = C.shape
    if X is None:
        X = torch.rand_like(C)
        X = prox(X)
    t = 1.
    Y = X
    for t in range(n_iter):
        G = grad(C, a, b, Y, tau1, tau2)
        XX = X - 0.99 / (tau1 + tau2) * G
        XX = prox(XX)

        tt = (1 + sqrt(1 + 4 * t**2)) / 2
        Y = XX + ((t - 1) / tt) * (XX - X)
        
        t = tt
        X = XX

    return X

def pgd(C, a, b, tau1, tau2, n_iter=100, X=None):
    if X is None:
        X = torch.rand_like(C)
        X = prox(X)

    for t in range(n_iter):
        G = grad(C, a, b, X, tau1, tau2)
        X = X - 0.99 / (tau1 + tau2) * G
        X = prox(X)

    return X


if __name__ == '__main__':
    a = torch.rand(100, 1)
    b = torch.rand(200, 1)
    C = torch.rand(100, 200)
    X = pgd(C, a, b, 1., 1.)
    print(fval(C, a, b, 1., 1., X))
    print(fdual(C, a, b, 1., 1., X))
