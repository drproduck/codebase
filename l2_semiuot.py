import torch
import pdb
from math import sqrt
import time
import pdb

def projection_simplex(v, z):
    """
    projection of x onto the simplex, scaled by z:
        p(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        if array, len(z) must be compatible with v
    """
    n_features = v.shape[1]
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
    return torch.sum(C * X) + tau / 2 * norm2sq(X.t().sum(dim=-1, keepdim=True) - b)

def fdual(C, a, b, tau, X):
    beta = tau * (b - X.t().sum(dim=-1, keepdim=True))
    alpha, _ = torch.min(C - beta.t(), dim=-1, keepdim=True)
    d = torch.sum(a * alpha) + torch.sum(b * beta) - 1. / (2 * tau) * norm2sq(beta)

    return d



def grad(C, b, X, tau):
    """
    C: [n_a, n_b]
    X: [n_a, n_b]
    a: [n_a, 1]
    b: [n_b, 1]
    """

    r, c = C.shape
    return C + tau * (X.sum(dim=0, keepdim=True) - b.t())


def prox(X, G, L, a):
    """
    X: [n_a, n_b]
    G: [n_a, n_b]
    """
    Z = X - (0.99 / L) * G
    P = projection_simplex(Z, a)
    return P


def fista(C, a, b, tau, n_iter=100, X=None):
    h, w = C.shape
    if X is None:
        X = torch.rand_like(C)
        X = projection_simplex(X, a)
    t = 1.
    Y = X
    for i in range(n_iter):
        G = grad(C, b, Y, tau)
        XX = prox(Y, G, tau, a)
        tt = (1 + sqrt(1 + 4 * t**2)) / 2
        Y = XX + ((t - 1) / tt) * (XX - X)
        
        t = tt
        X = XX
    return X

def pgd(C, a, b, tau, n_iter=100, X=None):
    h, w, = C.shape
    if X is None:
        X = torch.rand_like(C)
        X = projection_simplex(X, a)

    for t in range(n_iter):
        G = grad(C, b, X, tau)
        X = X - 0.99 / tau * G
        X = projection_simplex(X, a)

    return X


def fw(C, a, b, tau, n_iter=100, X=None):
    r, c = C.shape

    if X is None:
        X = torch.rand_like(C)
        X = X / X.sum(dim=-1, keepdim=True)
        X = a * X # normalize

    for t in range(n_iter):
        gamma = 2 / (t + 2)
        G = grad(C, b, X, tau)
        idx = torch.argmin(G, dim=-1)
        X = (1 - gamma) * X
        X[torch.arange(r), idx] = X[torch.arange(r), idx] + gamma * a.flatten()

    return X


if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.rand(200,1).cuda()
    b = torch.rand(4000,1).cuda()
    C = torch.rand(200, 4000).cuda()
    # a = np.array([0.25, 0.75]).reshape(-1, 1)
    # b = np.array([0.75, 0.25]).reshape(-1, 1)
    # C = np.array([[0,1],[1,0]])
    # # C = (C + C.T) / 2
    tau = 10.
    n_iter = 100

    # t_s = time.time()
    # Xs = a_pgd(C, a, b, tau, n_iter=n_iter, debug=True)
    # print(time.time() - t_s)
    # fs = [fval(C, a, b, tau, X) for X in Xs]
    # fds = [fdual(C, a, b, tau, X) for X in Xs]
    # plt.plot(np.arange(n_iter+1), fs)
    # plt.plot(np.arange(n_iter+1), fds)
    # plt.show()
    # print(Xs[-1])

#     t_s = time.time()
#     Xs = pgd(C, a, b, tau, gamma=0.99/tau, n_iter=n_iter)
#     print(time.time() - t_s)
#     fs = [fval(C, a, b, tau, X) for X in Xs]
#     fds = [fdual(C, a, b, tau, X) for X in Xs]
#     plt.plot(np.arange(n_iter+1), fs)
#     plt.plot(np.arange(n_iter+1), fds)
#     fs = fval(C, a, b, tau, X)
#     fds = fdual(C, a, b, tau, X)
#     print(fs)

    t_s = time.time()
    X = fw(C, a, b, tau, n_iter=n_iter)
    print(time.time() - t_s)
#     fs = [fval(C, a, b, tau, X) for X in Xs]
#     fds = [fdual(C, a, b, tau, X) for X in Xs]
#     plt.plot(np.arange(n_iter+1), fs)
#     plt.plot(np.arange(n_iter+1), fds)
    fs = fval(C, a, b, tau, X)
    fds = fdual(C, a, b, tau, X)
    print(fs, fds)

#     ex_res, ex_X = exact(C, a, b, tau)

#     print(f'exact: f={ex_res:.3f}, X={ex_X}')
#     print(ex_res)
#     print(np.count_nonzero(ex_X))

    # plt.show()
