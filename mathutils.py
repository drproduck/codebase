import numpy as np
import pdb
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from numpy.linalg import cholesky
import os
from torchvision.utils import save_image
import typing


def normal_sample(mu, var, device):
    return mu + torch.randn(mu.shape).to(device) * var.sqrt()


def interpolate(p1, p2, n_pts=100):
    alpha = np.linspace(0.0, 1.0, n_pts, endpoint=True,)[:,None]
    delta = p2 - p1
    deltas = np.tile(delta, (n_pts, 1))
    interms = p1 + alpha * deltas
    return interms

# def square_interpolate(p1, p2, p3, nrow=10):
#     """
#     Given 3 points, return points in a square grid evenly spaced by 10
#     """
    
def equi_points_2d(n_points=100,
                    center=0.0,
                    size=1.0):
    xstart = center - size
    xstop = center + size
    xco = np.linspace(xstart, xstop, num=n_points)
    yco = np.linspace(xstart, xstop, num=n_points)
    xv, yv = np.meshgrid(xco, yco, indexing='xy')
    xv = xv.reshape(-1, 1)
    yv = yv.reshape(-1, 1)
    pts = np.concatenate((xv, yv), axis=-1)

    return pts

def log_sum_exp(tensor, dim=-1, detach=True):
    """
    :arg detach: if used to compute gradient, must use detach
    """
    max_el = torch.max(tensor, dim=dim, keepdim=True)[0]
    if detach:
        max_el = max_el.detach()
    tensor = tensor - max_el
    log_sum_exp = tensor.exp().sum(dim=dim, keepdim=True).log()
    return max_el + log_sum_exp

def log_mean_exp(tensor, dim, detach=True):
    num_el = tensor.shape[-1]
    return log_sum_exp(tensor, dim, detach) - np.log(num_el)
    
def log_prob_normal(x, mu, var=None):
    """
    log probability of x w.r.t an isotropic gaussian (mu, diag(var)))
    """

    if var is None:
        exp_term = 0.5 * torch.sum((x - mu)**2, dim=-1, keepdim=True)
        norm_term = 0.5 * np.log(2*np.pi) * x.shape[-1]
    if var is not None:
        exp_term = torch.sum((x - mu)**2 / (2 * var), dim=-1, keepdim=True)
        norm_term = 0.5 * np.log(2*np.pi) * x.shape[-1] + 0.5 * torch.sum(torch.log(var), dim=-1, keepdim=True)

    return - exp_term - norm_term


def product_of_diag_normals(mus: typing.List, vars: typing.List):
    """
    vars are variances, not standard deviations
    """
    assert len(mus) == len(vars)
    D = len(mus)
    for i in range(1, len(mus)):
        assert(mus[0].shape == mus[i].shape)
        assert(vars[0].shape == vars[i].shape)
    
    mu = torch.Tensor(size=mus[0].shape).zero_().cuda()
    var = torch.Tensor(size=vars[0].shape).zero_().cuda()
    for i in range(D):
        var = var + vars[i] ** -1.0
    var = var ** -1.0
    for i in range(D):
        mu = mu + mus[i] * (vars[i] ** -1.0)
    mu = mu * var
    return mu, var

def log_prob_bernoulli(input_logits, target):
    logpdf = -1 * F.binary_cross_entropy_with_logits(input_logits, target, reduction='none')
    logpdf = logpdf.sum(dim=-1, keepdim=True)
    return logpdf

def log_prob_softmax(input_logits, target):
    # cross_entropy classes are in 2nd dimension. TODO
    original_shape = target.shape
    n_class = input_logits.shape[-1]
    input_logits = input_logits.view(-1, n_class)
    target = target.view(-1)
    logpdf = -1 * F.cross_entropy(input_logits, target, reduction='none')
    logpdf = logpdf.view(original_shape)
    return logpdf

def log_normal_prior(z):
    return log_prob_normal(z, 0)

def get_importance_bound(elbos):
    with torch.no_grad():
        weights =  elbos - elbos.max(dim=-1, keepdim=True)[0]
        weights = weights.exp()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        # check if weights requires grad
        assert(weights.requires_grad is False)
    weighted_elbos = weights * elbos
    return weighted_elbos

def norm2(X, sq=True):
    if sq:
        return torch.sum(X**2, dim=-1)
    else:
        return torch.sum(X**2, dim=-1).sqrt()

def norm1(X):
    return torch.sum(torch.abs(X))

def get_entropy(P):
    return -1 * torch.sum(P * torch.log(P) - P)


def init_loc_scale(X, n_component, method='gmm'):
    assert(method in ['gmm', 'kmeans'])
    X_dim = X.shape[-1]
    if method == 'gmm':
        gmm = GaussianMixture(n_component, covariance_type='full').fit(X)
        chols = np.zeros(gmm.covariances_.shape)
        for i in range(n_component):
            L = cholesky(gmm.covariances_[i])
            chols[i,:,:] = L
        Loc_init = torch.from_numpy(gmm.means_).type(torch.float32)
        Chol_init = torch.from_numpy(chols).type(torch.float32)
        return Loc_init, Chol_init

    elif method == 'kmeans':
        km = KMeans(n_clusters=n_component, init='k-means++').fit(X)
        Loc_init = torch.from_numpy(km.cluster_centers_).type(torch.float32)
        return Loc_init


def invert_softplus(x):
    return np.log(np.exp(x) - 1)
