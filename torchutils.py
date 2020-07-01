import torch
from torch import nn
from codebase.mathutils import *
import torch.nn.functional as F
import typing

        
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

        
def is_nan(tensor):
    if torch.sum(torch.isnan(tensor)) > 0:
        return True
    return False


def is_inf(tensor):
    if torch.sum(torch.isinf(tensor)) > 0:
        return True
    return False

def check_nan_inf(tensor, stop=True):
    n_nan = torch.sum(torch.isnan(tensor))
    n_inf = torch.sum(torch.isinf(tensor))
    if n_nan > 0 or n_inf > 0:
        if stop:
            raise Exception(f'tensor contains {n_nan} nan and {n_inf} inf')
        else:
            print(f'tensor contains {n_nan} nan and {n_inf} inf')
    
def repeat_newdim(tensor, n_repeat, dim):
    """
    make a new dimension along a given dim and repeat
    """
    tensor = tensor.unsqueeze(dim=dim)
    repeats = [1 for _ in range(len(tensor.shape))]
    repeats[dim] = n_repeat
    tensor = tensor.repeat(repeats)
    return tensor


def expand_newdim(tensor, n_repeat, dim):
    tensor = tensor.unsqueeze(dim=dim)
    repeats = [-1 for _ in range(len(tensor.shape))]
    repeats[dim] = n_repeat
    tensor = tensor.expand(repeats)
    return tensor


def transpose(tensor):
    """
    tranpose the last 2 dims of tensor
    """
    dims = list(range(len(tensor.shape)))
    t = dims[-1]
    dims[-1] = dims[-2]
    dims[-2] = t
    return tensor.permute(dims)



def grid_contour(pdf, low, high, npts):
    x = np.linspace(low, high, npts)
    y = np.linspace(low, high, npts)
    xv, yv = np.meshgrid(x, y)
    f = np.stack((xv, yv), axis=-1)
    p = pdf(f)
    return xv, yv, p


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

        
class ModuleWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.add_module(k, v)

        self.info = None
        
        
def gettensorinfo(tensor):
    print(tensor.shape, tensor.dtype, tensor.device)