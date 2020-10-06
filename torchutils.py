import torch
from torch import nn
from mathutils import *
import torch.nn.functional as F
import typing
import gc, sys, os, psutil
import PIL
from torchvision.transforms import ToTensor
import io

        
def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
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
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_params(*modules):
    params = []
    for module in modules:
        params += list(module.parameters())
        
    return params


def to_evals(*modules):
    for module in modules:
        module.eval()
    

def to_trains(*modules):
    for module in modules:
        module.train()
        
        
def tensor2img(x):
    """for pixels in [0, 1]
    """
    return x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8)


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def save_models(dir, epoch, prefix=None, postfix=None, **kwargs):
    """
    Save a model to a dir
    :param epoch: epoch number
    :param kwargs: model name to model
    :return:
    """
    # Save model checkpoints
    if prefix is None: prefix = ''
    if postfix is None: postfix = ''
    for name, model in kwargs.items():
        fname = f"{prefix}_{name}_{epoch:02d}_{postfix}"
        fname += ".pth"
        torch.save(model.state_dict(), os.path.join(dir, fname))
        
        
def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()        
            
            
def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
            

def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)
    
    
def interpolate(p1, p2, n_pts=100):
    """p1, p2 shape = [batch, dim]
    """
    alpha = torch.linspace(0.0, 1.0, n_pts).reshape(1, 1, -1).to(p1)
    delta = p2 - p1
    interpols = p1.unsqueeze(-1) + alpha * delta.unsqueeze(-1)
    return interpols


def to_tensor(x):
    return torch.from_numpy(x).type(torch.float32).cuda()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class Logger():
    def __init__(self, **kwargs):
        self.dict = kwargs
        
    def peek(self, arg):
        return self.dict[arg][-1]
    
    def put(self, arg, val):
        return self.dict[arg].append(val)
    
    def summarize(self, iteration):
        print(f"Iter {iteration}, " + "".join(f"{key}: {value[-1]:.4f}, " for key, value in self.dict.items()))
        
        
        
def plt2img(fig):
    """Create a pyplot plot and save to buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image
