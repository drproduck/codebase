import torch
from torchvision.utils import save_image
from codebase.torchutils import *
import math
from PIL import Image
import os
import numpy as np
from codebase.mathutils import equi_points_2d


class DictObject(object):
    def __init__(self, **kwargs):
        for k, w in kwargs.items():
            setattr(self, k, w)


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

def sample_images(dir, generator, noise, epoch, batches_done, 
        nrow=10, 
        num_images=100, 
        padding=2,
        pad_value=0,
        writer=None,
        ):
    """
    Saves a generated sample from the validation set
    Code taken from torchvision

    """

    samples = torch.sigmoid(generator(noise))[:num_images]
    samples = samples.view(-1, 28, 28)
    
    nmaps = samples.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(samples.size(1) + padding), int(samples.size(2) + padding)
    grid = samples.new_full((height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(0, y * height + padding, height - padding)\
                .narrow(1, x * width + padding, width - padding)\
                .copy_(samples[k])
            k = k + 1

    grid = grid.mul_(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)
    image = Image.fromarray(grid.numpy())
    if writer is None:
        image.save(os.path.join(dir, f"{epoch:02d}_{batches_done:05d}.png"))
    else:
        if len(grid.shape) == 2:
            repeat_newdim(grid, 1, 0)
        writer.add_image('images', grid, global_step=epoch)



def sample_2d_grid(dir, generator, epoch, batches_done, 
                    n_points = 20,
                    center=0.0,
                    size=1.0,
                    padding=2, 
                    pad_value=0,
                    writer=None,
                    ):
    """
    Save samples from an equally distributed latent points in a 2d square grid
    :arg nrow: number of interpolated points along each dimension
    :arg center: center of the square grid
    :arg size: size of the square grid
    """
    pts = equi_points_2d(n_points, center, size)    
    pts = torch.Tensor(pts).cuda()
    samples = torch.sigmoid(generator(pts))
    samples = samples.view(-1, 28, 28)
    
    nmaps = samples.size(0)
    xmaps = min(n_points, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(samples.size(1) + padding), int(samples.size(2) + padding)
    grid = samples.new_full((height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(0, y * height + padding, height - padding)\
                .narrow(1, x * width + padding, width - padding)\
                .copy_(samples[k])
            k = k + 1

    grid = grid.mul_(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8) 
    image = Image.fromarray(grid.numpy())
    if writer is None:
        image.save(os.path.join(dir, f"{epoch:02d}_{batches_done:05d}.png"))
    else:
        if len(grid.shape) == 2:
            grid = repeat_newdim(grid, 1, 0)
        writer.add('sample_grid', repeat_newdim(grid, 1, 0), global_step=epoch)


if __name__ == '__main__':
    pts = equi_points_2d(n_points=3)
    print(pts)