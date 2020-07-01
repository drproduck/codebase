import torch
from torch import nn
import torch.nn.functional as F

def linear_block(d1, d2, act=nn.ReLU()):
    if act is None:
        return nn.Sequential(nn.Linear(d1, d2))
            # nn.Dropout(0.25),
            # nn.BatchNorm1d(d2),
    else:
        return nn.Sequential(nn.Linear(d1, d2),
                            act,
                            )

def conv2d_block(in_c, out_c, stride):
    return nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=out_c, stride=stride),
                        nn.BatchNorm2d(num_features=out_c),
                        nn.ELU(),
                        )

