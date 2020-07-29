from torchvision import datasets
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def load_mnist(dir, batch_size, valid_size=0.1, transform=None, seed=42):
    state = np.random.get_state()
    np.random.seed(seed)
    trainset = datasets.MNIST(root=dir, train=True, download=False, transform=transform)
    testset =  datasets.MNIST(root=dir, train=False, download=False, transform=transform) 

    num_train = len(trainset)
    indices = list(range(num_train))
    
    if 0 < valid_size < 1:
        split = int(valid_size * num_train)
    elif valid_size >= 1:
        split = valid_size
    else:
        split = 0

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, drop_last=True,
                sampler=train_sampler,
                )
    validloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, drop_last=True,
                sampler=valid_sampler
                )

    testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=batch_size, drop_last=True)
    train_size = len(train_idx)
    valid_size = len(trainset) - train_size
    test_size = len(testset)
    np.random.set_state(state)
    return trainloader, validloader, testloader, train_size, valid_size, test_size


if __name__ == '__main__':
    trainloader, validloader, testloader, train_size, valid_size, test_size = load_mnist('/vinai/khiempd1/torch_datasets', valid_size=0, batch_size=128)
    assert(train_size == 60000)
    assert(test_size == 10000)
    assert(valid_size == 0)
