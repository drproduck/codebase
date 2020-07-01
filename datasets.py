import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import MNIST
import random
import pdb
import os 
import pandas as pd
from PIL import Image
import numpy as np
import random
from torchvision.datasets import MNIST
from torchvision import transforms

import gzip
import pickle as pkl
from urllib import request
from tqdm import tqdm

def _download_mnist(root_dir):
    data_loc = os.path.join(root_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_loc):
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print("Downloading data from:", url)
        data_loc = os.path.join(root_dir, 'mnist.pkl.gz')
        data_loc, _ = request.urlretrieve(url, data_loc)
    else: print('WARNING: data might already exist')
    return data_loc

def _load_mnist(root_dir,  split_type, download):
    if download:
        data_loc = _download_mnist(root_dir)
    else:
        data_loc = os.path.join(root_dir, 'mnist.pkl.gz')
    f = gzip.open(data_loc, 'rb')

    train, valid, test = pkl.load(f, encoding='bytes')
    f.close()
    if split_type == 'train':
        x, y = train[0], train[1]
    if split_type == 'valid':
        x, y = valid[0], valid[1]
    if split_type == 'test':
        x, y = test[0], test[1]
    return x, y


class MNISTAddOneDataset(Dataset):
    def __init__(self, mnist_root, transform=None):
        super().__init__()
        self.mnist_root = mnist_root
        self.transform = transform
        mnist_trainset = MNIST(mnist_root, train=True, download=True, transform=transform)
        mnist_testset = MNIST(mnist_root, train=False, download=True, transform=transform)
        train_data = mnist_trainset.data
        train_labels = mnist_trainset.targets
        test_data = mnist_testset.data
        test_labels = mnist_testset.targets
        self.data = torch.cat((train_data, test_data), dim=0)
        self.labels = torch.cat((train_labels, test_labels), dim=0)
        self.list_by_label = []
        pdb.set_trace()

        for i in range(10):
            data_by_label = self.data[self.labels == i]
            self.list_by_label.append(data_by_label)

    def randomize(self, tensor):
        length = len(tensor)
        idx = random.randrange(length)
        return tensor[idx]

    def __getitem__(self, i):
        datapoint = self.data[i]
        label = self.labels[i]
        label_add_one = (label + 1) % 10
        datapoint_add_one = self.randomize(self.list_by_label[label_add_one])
        return (datapoint, label), (datapoint_add_one, label_add_one)


class AugmentedMNIST(Dataset):
    def __init__(self, root_dir, split_type='train'):
        super().__init__()
        self.root_dir = root_dir
        self.split_type = split_type
        self.img_dir = os.path.join(self.root_dir, split_type)
        labels_file = os.path.join(self.root_dir, split_type + '_labels.csv')
        self.labels_file = pd.read_csv(labels_file)

    def __len__(self):
        return len(self.labels.file)
        
    def __getitem__(self, i):
        img_name  = os.path.join(self.root_dir, self.split_type, self.split_type + f'_{i}.png')
        img = np.array(Image.open(img_name))
        label_string = self.labels_file.label.iloc[i]
        labels = list(label_string.strip('[]').split())
        # assert (np.allclose(img[:,:,0], img[:,:,1]))
        # assert (np.allclose(img[:,:,1], img[:,:,2]))
        return (img[:,:,0], int(labels[0]))


class SimpleSSLMNIST(Dataset):
    def __init__(self, root_dir, split_type, n_labeled=None, transform=None, download=False, binarize=True, seed=42):
        """
        Code adapted from Rui Shu's github: https://github.com/RuiShu/bcde
        """
        super().__init__()
        self.root_dir = root_dir
        self.split_type = split_type
        self.n_labeled = n_labeled
        # self.data = MNIST(root=root_dir, train=train, download=False, transform=transform)
        if not os.path.exists(root_dir):
            if not download:
                raise Exception('data directory does not exist')
        self.x, self.y = _load_mnist(root_dir, split_type, download)
        if binarize:
            state = np.random.get_state()
            np.random.seed(seed)
            self.x = np.random.binomial(1, self.x)
            np.random.set_state(state)
        if split_type == 'train' and n_labeled is not None:
            n_classes = 10
            self.xl, self.yl, self.xu, self.yu = self._convert_to_ssl(self.x, self.y, n_labeled, n_classes, seed)
        
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.split_type == 'train' and self.n_labeled is not None:
            xl_idx = random.randrange(len(self.xl))
            xu_idx = random.randrange(len(self.xu))
            yu_idx = random.randrange(len(self.yl))
            return self.xl[xl_idx], self.yl[xl_idx], self.xu[xu_idx], self.yu[yu_idx]
        else:
            return self.x[i], self.y[i], self.x[i], self.y[i]




    @staticmethod
    def _convert_to_ssl(x, y, n_labeled, n_classes, seed):
        """
        :arg x: data feature
        :arg y: data class
        :n_labeled: number of datapoints to keep labels
        :n_classes: number of classes y
        """

        state = np.random.get_state()
        np.random.seed(seed)
        x_label, y_label = [], []
        x_comp, y_comp = [], []

        for i in range(n_classes):
            idx = (y == i)
            x_cand, y_cand = x[idx], y[idx]
            idx = np.random.choice(len(x_cand), n_labeled // n_classes, replace=False)
            x_select, y_select = x_cand[idx], y_cand[idx]
            x_label += [x_select]
            y_label += [y_select]

            x_select, y_select = np.delete(x_cand, idx, 0), np.delete(y_cand, idx, 0)
            x_comp += [x_select]
            y_comp += [y_select]

        x_label = np.concatenate(x_label, axis=0)
        y_label = np.concatenate(y_label, axis=0)

        x_comp = np.concatenate(x_comp, axis=0)
        y_comp = np.concatenate(y_comp, axis=0)

        np.random.set_state(state)
        return x_label, y_label, x_comp, y_comp


################################################################
TOPHALF_TRANSFORM =  transforms.RandomAffine(degrees=[-20, 20],
                                            translate=[0.2, 0.2],
                                            scale=[0.9, 1.1],
                                            # shear=[-0.2, 0.2],
                                            )
################################################################


class RandomTopHalfMnist(Dataset):
    def __init__(self, root_dir, split_type, n_labeled=None, transform=TOPHALF_TRANSFORM, download=False, binarize=True, seed=42,
                static_transform=True,
                ):
        """
        Code adapted from Rui Shu's github: https://github.com/RuiShu/bcde
        """
        super().__init__()
        self.root_dir = root_dir
        self.split_type = split_type
        if not os.path.exists(root_dir):
            if not download:
                raise Exception('data directory does not exist')
        self.x, self.y = _load_mnist(root_dir, split_type, download)
        if binarize:
            state = np.random.get_state()
            np.random.seed(seed)
            self.x = np.random.binomial(1, self.x)
            np.random.set_state(state)

        # always has to transform
        if transform is None:
            transform = TOPHALF_TRANSFORM

        if static_transform:
            self.tops = [] 
            self.bottoms  = []
            for i in range(len(self.x)):
                tophalf = self.x[i][:14, :]
                bottomhalf = self.x[i][14:, :]
                tophalf_img = Image.fromarray(tophalf)
                tophalf_img_tf = transform(tophalf_img)
                self.tops.append(tophalf)
                self.bottoms.append(bottomhalf)
        

    def __getitem__(self, i):
        return self.tops[i], self.bottoms[i], self.y[i]


class StaticRandomTopHalfMnist(Dataset):
    """
    TODO: use split_type to determine datasets. Currently has to specify in root_dir
    """
    def __init__(self,root_dir, split_type='train'):
        self.root_dir = root_dir
        data = np.load(self.root_dir)
        self.tops, self.bottoms, self.labels = data['tops'], data['bottoms'], data['labels']
    
    def __getitem__(self, i):
        return self.tops[i], self.bottoms[i], self.labels[i]

    def __len__(self):
        return len(self.tops)
        
#######################################################
# create a random mnist dataset
from copy import copy
def create_mnist_random_tophalf_transform_dataset(root_dir, seed, split_type='train', transform=TOPHALF_TRANSFORM, binarize=True, download=True,
                                                repeat_each=10,
                                                ):
    """
    :arg repeat_each: repeat 10 random transforms for each image
    """
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    root_dir = root_dir
    split_type = split_type
    if not os.path.exists(root_dir):
        if not download:
            raise Exception('data directory does not exist')
    x, y = _load_mnist(root_dir, split_type, download)
    x = x.reshape(-1, 28, 28)

    tops = np.zeros(shape=(len(x) * repeat_each, 14 ,28), dtype=x.dtype)
    bottoms = np.zeros(shape=(len(x) * repeat_each, 14 ,28), dtype=x.dtype)
    labels = np.zeros(shape=(len(x) * repeat_each), dtype=y.dtype)
    for i in tqdm(range(len(x))):
        tophalf = x[i][:14, :]
        bottomhalf = x[i][14:, :]
        tophalf_img = Image.fromarray(tophalf)
        for j in range(repeat_each):
            tophalf_img_tf = transform(copy(tophalf_img))
            tops[i * repeat_each + j] = np.array(tophalf_img_tf)
            # tops.append(np.array(tophalf_img_tf))
            # bottoms.append(copy(bottomhalf)) 
            bottoms[i * repeat_each + j] = copy(bottomhalf)
            labels[i * repeat_each + j] = copy(y[i])

        # if (i+1) % 100 == 0: print(f'---processed {i+1} images')
 
    permute_idx = np.random.permutation(np.arange(len(x) * repeat_each))
    tops = tops[permute_idx]
    bottoms = bottoms[permute_idx]
    labels = labels[permute_idx]

    # have to binarize here because PIL doesnt read int64
    if binarize:
        # tops = (tops > np.random.uniform(size=tops.shape)).astype(tops.dtype)
        # bottoms = (tops > np.random.uniform(size=bottoms.shape)).astype(bottoms.dtype)
        tops = np.random.binomial(1, tops)
        bottoms = np.random.binomial(1, bottoms)

    np.savez(f'random_tophalf_mnist_{split_type}_large_nobinary.npz', tops=tops, bottoms=bottoms, labels=labels)
    np.random.set_state(state)


def create_train_valid_test():
    root_dir = '.'
    seed = 42
    # print('create valid set---')
    # create_mnist_random_tophalf_transform_dataset(root_dir, seed, split_type='valid', transform=TOPHALF_TRANSFORM, binarize=True, download=False,
    #                                             repeat_each=1,
    #                                             )
    print('create train set---')
    create_mnist_random_tophalf_transform_dataset(root_dir, seed, split_type='train', transform=TOPHALF_TRANSFORM, binarize=False, download=True,
                                                repeat_each=10,
                                                )
    # print('create test set---')
    # create_mnist_random_tophalf_transform_dataset(root_dir, seed, split_type='test', transform=TOPHALF_TRANSFORM, binarize=True, download=False,
    #                                             repeat_each=1,
    #                                             )


###############################
if __name__ == '__main__':
    # data = SimpleSSLMNIST(root_dir='.', split_type='train', n_labeled=20000, download=False)
    # import matplotlib.pyplot as plt
    # (xl, yl, xu, _) = data[0]
    # xl = xl.reshape((28, 28))
    # print(yl)
    # plt.imshow(xl)    
    # plt.savefig('hello.png')

    create_train_valid_test()
    exit(0)