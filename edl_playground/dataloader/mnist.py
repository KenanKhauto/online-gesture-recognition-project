import torch
from torchvision.datasets import MNIST

def get_train_loader(root='./data', transform=None, download=True, kwargs={}):
    dataset = MNIST(root, train=True, download=download, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
    return dataloader

def get_test_loader(root='./data', transform=None, download=False, kwargs={}):
    dataset = MNIST(root, train=False, download=download, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
    return dataloader

