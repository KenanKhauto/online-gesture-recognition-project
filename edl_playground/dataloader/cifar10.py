import torch
from torchvision.datasets import CIFAR10

from .utils import first_n_classes_subset, n_samples_per_class


CLASSES = 10


def get_train_loader(root='./data', transform=None, download=True, kwargs={}, subset_first_n_classes=None, n_samples=None):
    dataset = CIFAR10(root, train=True, download=download, transform=transform)
    classes = CLASSES
    if subset_first_n_classes:
        dataset = first_n_classes_subset(dataset, subset_first_n_classes)
        classes = subset_first_n_classes
    if n_samples:
        dataset = n_samples_per_class(dataset, n_samples, classes)
    dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
    return dataloader


def get_test_loader(root='./data', transform=None, download=False, kwargs={}, subset_first_n_classes=None, n_samples=None):
    dataset = CIFAR10(root, train=False, download=download, transform=transform)
    classes = CLASSES
    if subset_first_n_classes:
        dataset = first_n_classes_subset(dataset, subset_first_n_classes)
        classes = subset_first_n_classes
    if n_samples:
        dataset = n_samples_per_class(dataset, n_samples, classes)
    dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
    return dataloader

