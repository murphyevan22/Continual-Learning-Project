import collections
import glob
import os

import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


def CIFAR_loader(args, num_workers=4, pin_memory=False, normalize=None):
    dl, ds = {}, {}
    # Normalize data sets
    # if per_task_norm:
    #     tsforms = {
    #         "train": transforms.Compose([
    #             transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=mean[task], std=std[task])
    #         ]),
    #         "test": transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=mean[task], std=std[task])
    #         ])
    #     }
    # else:
    tsforms = {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
    }
    ds['train'] = datasets.ImageFolder(args.train_path,
                                       tsforms['train'])
    ds['test'] = datasets.ImageFolder(args.test_path,
                                      tsforms['test'])
    dl['train'] = torch.utils.data.DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True,
                                              num_workers=4)
    dl['test'] = torch.utils.data.DataLoader(ds['test'], batch_size=args.batch_size, shuffle=False,
                                             num_workers=4)
    for loader in dl.keys():
        print("Dataset:", loader, "\tSize:", len(dl[loader].dataset))
    return dl['train'], dl['test']


def train_loader(path, batch_size, num_workers=4, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)


def test_loader(path, batch_size, num_workers=4, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


def test_loader_caffe(path, batch_size, num_workers=4, pin_memory=False):
    """Legacy loader for caffe. Used with models loaded from caffe."""
    # Returns images in 256 x 256 to subtract given mean of same size.
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 Scale((256, 256)),
                                 # transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


def train_loader_cropped(path, batch_size, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 Scale((224, 224)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)


def test_loader_cropped(path, batch_size, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 Scale((224, 224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


# Note: This might not be needed anymore given that this functionality exists in
# the newer PyTorch versions.
class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)
