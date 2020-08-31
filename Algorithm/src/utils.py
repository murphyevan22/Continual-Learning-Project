"""Contains a bunch of utility functions."""

import numpy as np


def step_lr(epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
    """Handles step decay of learning rate."""
    factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
    new_lr = base_lr * factor
    new_lr = round(new_lr, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('Set lr to ', new_lr)
    return optimizer


def step_lr_CIFAR(epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
    """Handles step decay of learning rate."""
    new_lr = base_lr
    if epoch >= 75:
        new_lr = base_lr * lr_decay_factor
        new_lr = round(new_lr, 5)
        if epoch >= 140:
            new_lr *= 0.1
    for group in optimizer.param_groups:
        group['lr'] = new_lr
    #print('Set lr to ', new_lr)

    return optimizer


def record_softlabels():
    # Make copy of model
    # train model
    # record logits of model
    # record logits of prev model
    # add loss terms and get dist loss between orignal logits and updated logits
    return "Method not finished"


