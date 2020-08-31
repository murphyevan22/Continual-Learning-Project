import torch
import torch.nn as nn


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


def main(loadname):
    print("Running CIFAR100 evaluation")
    ckpt = torch.load(loadname)
    model = ckpt['model']
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    previous_masks = ckpt['previous_masks']
    dataset2idx = ckpt['dataset2idx']
    if 'dataset2biases' in ckpt:
        dataset2biases = ckpt['dataset2biases']
    else:
        dataset2biases = {}


        


if __name__ == '__main__':
    name = "/Users/evanm/Documents/College Downloads/Masters Project/Packnet/checkpoints/CIFAR100/vehicles_2/CIFAR100_0.75__pruned_postprune.pt"
    main(loadname=name)
