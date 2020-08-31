"""Main entry point for doing all stuff."""
import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo

import logging
import os
import sys
import pdb
import math
from tqdm import tqdm
import numpy as np

import utils
from utils import Optimizers, set_logger
from utils.manager import Manager
import utils.fine_grained_dataset as dataset
import models
import models.layers as nl


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='resnet50',
                   help='Architectures')
parser.add_argument('--num_classes', type=int, default=-1,
                   help='Num outputs for dataset')

# Optimization options.
parser.add_argument('--lr', type=float, default=0.1,
                   help='Learning rate for parameters, used for baselines')
parser.add_argument('--lr_mask', type=float, default=1e-4,
                   help='Learning rate for mask')
parser.add_argument('--lr_mask_decay_every', type=int,
                   help='Step decay every this many epochs')
parser.add_argument('--batch_size', type=int, default=32,
                   help='input batch size for training')
parser.add_argument('--val_batch_size', type=int, default=100,
                   help='input batch size for validation')
parser.add_argument('--workers', type=int, default=24, help='')
parser.add_argument('--weight_decay', type=float, default=0.0,
                   help='Weight decay')

# Masking options.
parser.add_argument('--mask_init', default='1s',
                   choices=['1s', 'uniform', 'weight_based_1s'],
                   help='Type of mask init')
parser.add_argument('--mask_scale', type=float, default=1e-2,
                   help='Mask initialization scaling')
parser.add_argument('--mask_scale_gradients', type=str, default='none',
                   choices=['none', 'average', 'individual'],
                   help='Scale mask gradients by weights')
parser.add_argument('--threshold_fn',
                   choices=['binarizer', 'ternarizer'],
                   help='Type of thresholding function')
parser.add_argument('--threshold', type=float, default=2e-3, help='')

# Paths.
parser.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
parser.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
parser.add_argument('--val_path', type=str, default='',
                   help='Location of test data')
parser.add_argument('--save_prefix', type=str, default='checkpoints/',
                   help='Location to save model')

# Other.
parser.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--checkpoint_format', type=str,
                    default='./{save_folder}/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--epochs', type=int, default=160,
                    help='number of epochs to train')
parser.add_argument('--restore_epoch', type=int, default=0, help='')
parser.add_argument('--image_size', type=int, default=32, help='')
parser.add_argument('--save_folder', type=str,
                    help='folder name inside one_check folder')
parser.add_argument('--load_folder', default='', help='')
parser.add_argument('--pruning_interval', type=int, default=100, help='')
parser.add_argument('--pruning_frequency', type=int, default=10, help='')
parser.add_argument('--initial_sparsity', type=float, default=0.0, help='')
parser.add_argument('--target_sparsity', type=float, default=0.1, help='')
parser.add_argument('--mode',
                    choices=['finetune', 'prune', 'inference'],
                    help='Run mode')
parser.add_argument('--jsonfile', type=str, help='file to restore baseline validation accuracy')
parser.add_argument('--network_width_multiplier', type=float, default=1.0, help='the multiplier to scale up the channel width')
parser.add_argument('--use_imagenet_pretrained', action='store_true', default=False,
                    help='')
parser.add_argument('--test_piggymask', action='store_true', default=False, help='')
parser.add_argument('--progressive_init', action='store_true', default=False, help='')
parser.add_argument('--finetune_again', action='store_true', default=False, help='')
parser.add_argument('--pruning_ratio_to_acc_record_file', type=str, help='')
parser.add_argument('--log_path', type=str, help='')


def main():
    """Do stuff."""
    args = parser.parse_args()

    # args.batch_size = args.batch_size * torch.cuda.device_count()
    args.network_width_multiplier = math.sqrt(args.network_width_multiplier)

    if args.mode == 'prune':
        args.save_folder = os.path.join(args.save_folder, str(args.target_sparsity))
        if args.initial_sparsity != 0.0:
            args.load_folder = os.path.join(args.load_folder, str(args.initial_sparsity))

    if args.pruning_ratio_to_acc_record_file and not os.path.isdir(args.pruning_ratio_to_acc_record_file.rsplit('/', 1)[0]):
        os.makedirs(args.pruning_ratio_to_acc_record_file.rsplit('/', 1)[0])

    if args.save_folder and not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.log_path:
        set_logger(args.log_path)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        args.cuda = False

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    resume_folder = args.load_folder
    for try_epoch in range(200, 0, -1):
        if os.path.exists(args.checkpoint_format.format(
            save_folder=resume_folder, epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    if args.restore_epoch:
        resume_from_epoch = args.restore_epoch

    # Set default train and test path if not provided as input.
    utils.set_dataset_paths(args)

    if resume_from_epoch:
        filepath = args.checkpoint_format.format(save_folder=resume_folder, epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        checkpoint_keys = checkpoint.keys()
        dataset_history = checkpoint['dataset_history']
        dataset2num_classes = checkpoint['dataset2num_classes']
        masks = checkpoint['masks']
        shared_layer_info = checkpoint['shared_layer_info']
        if 'num_for_construct' in checkpoint_keys:
            num_for_construct = checkpoint['num_for_construct']
        if args.mode == 'inference' and 'network_width_multiplier' in shared_layer_info[args.dataset]:
            args.network_width_multiplier = shared_layer_info[args.dataset]['network_width_multiplier']
    else:
        dataset_history = []
        dataset2num_classes = {}
        masks = {}
        shared_layer_info = {}

    if args.arch == 'resnet50':
        # num_for_construct = [64, 64, 64*4, 128, 128*4, 256, 256*4, 512, 512*4]
        model = models.__dict__[args.arch](dataset_history=dataset_history, dataset2num_classes=dataset2num_classes,
            network_width_multiplier=args.network_width_multiplier, shared_layer_info=shared_layer_info)
    elif 'vgg' in args.arch:
        custom_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        model = models.__dict__[args.arch](custom_cfg, dataset_history=dataset_history, dataset2num_classes=dataset2num_classes,
            network_width_multiplier=args.network_width_multiplier, shared_layer_info=shared_layer_info, progressive_init=args.progressive_init)
    else:
        print('Error!')
        sys.exit(1)

    # Add and set the model dataset.
    model.add_dataset(args.dataset, args.num_classes)
    model.set_dataset(args.dataset)

    # Move model to GPU
    model = nn.DataParallel(model)
    model = model.cuda()

    # For datasets whose image_size is 224 and also the first task
    if args.use_imagenet_pretrained and model.module.datasets.index(args.dataset) == 0:
        curr_model_state_dict = model.state_dict()
        if args.arch == 'custom_vgg':
            state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
            for name, param in state_dict.items():
                if 'classifier' not in name:
                    curr_model_state_dict['module.' + name].copy_(param)
            curr_model_state_dict['module.features.45.weight'].copy_(state_dict['classifier.0.weight'])
            curr_model_state_dict['module.features.45.bias'].copy_(state_dict['classifier.0.bias'])
            curr_model_state_dict['module.features.48.weight'].copy_(state_dict['classifier.3.weight'])
            curr_model_state_dict['module.features.48.bias'].copy_(state_dict['classifier.3.bias'])
            if args.dataset == 'imagenet':
                curr_model_state_dict['module.classifiers.0.weight'].copy_(state_dict['classifier.6.weight'])
                curr_model_state_dict['module.classifiers.0.bias'].copy_(state_dict['classifier.6.bias'])
        elif args.arch == 'resnet50':
            state_dict = model_zoo.load_url(model_urls['resnet50'])
            for name, param in state_dict.items():
                if 'fc' not in name:
                    curr_model_state_dict['module.' + name].copy_(param)
            if args.dataset == 'imagenet':
                curr_model_state_dict['module.classifiers.0.weight'].copy_(state_dict['fc.weight'])
                curr_model_state_dict['module.classifiers.0.bias'].copy_(state_dict['fc.bias'])
        else:
            print("Currently, we didn't define the mapping of {} between imagenet pretrained weight and our model".format(args.arch))
            sys.exit(5)

    if not masks:
        for name, module in model.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                mask = mask.cuda()
                masks[name] = mask
    else:
        # when we expand network, we need to allocate new masks
        NEED_ADJUST_MASK = False
        for name, module in model.named_modules():
            if isinstance(module, nl.SharableConv2d):
                if masks[name].size(1) < module.weight.data.size(1):
                    assert args.mode == 'finetune'
                    NEED_ADJUST_MASK = True
                elif masks[name].size(1) > module.weight.data.size(1):
                    assert args.mode == 'inference'
                    NEED_ADJUST_MASK = True


        if NEED_ADJUST_MASK:
            if args.mode == 'finetune':
                for name, module in model.named_modules():
                    if isinstance(module, nl.SharableConv2d):
                        mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                        mask = mask.cuda()
                        mask[:masks[name].size(0), :masks[name].size(1), :, :].copy_(masks[name])
                        masks[name] = mask
                    elif isinstance(module, nl.SharableLinear):
                        mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                        mask = mask.cuda()
                        mask[:masks[name].size(0), :masks[name].size(1)].copy_(masks[name])
                        masks[name] = mask
            elif args.mode == 'inference':
                for name, module in model.named_modules():
                    if isinstance(module, nl.SharableConv2d):
                        mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                        mask = mask.cuda()
                        mask[:, :, :, :].copy_(masks[name][:mask.size(0), :mask.size(1), :, :])
                        masks[name] = mask
                    elif isinstance(module, nl.SharableLinear):
                        mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                        mask = mask.cuda()
                        mask[:, :].copy_(masks[name][:mask.size(0), :mask.size(1)])
                        masks[name] = mask

    if args.dataset not in shared_layer_info:

        shared_layer_info[args.dataset] = {
            'bias': {},
            'bn_layer_running_mean': {},
            'bn_layer_running_var': {},
            'bn_layer_weight': {},
            'bn_layer_bias': {},
            'piggymask': {}
        }

        piggymasks = {}
        task_id = model.module.datasets.index(args.dataset) + 1
        if task_id > 1:
            for name, module in model.module.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    piggymasks[name] = torch.zeros_like(masks['module.' + name], dtype=torch.float32)
                    piggymasks[name].fill_(0.01)
                    piggymasks[name] = Parameter(piggymasks[name])
                    module.piggymask = piggymasks[name]
    else:
        piggymasks = shared_layer_info[args.dataset]['piggymask']
        task_id = model.module.datasets.index(args.dataset) + 1
        if task_id > 1:
            for name, module in model.module.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    module.piggymask = piggymasks[name]

    shared_layer_info[args.dataset]['network_width_multiplier'] = args.network_width_multiplier

    if 'cropped' in args.dataset:
        train_loader = dataset.train_loader_cropped(args.train_path, args.batch_size)
        val_loader = dataset.val_loader_cropped(args.val_path, args.val_batch_size)
    else:
        train_loader = dataset.train_loader(args.train_path, args.batch_size)
        val_loader = dataset.val_loader(args.val_path, args.val_batch_size)

    # if we are going to save checkpoint in other folder, then we recalculate the starting epoch
    if args.save_folder != args.load_folder:
        start_epoch = 0
    else:
        start_epoch = resume_from_epoch

    curr_prune_step = begin_prune_step = start_epoch * len(train_loader)
    end_prune_step = curr_prune_step + args.pruning_interval * len(train_loader)

    manager = Manager(args, model, shared_layer_info, masks, train_loader, val_loader, begin_prune_step, end_prune_step)

    if args.mode == 'inference':
        manager.load_checkpoint_only_for_evaluate(resume_from_epoch, resume_folder)
        manager.validate(resume_from_epoch-1)
        return

    lr = args.lr
    lr_mask = args.lr_mask
    # update all layers
    named_params = dict(model.named_parameters())
    params_to_optimize_via_SGD = []
    named_of_params_to_optimize_via_SGD = []
    masks_to_optimize_via_Adam = []
    named_of_masks_to_optimize_via_Adam = []

    for name, param in named_params.items():
        if 'classifiers' in name:
            if '.{}.'.format(model.module.datasets.index(args.dataset)) in name:
                params_to_optimize_via_SGD.append(param)
                named_of_params_to_optimize_via_SGD.append(name)
            continue
        elif 'piggymask' in name:
            masks_to_optimize_via_Adam.append(param)
            named_of_masks_to_optimize_via_Adam.append(name)
        else:
            params_to_optimize_via_SGD.append(param)
            named_of_params_to_optimize_via_SGD.append(name)

    optimizer_network = optim.SGD(params_to_optimize_via_SGD, lr=lr,
                          weight_decay=0.0, momentum=0.9, nesterov=True)
    optimizers = Optimizers()
    optimizers.add(optimizer_network, lr)

    if masks_to_optimize_via_Adam:
        optimizer_mask = optim.Adam(masks_to_optimize_via_Adam, lr=lr_mask)
        optimizers.add(optimizer_mask, lr_mask)

    manager.load_checkpoint(optimizers, resume_from_epoch, resume_folder)

    """Performs training."""
    curr_lrs = []
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            curr_lrs.append(param_group['lr'])
            break

    if args.jsonfile is None or not os.path.isfile(args.jsonfile):
        sys.exit(3)
    with open(args.jsonfile, 'r') as jsonfile:
        json_data = json.load(jsonfile)
        baseline_acc = float(json_data[args.dataset])

    if args.mode == 'prune':
        if args.dataset != 'imagenet':
            history_best_avg_val_acc_when_prune = 0.0
            #history_best_avg_val_acc_when_prune = baseline_acc - 0.005
        else:
            if 'vgg' in args.arch:
                baseline_acc = 0.7336
                history_best_avg_val_acc_when_prune = baseline_acc - 0.005
            elif 'resnet50' in args.arch:
                baseline_acc = 0.7616
                history_best_avg_val_acc_when_prune = baseline_acc - 0.005
            else:
                print('Something is wrong')
                exit(1)

        stop_prune = True

        if 'gradual_prune' in args.load_folder and args.save_folder == args.load_folder:
            if args.dataset == 'imagenet':
                args.epochs = 10 + resume_from_epoch
            else:
                args.epochs = 20 + resume_from_epoch
        logging.info('')
        logging.info('Before pruning: ')
        logging.info('Sparsity range: {} -> {}'.format(args.initial_sparsity, args.target_sparsity))
        manager.validate(start_epoch-1)
        logging.info('')

    elif args.mode == 'finetune':
        manager.pruner.make_finetuning_mask()

        if args.dataset == 'imagenet':
            manager.validate(0)
            manager.save_checkpoint(optimizers, 0, args.save_folder)
            return

        history_best_avg_val_acc = 0.0
        num_epochs_that_criterion_does_not_get_better = 0
        times_of_decaying_learning_rate = 0

    for epoch_idx in range(start_epoch, args.epochs):
        avg_train_acc, curr_prune_step = manager.train(optimizers, epoch_idx, curr_lrs, curr_prune_step)
        avg_val_acc = manager.validate(epoch_idx)

        if args.mode == 'prune' and (epoch_idx+1) >= (args.pruning_interval + start_epoch) and (
            avg_val_acc > history_best_avg_val_acc_when_prune):
            stop_prune = False
            history_best_avg_val_acc_when_prune = avg_val_acc
            if args.save_folder is not None:
                paths = os.listdir(args.save_folder)
                if paths and '.pth.tar' in paths[0]:
                    for checkpoint_file in paths:
                        os.remove(os.path.join(args.save_folder, checkpoint_file))
            else:
                print('Something is wrong! Block the program with pdb')
                pdb.set_trace()

            manager.save_checkpoint(optimizers, epoch_idx, args.save_folder)

        if args.mode == 'finetune':

            if avg_val_acc > history_best_avg_val_acc:
                num_epochs_that_criterion_does_not_get_better = 0
                if args.save_folder is not None:
                    paths = os.listdir(args.save_folder)
                    if paths and '.pth.tar' in paths[0]:
                        for checkpoint_file in paths:
                            os.remove(os.path.join(args.save_folder, checkpoint_file))
                else:
                    print('Something is wrong! Block the program with pdb')
                    pdb.set_trace()

                history_best_avg_val_acc = avg_val_acc
                manager.save_checkpoint(optimizers, epoch_idx, args.save_folder)
            else:
                num_epochs_that_criterion_does_not_get_better += 1

            if times_of_decaying_learning_rate >=3:
                print()
                print("times_of_decaying_learning_rate reach {}, stop training".format(
                        times_of_decaying_learning_rate))
                break

            if num_epochs_that_criterion_does_not_get_better >= 5:
                times_of_decaying_learning_rate += 1
                num_epochs_that_criterion_does_not_get_better = 0
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] *= 0.1
                curr_lrs[0] = param_group['lr']
                print()
                print("continously {} epochs doesn't get higher acc, "
                      "decay learning rate by multiplying 0.1".format(
                        num_epochs_that_criterion_does_not_get_better))

                if times_of_decaying_learning_rate == 1 and len(optimizers.lrs) == 2:
                   for param_group in optimizers[1].param_groups:
                       param_group['lr'] *= 0.2
                   curr_lrs[1] = param_group['lr']


    print('-' * 16)

    if args.pruning_ratio_to_acc_record_file:
        json_data = {}
        if os.path.isfile(args.pruning_ratio_to_acc_record_file):
            with open(args.pruning_ratio_to_acc_record_file, 'r') as json_file:
                json_data = json.load(json_file)

    if args.mode == 'finetune' and not args.test_piggymask:
        if args.pruning_ratio_to_acc_record_file:
            json_data[0.0] = round(history_best_avg_val_acc, 4)
            with open(args.pruning_ratio_to_acc_record_file, 'w') as json_file:
                json.dump(json_data, json_file)

        if history_best_avg_val_acc - baseline_acc > -0.005: # TODO
            #json_data = {}
            #json_data['acc_before_prune'] = '{:.4f}'.format(history_best_avg_val_acc)
            #with open(args.tmp_benchmark_file, 'w') as jsonfile:
            #    json.dump(json_data, jsonfile)
            pass
        else:
            print("It's time to expand the Network")
            print('Auto expand network')
            sys.exit(2)

        if manager.pruner.calculate_curr_task_ratio() == 0.0:
            print('There is no left space in convolutional layer for curr task, so needless to prune')
            sys.exit(5)

    elif args.mode == 'prune':
#        if stop_prune:
#            print('Acc too low, stop pruning.')
#            sys.exit(4)
        if args.pruning_ratio_to_acc_record_file:
            json_data[args.target_sparsity] = round(history_best_avg_val_acc_when_prune, 4)
            with open(args.pruning_ratio_to_acc_record_file, 'w') as json_file:
                json.dump(json_data, json_file)


if __name__ == '__main__':
    main()
