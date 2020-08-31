"""
Modified by Evan Murphy (DCU)
Original Source: https://github.com/ngailapdi/LWF
"""
import json

from manager import Manager
import torch

# torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
from numpy import random
import os
from utils import *
import sys
import math
from main import evaluate_past_tasks

parser = argparse.ArgumentParser()
parser.add_argument('--outfile', default='LwF_show_forgetting_newMetric.csv', type=str, help='Output file name')
parser.add_argument('--num_classes', default=5, help='Number of new classes', type=int)
parser.add_argument('--init_lr', default=0.01, type=float, help='Init learning rate')
parser.add_argument('--cls_loss_scale', default=10, type=float, help='Scale regular loss values')
parser.add_argument('--dist_loss_scale', default=0, type=float, help='Scale Dist loss values')
parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs')

parser.add_argument('--batch_size', default=256, type=int, help='Mini batch size')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight Decay Value')
parser.add_argument('--num_tasks', default=5, type=int, help='Total number of tasks')
parser.add_argument('--save_path', default="checkpoints/", type=str, help='Path to save checkpoints (.pt)')
parser.add_argument('--lr_decay', default=10, type=int, help='Decay LR by factor')
parser.add_argument('--T', default=2, type=int, help='Trade off value between current and previous task performance')
parser.add_argument('--early_stop', default=False, type=bool, help='Stop training if acc > 95%')
parser.add_argument('--select_best_weights', default=False, type=bool, help='Save the model with the best accuracy '
                                                                            'for the current task')
parser.add_argument('--resnet_cifar', default=True, type=bool, help='Use Resnet18 designed for CIFAR100 ~10% better')


def show_forgetting():
    # Seed Random Processes
    print("Current Device ID:", torch.cuda.current_device())
    print("Current Device: Address", torch.cuda.device(torch.cuda.current_device()))
    print("Current Device: Name", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Device Count:", torch.cuda.device_count())
    print("Cuda Available: ", torch.cuda.is_available())

    seed = 123
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)  # CPU seed
    if device == torch.device("cuda:0"):
        print("Cuda Seeding..")
        torch.cuda.manual_seed_all(seed)  # GPU seed
    random.seed(seed)  # python seed for image transformation
    np.random.seed(seed)

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        print(f"Making path: {args.save_path}")
        os.mkdir(args.save_path)

    num_classes = args.num_classes
    num_tasks = args.num_tasks
    total_classes = num_classes * num_tasks
    prev_dl = None
    with open(args.outfile, 'w') as file:
        for k, v in args.__dict__.items():
            print(f"{k},{v}", file=file, flush=True)
        print("Known,Task,Test Acc,Test Loss,Train Acc,Train Loss,Epoch", file=file)

        manager = Manager(args, device)
        # Show forgetting by not using distilation loss
        manager.use_soft_labels = False

        prev_dls = {}
        for task_id in range(1, num_tasks + 1):
            # Load Datasets
            curr_task = CIFAR100[task_id]
            print(f'Task: {curr_task} ({task_id})')
            train_loader, test_loader = CIFAR_SuperClass_loader(curr_task, args.batch_size)
            dl = {"train": train_loader, "test": test_loader}

            # Expand FC and train net
            train_acc, train_loss, epoch = manager.train(dl, curr_task)
            manager.seen_classes = manager.n_classes
            print("%d, " % manager.seen_classes, file=file, end="")
            print("model classes : %d, " % manager.seen_classes)

            # Evaluate on test
            test_acc, test_loss = manager.evaluate_per_task(test_loader, task_id)
            print(f"{curr_task}, {test_acc}, {test_loss}, {train_acc}, {train_loss}, {epoch}", file=file)
            #
            # # Evaluate on previous task
            # if len(prev_dls) != 0:
            #     print("Evaluating on previous tasks")
            #     for task, data in prev_dls.items():
            #         print(f"Task: {task}")
            #         print(f"Previous Loss: {data['loss']},  Acc: {data['acc']}")
            #         manager.evaluate(data['dl'])

            # Save Best model
            # save_path = args.save_path + f"{curr_task}_{test_acc}_{test_loss}"
            save_path = args.save_path + f"{curr_task}"
            save_model(manager.model, args, test_acc, test_loss, train_acc, train_loss, save_path, task_id, epoch)

            prev_dls[curr_task] = {"dl": dl['test'], "acc": test_acc, "loss": test_loss}

            if math.isnan(train_loss):
                print("Loss is NaN!", file=file)
                sys.exit(0)
        # Now load last saved and eval on all tasks
        json_file = args.outfile.replace(".csv", ".txt")
        with open(json_file, 'w') as f:
            f.write(json.dumps(manager.track_predictions))
            f.close()
        evaluate_past_tasks(file, args, device)

        # # Now load last saved and eval on all tasks
        # del manager
        # print("Evaluating Forgetting", file=file)
        # print("Task,Prev_Acc,Curr_Acc,Prev_Loss,Curr_Loss,Forgetting", file=file)
        # print(f"Using Save Path: {save_path}")
        # avg_prev_acc = 0.0
        # avg_curr_acc = 0.0
        # avg_forgetting = 0.0
        #
        # ckpt = load_checkpoint(save_path + '.pth')
        # manager = Manager(args=ckpt['args'], device=device)
        # manager.model.fc = nn.Linear(512, num_tasks * 5, bias=True)
        #
        # manager.model.load_state_dict(ckpt['model'])
        # manager.model = manager.model.to(manager.device)
        # manager.model = manager.model.eval()
        # # Record Acc for each task
        #
        # for task_id in range(1, num_tasks + 1):
        #     # Record results
        #     print("* " * 50)
        #     task = CIFAR100[task_id]
        #     tmp_ckpt = load_checkpoint(args.save_path + task + '.pth')
        #     prev_acc, prev_loss = tmp_ckpt['test_accuracy'], tmp_ckpt['test_loss']
        #
        #     print(f"Evaluating for task: {task} ({task_id})")
        #     print('Prev Loss: {:.4f}, Accuracy: {:.4f}'.format(prev_loss, prev_acc))
        #
        #     _, test_loader = CIFAR_SuperClass_loader(task, batch_size=500)
        #     # Evaluate on test
        #     test_acc, test_loss = manager.evaluate(test_loader)
        #     forgetting = prev_acc - test_acc
        #     print(f"{task},{prev_acc},{test_acc},{prev_loss},{test_loss},{forgetting}", file=file)
        #
        #     # Update Averages
        #     avg_prev_acc += prev_acc
        #     avg_curr_acc += test_acc
        #     avg_forgetting += forgetting
        #     print("* " * 50)
        # avg_prev_acc = round(avg_prev_acc / num_tasks, 2)
        # avg_curr_acc = round(avg_curr_acc / num_tasks, 2)
        # avg_forgetting = round(avg_forgetting / num_tasks, 2)
        #
        # print(f"Avg_prev_acc,{avg_prev_acc}", file=file)
        # print(f"Avg_curr_acc,{avg_curr_acc}", file=file)
        # print(f"Avg_forgetting,{avg_forgetting}", file=file)


if __name__ == '__main__':
    show_forgetting()
