"""
Modified by Evan Murphy (DCU)
Original Source: https://github.com/ngailapdi/LWF
"""

import torch
import ResNet_CIFAR100

# torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import copy
import torchvision.models as models
from utils import *


class Manager:
    def __init__(self, args, device):
        self.args = args
        # Hyper Parameters
        self.init_lr = args.init_lr
        self.lr = self.init_lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.lr_decay = args.lr_decay
        self.device = device
        self.cls_loss_scale = args.cls_loss_scale
        self.dist_loss_scale = args.dist_loss_scale
        self.T = args.T
        self.momentum = 0.9
        self.weight_decay = args.weight_decay

        self.pretrained = False
        self.use_soft_labels = True
        self.first_task = True
        self.track_predictions = {}

        # Learning rate decay intervals
        # self.lower_rate_epoch = [int(0.4 * self.num_epochs), int(0.7 * self.num_epochs)]
        self.lower_rate_epoch = [40]

        # Network architecture
        if args.resnet_cifar:
            # ResNet18 for CIFAR100
            self.model = ResNet_CIFAR100.resnet18(args.num_classes)
        else:
            # ResNet
            # self.model = models.resnet34(pretrained=self.pretrained)
            self.model = models.resnet18(pretrained=self.pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, args.num_classes)
            self.model.apply(kaiming_normal_init)

        self.model = self.model.to(device)
        # self.model.apply(set_bn_eval)

        # for m in self.model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         #print("Freezing BN layer..")
        #         m.eval()
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False
        # for module in self.model.modules():
        #     print(module)
        # for name, module in self.model.named_children():
        #     print(f"Name: {name}, Module: {module}")
        #     if "bn" in name:
        #         print(f"Name: {name}, Module: {module}")
        #         module.eps = self.epsilon
        #         print(f"Name: {name}, Module: {module}")
        #     elif "layer" in name:
        #         for n,m in module.named_children():
        #             print(f"Name: {n}, Module: {m}")
        #             if "bn" in n:
        #                 print(f"Name: {n}, Module: {m}")
        #                 m.eps = self.epsilon
        #                 print(f"Name: {n}, Module: {m}")

        # VGG
        # self.model = models.vgg16(pretrained=False)
        # self.model.fc = self.model.classifier
        # num_features = self.model.classifier[-1].in_features
        # self.model.fc = nn.Linear(num_features, args.num_classes)

        # n_classes is incremented before processing new data in an iteration
        # seen_classes is set to n_classes after all data for an iteration has been processed
        self.n_classes = 0
        self.seen_classes = 0

    def increment_classes(self, num_classes):
        """Add number of new classes in the classifier layer"""
        # Record number of input and output features for existing network
        in_features = self.model.fc.in_features
        out_features = self.model.fc.out_features
        # Record weight and bias values
        weight = self.model.fc.weight.data
        bias = self.model.fc.bias.data

        # If first task
        if self.seen_classes == 0:
            new_out_features = num_classes
        else:
            # Extend fc layer by number of new classes
            new_out_features = out_features + num_classes

        # Create new fc layer for the network
        self.model.fc = nn.Linear(in_features, new_out_features)

        # Init as per LwF paper
        kaiming_normal_init(self.model.fc.weight)
        # Reassign recorded weight and bias values to new layer
        self.model.fc.weight.data[:out_features] = weight
        self.model.fc.bias.data[:out_features] = bias
        # Increment number of classes
        self.n_classes += num_classes

    def train(self, data_loaders, task, new_classes=True):

        self.track_predictions[task] = {}
        prev_model = None
        if self.seen_classes > 0:
            # prev_weights = copy.deepcopy(self.model.state_dict())
            # prev_model = models.resnet18()
            # prev_model.fc = nn.Linear(512, self.seen_classes)
            # prev_model.load_state_dict(prev_weights)
            # Save a copy to compute distillation outputs
            prev_model = copy.deepcopy(self.model)
            # Don't need gradients for previous model
            prev_model.zero_grad()
            prev_model = prev_model.to(self.device)
            prev_model = prev_model.eval()

        best_acc = 0.0
        best_ep = self.num_epochs

        print('Classes seen: ', self.seen_classes)
        if new_classes:
            self.increment_classes(len(data_loaders['train'].dataset.classes))
            self.model.to(self.device)
        # Set model to train mode
        # self.model.train()

        # self.model.apply(set_bn_eval)
        # Reset LR and create optimizer
        self.lr = self.init_lr
        optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)

        with tqdm(total=self.num_epochs) as pbar:
            for epoch in range(1, self.num_epochs + 1):

                # Modify learning rate
                if epoch in self.lower_rate_epoch:
                    # print(f"Epoch: {epoch}")
                    print(f"Updating LR {self.lr} --> ", end="")
                    # if not hasattr(self, 'lr'):
                    #     self.lr = self.init_lr
                    self.lr = self.lr * (1 / self.lr_decay)
                    self.lr = round(self.lr, 5)
                    print(f"{self.lr}")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.lr

                train_acc, train_loss = self.train_epoch(epoch, optimizer, data_loaders, prev_model, task)
                #test_acc, _ = self.evaluate(data_loaders['test'])
                test_acc, _ = self.evaluate_per_task(data_loaders['test'], CIFAR100.index(task))
                # Save best performance model
                # This adds bias towards new task perf
                if self.args.select_best_weights and best_acc < test_acc:
                    best_acc = test_acc
                    best_wts = copy.deepcopy(self.model.state_dict())
                    best_ep = epoch

                if self.args.early_stop and train_acc > 90:
                    best_ep = epoch
                    print(f"Early Termination (Ep:{epoch}), Accuracy: {train_acc}")
                    break
                    # return train_acc, train_loss, test_acc,epoch

                pbar.update(1)
        if self.args.select_best_weights:
            self.model.load_state_dict(best_wts)
        # From now on include scaling on cls loss
        self.first_task = False
        return train_acc, train_loss, best_ep

    def evaluate(self, loader):
        print("Don't want to use this evaluation metric\nUse evaluate_per_task() instead")
        with torch.no_grad():
            self.model.eval()
            test_loss = 0.0
            correct = 0.0

            for (images, labels) in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                test_loss += loss.item()
                _, preds = outputs.max(1)
                # _, preds_softmax = torch.max(torch.softmax(self.model(images), dim=1), dim=1, keepdim=False)

                correct += preds.eq(labels).sum()

                # print("Labels:\t\t", labels.tolist())
                # print("Preds:\t\t", preds.tolist())

                # if preds.tolist() != preds_softmax.tolist():
                #     print("Predictions not the same!!")

            test_acc = correct.item() / (len(loader.dataset))
            test_loss = (test_loss / (len(loader.dataset)))
            test_acc = round(test_acc * 100, 2)
            test_loss = round(test_loss, 5)
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))
            print()

            # if hasattr(self, 'metrics'):
            #     self.metrics[-1]['Test_Acc'] = round(test_acc.item() * 100, 2)
            #     self.metrics[-1]['Test_Loss'] = test_loss
        return test_acc, test_loss

    def train_epoch(self, epoch, optimizer, data_loaders, prev_model, task):
        self.model.train()
        train_loss = 0.0
        correct = 0.0
        total_preds = []

        for batch_index, (images, labels) in enumerate(data_loaders['train']):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)

            # if self.first_task:
            #     # Dont scale cls loss for first task
            #     cls_loss = calculate_loss(outputs, labels, "CrossEntropy", self.device, cls_scale=1)
            # else:
            cls_loss = calculate_loss(outputs, labels, "CrossEntropy", self.device, cls_scale=self.cls_loss_scale)
            if self.n_classes // len(data_loaders['train'].dataset.classes) > 1 and self.use_soft_labels:
                with torch.no_grad():
                    prev_logits = prev_model(images)
                # Mention removing extra logits instead of perhaps zeroing them
                # labels_one_hot = F.one_hot(labels)
                logits_dist = outputs[:, :-(self.n_classes - self.seen_classes)]

                dist_loss = calculate_loss(logits_dist, prev_logits, "Distillation",
                                           self.device, T=self.T, dist_scale=self.dist_loss_scale)
                print(f"Losses: CLS: {round(cls_loss.item(), 3)}, Dist: {round(dist_loss.item(), 3)}")
                # print("Scaling...")
                # dist_loss = dist_loss * self.loss_scale
                # loss = loss * self.loss_alpha
                # loss_total = (self.loss_alpha * loss) + (self.loss_scale * dist_loss)
                loss_total = cls_loss + dist_loss

            else:
                loss_total = cls_loss

            # loss_total.sum().backward()
            loss_total.backward()
            optimizer.step()

            # train_loss += loss_total
            train_loss += loss_total.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
            total_preds.extend(preds.tolist())

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.4f}'.format(
                loss_total.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * self.batch_size + len(images),
                total_samples=int(len(data_loaders['train'].dataset))
            ))

        # Record Metric
        train_acc = correct.float() / (len(data_loaders['train'].dataset))
        train_loss = train_loss / (len(data_loaders['train'].dataset))
        train_acc = round(train_acc.item() * 100, 2)
        train_loss = round(train_loss, 5)
        self.track_predictions[task][epoch] = record_preds(total_preds)
        print(f"Average Training Loss: {train_loss}, Acc: {train_acc}")
        if hasattr(self, 'metrics'):
            self.metrics.append(
                {"Epoch": epoch, "Train_Acc": round(train_acc.item() * 100, 2), "Train_Loss": train_loss})

        return train_acc, train_loss

    def evaluate_per_task(self, loader, eval_task_id):
        with torch.no_grad():
            self.model.eval()
            test_loss = 0.0
            correct = 0.0

            for (images, labels) in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                test_loss += loss.item()
                start = (eval_task_id - 1) * 5
                end = eval_task_id * 5
                task_outputs = []
                for out in list(outputs):
                    # print("*" * 20)
                    # print(out[0:5])
                    # outputs.
                    out = out[start:end]
                    task_outputs.append(out)
                    # out = torch.cat([out[:0], out[:5]])
                    # print(out)
                task_outputs = torch.stack(task_outputs)
                # _, preds = outputs.max(1)
                _, preds = task_outputs.max(1)

                # _, preds_softmax = torch.max(torch.softmax(self.model(images), dim=1), dim=1, keepdim=False)
                # Subtract offset from labels to account for shift in logits
                labels = labels.add(-start)
                correct += preds.eq(labels).sum()

                # print("Labels:\t\t", labels.tolist())
                # print("Preds:\t\t", preds.tolist())

                # if preds.tolist() != preds_softmax.tolist():
                #     print("Predictions not the same!!")

            test_acc = correct.item() / (len(loader.dataset))
            test_loss = (test_loss / (len(loader.dataset)))
            test_acc = round(test_acc * 100, 2)
            test_loss = round(test_loss, 5)
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))
            print()

        return test_acc, test_loss

    def scale_grads(self, scale=0.01):
        """Sets grads of fixed weights to 0."""
        assert self.model.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]

                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(
                        self.current_dataset_idx)] = 0
                    if not self.train_bias:
                        # Biases are fixed.
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)
            elif 'BatchNorm' in str(type(module)):
                # Set grads of batchnorm params to 0.
                # if not self.train_bn:
                module.weight.grad.data.fill_(0)
                module.bias.grad.data.fill_(0)
