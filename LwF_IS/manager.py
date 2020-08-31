"""
Modified by Evan Murphy (DCU)
Original Source: https://github.com/ngailapdi/LWF
"""

import torch

# torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import copy
import torchvision.models as models
import ResNet_CIFAR100
from utils import *
import torch


class Manager:
    def __init__(self, args, device):
        self.curr_task = None
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
        self.samples_dir = args.samples_dir
        self.data_dir = '../Data/cifar100_org/'
        self.samples_records_dir = "ImportantSamplesRecords/FINAL_IMS_SAMPLES_RN_CIFAR/"
        self.test_data_loaders = {}
        self.retain_percent = args.retain_percent
        self.random_samples = args.random_samples
        self.samples = None

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

        # n_classes is incremented before processing new data in an iteration
        # seen_classes is set to n_classes after all data for an iteration has been processed
        self.n_classes = 0
        self.seen_classes = 0
        # Dict for forgetting events
        self.forget_events = {task: {cls: {} for cls in task_to_classes[task]} for task in task_to_classes.keys()}
        self.prev_correct = {}

    def increment_classes(self, num_classes):
        """Add number of new classes in the classifier layer"""
        print('New classes: ', num_classes)
        in_features = self.model.fc.in_features
        out_features = self.model.fc.out_features
        weight = self.model.fc.weight.data
        bias = self.model.fc.bias.data

        if self.seen_classes == 0:
            new_out_features = num_classes
        else:
            new_out_features = out_features + num_classes
        print('New out features: ', new_out_features)

        self.model.fc = nn.Linear(in_features, new_out_features)

        kaiming_normal_init(self.model.fc.weight)
        self.model.fc.weight.data[:out_features] = weight
        self.model.fc.bias.data[:out_features] = bias
        self.n_classes += num_classes

    def train(self, data_loaders, task, new_classes=True):

        # Save a copy to compute distillation outputs
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
            # self.increment_classes(self.args.num_classes)
            self.increment_classes(len(data_loaders['train'].dataset.classes))
            self.model.to(self.device)
        # Set model to train mode
        self.model.train()
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

                train_acc, train_loss = self.train_epoch(epoch, optimizer, data_loaders, prev_model)
                test_acc, _ = self.evaluate_per_task(data_loaders['test'], CIFAR100.index(task))

                # Now retrain using retained samples

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

    def evaluate_per_task(self, loader, eval_task_id):
        with torch.no_grad():
            self.model.eval()
            test_loss = 0.0
            correct = 0.0

            for (images, labels, _) in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
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

            test_acc = correct.item() / (len(loader.dataset))
            test_loss = (test_loss / (len(loader.dataset)))
            test_acc = round(test_acc * 100, 2)
            test_loss = round(test_loss, 5)
            print('Test set: Average loss: {:.5f}, Accuracy: {:.4f}'.format(test_loss, test_acc))
            print()

            # self.model.train()

            # if hasattr(self, 'metrics'):
            #     self.metrics[-1]['Test_Acc'] = round(test_acc.item() * 100, 2)
            #     self.metrics[-1]['Test_Loss'] = test_loss
        return test_acc, test_loss

    # def evaluate_per_task(self, loader, eval_task_id):
    #     with torch.no_grad():
    #         self.model.eval()
    #         test_loss = 0.0
    #         correct = 0.0
    #
    #         for (images, labels) in loader:
    #             images = images.to(self.device)
    #             labels = labels.to(self.device)
    #
    #             outputs = self.model(images)
    #             loss = nn.CrossEntropyLoss()(outputs, labels)
    #             test_loss += loss.item()
    #             start = (eval_task_id - 1) * 5
    #             end = eval_task_id * 5
    #             task_outputs = []
    #             for out in list(outputs):
    #                 # print("*" * 20)
    #                 # print(out[0:5])
    #                 # outputs.
    #                 out = out[start:end]
    #                 task_outputs.append(out)
    #                 # out = torch.cat([out[:0], out[:5]])
    #                 # print(out)
    #             task_outputs = torch.stack(task_outputs)
    #             # _, preds = outputs.max(1)
    #             _, preds = task_outputs.max(1)
    #
    #             # _, preds_softmax = torch.max(torch.softmax(self.model(images), dim=1), dim=1, keepdim=False)
    #             # Subtract offset from labels to account for shift in logits
    #             labels = labels.add(-start)
    #             correct += preds.eq(labels).sum()
    #
    #             # print("Labels:\t\t", labels.tolist())
    #             # print("Preds:\t\t", preds.tolist())
    #
    #             # if preds.tolist() != preds_softmax.tolist():
    #             #     print("Predictions not the same!!")
    #
    #         test_acc = correct.item() / (len(loader.dataset))
    #         test_loss = (test_loss / (len(loader.dataset)))
    #         test_acc = round(test_acc * 100, 2)
    #         test_loss = round(test_loss, 5)
    #         print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))
    #         print()
    #
    #     return test_acc, test_loss

    def train_epoch(self, epoch, optimizer, data_loaders, prev_model=None):
        self.model.train()
        train_loss = 0.0
        correct = 0.0
        prev_correct_record = self.prev_correct
        self.prev_correct = []
        if not self.first_task:
            print(f"Training data: {len(data_loaders['train'].dataset)}")
            print(f"Retain data: {len(data_loaders['retain'].dataset)}")
            all_datasets = [data_loaders['train'].dataset, data_loaders['retain'].dataset]
            concat_data = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(all_datasets),
                                                      batch_size=self.batch_size, shuffle=True, num_workers=4)

            # concat_data = zip(data_loaders['train'], data_loaders['retain'])
            print(f"Total data: {len(concat_data.dataset)}")
            # print(f"Total classes: {len(concat_data.dataset.classes)}")
        else:
            if os.path.exists(self.samples_dir[:-1]):
                print("Removing existing samples...")
                shutil.rmtree(self.samples_dir[:-1])
            # print(f"Making path: {self.samples_dir[:-1]}")
            # os.mkdir(self.samples_dir[:-1])
            concat_data = data_loaders['train']
            # dataset = torch.utils.data.TensorDataset(data_loaders['train'], data_loaders['train'])
            # concat_data = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
        # if not self.first_task:
        # for batch_index, (images, labels), (images2, labels2) in enumerate(concat_data):
        # else:
        for batch_index, (images, labels, fnames) in enumerate(concat_data):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)

            #if self.use_soft_labels:
            cls_loss = calculate_loss(outputs, labels, "CrossEntropy", self.device, cls_scale=self.cls_loss_scale)
            #if self.n_classes // len(data_loaders['train'].dataset.classes) > 1 and self.use_soft_labels:
            if not self.first_task and self.use_soft_labels:
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
           # else:
            #loss_total = calculate_loss(outputs, labels, "CrossEntropy", self.device)

            # Perform BackProp
            loss_total.backward()
            optimizer.step()

            train_loss += loss_total.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

            for fname, label, pred in zip(fnames, labels, preds):
                if idx2class[label.item()] in task_to_classes[self.curr_task]:
                    # Check has been misclassified and was classified correctly last time
                    if label != pred and fname in prev_correct_record:
                        if fname in self.forget_events[self.curr_task][idx2class[label.item()]]:
                            self.forget_events[self.curr_task][idx2class[label.item()]][fname] += 1
                        else:
                            self.forget_events[self.curr_task][idx2class[label.item()]][fname] = 1
                    elif label == pred:
                        self.prev_correct.append(fname)

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.4f}'.format(
                loss_total.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * self.batch_size + len(images),
                total_samples=int(len(concat_data.dataset))
            ))

        # Record Metric
        train_acc = correct.float() / (len(concat_data.dataset))
        train_loss = train_loss / (len(concat_data.dataset))
        train_acc = round(train_acc.item() * 100, 2)
        train_loss = round(train_loss, 5)

        print(f"Average Training Loss: {train_loss}, Acc: {train_acc}")
        if hasattr(self, 'metrics'):
            self.metrics.append(
                {"Epoch": epoch, "Train_Acc": round(train_acc.item() * 100, 2), "Train_Loss": train_loss})

        return train_acc, train_loss

    def train_epoch_old(self, epoch, optimizer, data_loaders, prev_model=None):
        self.model.train()
        train_loss = 0.0
        correct = 0.0
        if not self.first_task:
            print(f"Training data: {len(data_loaders['train'].dataset)}")
            print(f"Retain data: {len(data_loaders['retain'].dataset)}")
            all_datasets = [data_loaders['train'].dataset, data_loaders['retain'].dataset]
            concat_data = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(all_datasets),
                                                      batch_size=self.batch_size, shuffle=True, num_workers=4)

            print(f"Total data: {len(concat_data.dataset)}")
            # print(f"Total classes: {len(concat_data.dataset.classes)}")
        else:
            if os.path.exists(self.samples_dir[:-1]):
                print("Removing existing samples...")
                shutil.rmtree(self.samples_dir[:-1])
            # print(f"Making path: {self.samples_dir[:-1]}")
            # os.mkdir(self.samples_dir[:-1])
            concat_data = data_loaders['train']
            # dataset = torch.utils.data.TensorDataset(data_loaders['train'], data_loaders['train'])
            # concat_data = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
        # if not self.first_task:
        # for batch_index, (images, labels), (images2, labels2) in enumerate(concat_data):
        # else:
        for batch_index, (images, labels) in enumerate(concat_data):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)

            if self.use_soft_labels:
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
            else:
                loss_total = calculate_loss(outputs, labels, "CrossEntropy", self.device)

            # Perform BackProp
            loss_total.backward()
            optimizer.step()

            train_loss += loss_total.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.4f}'.format(
                loss_total.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * self.batch_size + len(images),
                total_samples=int(len(concat_data.dataset))
            ))

        # Record Metric
        train_acc = correct.float() / (len(concat_data.dataset))
        train_loss = train_loss / (len(concat_data.dataset))
        train_acc = round(train_acc.item() * 100, 2)
        train_loss = round(train_loss, 5)

        print(f"Average Training Loss: {train_loss}, Acc: {train_acc}")
        if hasattr(self, 'metrics'):
            self.metrics.append(
                {"Epoch": epoch, "Train_Acc": round(train_acc.item() * 100, 2), "Train_Loss": train_loss})

        return train_acc, train_loss

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
