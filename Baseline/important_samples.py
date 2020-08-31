import copy
import numpy as np
import os
import random
import torch
import torchvision.models as models

import ResNet_CIFAR100
import train_utils
import json
import torch.nn as nn


class ImportantSamples:
    def __init__(self, task, epochs=200, num_classes=5, exp_name="", important_samples=False, seed=123, percent=25,
                 per_task_norm=True, cifar_RN_model=True):
        # Seed Random Processes
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
        torch.manual_seed(seed)  # CPU seed
        if self.device == torch.device("cuda:0"):
            print("Cuda Seeding..")
            torch.cuda.manual_seed_all(seed)  # GPU seed
        random.seed(seed)  # python seed for image transformation
        np.random.seed(seed)

        print("*" * 50)
        print("Task:\t", task)
        if len(exp_name) != 0:
            print("Experiment:\t", exp_name)
            exp_name += "/"
        print("Using Device: ", self.device)
        print("*" * 50)

        # Directory Config
        self.task = task
        self.data_dir = "/Users/evanm/Documents/College Downloads/Masters Project/Data/cifar100_org/"

        if not os.path.exists(self.data_dir):
            # paths for remote GPU server
            self.home = "/home/evan/Baseline/ImportantSamples/"
            self.data_dir = "/home/evan/CPG/data/cifar100_org/"
        else:
            self.home = "/Users/evanm/Documents/College Downloads/Masters Project/Baseline/ImportantSamples/"

        if not os.path.exists(self.home + exp_name):
            os.mkdir(self.home + exp_name)
        os.chdir(self.home + exp_name)
        self.model_path = "models/"
        self.log_path = "metrics/"

        self.overall_log_file = self.home + exp_name.replace("/", "") + "important_samples_log.csv"
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.log_path += task + '.csv'
        # create folder to save model
        # self.model_path = self.model_path + "/" + self.task

        # Model Params
        self.epochs = epochs
        self.lr = 0.1
        self.batch_size = {"train": 256, "test": 500}
        if cifar_RN_model:
            self.model = ResNet_CIFAR100.resnet18(num_classes)
        else:
            self.model = models.resnet18(pretrained=False)
            self.model.fc = nn.Linear(512, num_classes)

        self.model = self.model.to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        self.metrics, self.overall_log = [], []

        # Get data loaders for datasets
        # Use important Samples
        if important_samples:
            self.imp_samples_dir = self.home
            self.samples_records_dir = '/'.join(self.home.split('/')[:-2]) + '/ImportantSamplesRecords/'
            print("Samples Records Dir: ", self.samples_records_dir)
            train_utils.update_samples(self, percent=percent)
            self.data_loaders = train_utils.CIFAR_important_samples(self, task, per_task_norm)
        else:
            self.data_loaders = train_utils.CIFAR_dl_task(self, task, per_task_norm, imp_samples=True)

        # Dict for forgetting events
        self.classes = self.data_loaders['train'].dataset.classes
        self.forget_events = {self.classes[0]: {}, self.classes[1]: {}, self.classes[2]: {}, self.classes[3]: {},
                              self.classes[4]: {}, }
        self.prev_correct = {}

    def calculate_important_samples(self):
        best_acc = 0.0
        print("Beginning Training for", self.epochs, " Epochs")

        for epoch in range(1, self.epochs + 1):
            if epoch == 80:
                self.lr = 0.01
                for group in self.optimizer.param_groups:
                    group['lr'] = self.lr
            elif epoch == 140:
                self.lr = 0.001
                for group in self.optimizer.param_groups:
                    group['lr'] = self.lr

            self.custom_train(epoch)
            acc, loss = self.custom_evaluate()
            #acc = round(acc.item(), 4)
            loss = round(loss, 4)

            # Save best performance model
            if best_acc < acc:
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                best_acc = acc
                best_loss = loss

        for a in range(0, 5):
            print(f"Class={self.classes[a]}")
            self.forget_events[self.classes[a]] = {k: v for k, v in
                                                   sorted(self.forget_events[self.classes[a]].items(),
                                                          key=lambda item: item[1],
                                                          reverse=True)}
            for key, val in self.forget_events[self.classes[a]].items():
                print(f"Class={self.classes[a]}:\t{key}: {val}")

        with open(f'{self.task}.json', 'w') as fp:
            json.dump(self.forget_events, fp)

        # Save Best model
        # torch.save(best_model_wts, self.model_path.format(task=self.task, epoch=best_epoch, acc=round(best_acc * 100, 2)))

        # Record Metrics
        train_utils.record_metrics(self)

        self.overall_log.append(
            {"Task": self.task, "Epoch": best_epoch, "Test_Acc": round(best_acc * 100, 2), "Test_Loss": best_loss})
        train_utils.record_overall_metrics(self)

    def custom_train(self, epoch):
        self.model.train()
        train_loss = 0.0
        correct = 0.0
        prev_correct_record = self.prev_correct
        self.prev_correct = []
        for batch_index, (images, labels, fnames) in enumerate(self.data_loaders['train']):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
            for fname, label, pred in zip(fnames, labels, preds):
                # Check has been misclassified and was classified correctly last time
                if label != pred and fname in prev_correct_record:
                    if fname in self.forget_events[self.classes[label.item()]]:
                        self.forget_events[self.classes[label.item()]][fname] += 1
                    else:
                        self.forget_events[self.classes[label.item()]][fname] = 1
                elif label == pred:
                    self.prev_correct.append(fname)

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                self.optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * self.batch_size['train'] + len(images),
                total_samples=int(len(self.data_loaders['train'].dataset))
            ))
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        # Record Metric
        train_acc = correct.float() / (len(self.data_loaders['train'].dataset))
        train_loss = train_loss / (len(self.data_loaders['train'].dataset))
        if hasattr(self, 'metrics'):
            self.metrics.append(
                {"Epoch": epoch, "Train_Acc": round(train_acc.item() * 100, 2), "Train_Loss": train_loss})

    def custom_evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0.0

        for (images, labels, fnames) in self.data_loaders['test']:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        test_acc = correct.float() / (len(self.data_loaders['test'].dataset))
        test_acc = round(test_acc.item() * 100, 2)

        test_loss = (test_loss / (len(self.data_loaders['test'].dataset)))
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))
        print()

        if hasattr(self, 'metrics'):
            self.metrics[-1]['Test_Acc'] = test_acc
            self.metrics[-1]['Test_Loss'] = test_loss
        return test_acc, test_loss

    def train_model(self):
        best_acc = 0.0
        print("Beginning Training for", self.epochs, " Epochs")

        for epoch in range(1, self.epochs + 1):
            if epoch == 80:
                self.lr = 0.01
                for group in self.optimizer.param_groups:
                    group['lr'] = self.lr
            elif epoch == 140:
                self.lr = 0.001
                for group in self.optimizer.param_groups:
                    group['lr'] = self.lr

            train_utils.train(self, epoch)
            acc, loss = train_utils.evaluate(self)
            acc = round(acc.item(), 4)
            loss = round(loss, 4)

            # Save best performance model
            if best_acc < acc:
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                best_acc = acc
                best_loss = loss

        # Save Best model
        # torch.save(best_model_wts, self.model_path.format(task=self.task, epoch=best_epoch, acc=round(best_acc * 100, 2)))

        # Record Metrics
        train_utils.record_metrics(self)

        self.overall_log.append(
            {"Task": self.task, "Epoch": best_epoch, "Test_Acc": round(best_acc * 100, 2), "Test_Loss": best_loss})
        train_utils.record_overall_metrics(self)


if __name__ == '__main__':
    print("Running Important Samples Class")
    for idx, classes in enumerate(train_utils.superclass):
        if idx != 0:
            exp = ImportantSamples(task=classes, exp_name="FINAL_IMS_SAMPLES_RN_CIFAR", epochs=100, per_task_norm=True)
            exp.calculate_important_samples()
    for idx, classes in enumerate(train_utils.superclass):
        if idx != 0:
            exp = ImportantSamples(task=classes, exp_name="FINAL_IMS_SAMPLES_PyTorch_RN", epochs=100, per_task_norm=True, cifar_RN_model=False)
            exp.calculate_important_samples()
    # Check for acc while training using important samples
    # for idx, classes in enumerate(train_utils.superclass):
    #     if idx != 0:
    #         exp = ImportantSamples(task=classes, exp_name="25_pc_retain", important_samples=True, percent=25, epochs=200, per_task_norm=True)
    #         exp.train_model()

    #
    # for idx, classes in enumerate(train_utils.superclass):
    #     if idx != 0:
    #         exp = ImportantSamples(task=classes, exp_name="50_pc_retain", important_samples=True, percent=50, epochs=60, per_task_norm=True)
    #         exp.train_model()
    #
    # for idx, classes in enumerate(train_utils.superclass):
    #     if idx != 0:
    #         exp = ImportantSamples(task=classes, exp_name="75_pc_retain", important_samples=True, percent=75, epochs=200, per_task_norm=True)
    #         exp.train_model()
    #
    # # Check for acc while training using random samples
