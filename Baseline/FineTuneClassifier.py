import numpy as np
import train_utils
import copy
import os
import csv
import torch
import torch.nn as nn
import random
import glob


class FineTuneClassifier:
    def __init__(self, backbone, exp_name="", per_task_norm=True, seed=123):
        # Seed Random Processes
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
        torch.manual_seed(seed)  # CPU seed
        if self.device == torch.device("cuda:0"):
            torch.cuda.manual_seed_all(seed)  # GPU seed
        random.seed(seed)  # python seed for image transformation
        np.random.seed(seed)
        print("Using Device: ", self.device)

        # Directory Config
        self.data_dir = "/Users/evanm/Documents/College Downloads/Masters Project/CPG/data/cifar100_org/"
        if not os.path.exists(self.data_dir):
            # paths for remote GPU server
            self.data_dir = "/home/evan/CPG/data/cifar100_org/"
            self.home = "/home/evan/Baseline/UpperBound/"

        else:
            self.home = "/Users/evanm/Documents/College Downloads/Masters Project/Baseline/UpperBound/"

        self.classifier_path = self.home + "/Classifiers/"
        # create folder to save model
        if not os.path.exists(self.classifier_path):
            os.mkdir(self.classifier_path)
        self.classifier_path = os.path.join(self.classifier_path, 'Exp{exp}--{task}-{epoch}-{acc}.pth')

        # Model Params
        self.lr = 0.01
        self.model = None
        self.backbone_model, self.loss_function, _ = train_utils.setup_model(self, num_classes=100)
        self.optimizer = None

        self.backbone_model.load_state_dict(torch.load(backbone), strict=True)
        self.backbone_model = self.backbone_model.to(self.device)
        self.batch_size = {"train": 256, "test": 256}
        self.epochs = 30
        self.classifier_results = []
        self.per_task_norm = per_task_norm
        self.data_loaders = {}

    def finetune_classifier(self, task, ittr="0"):
        print('-' * 50)
        print("Training task:\t", task)
        self.data_loaders = train_utils.CIFAR_dl_task(self, task, self.per_task_norm)
        best_acc = 0.0

        # Setup Model
        model = self.backbone_model
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, 5)
        self.model = model.to(self.device)
        self.lr = 0.01
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        print("Finetuning for", self.epochs, " Epochs")
        for epoch in range(1, self.epochs + 1):
            if epoch == 10:
                self.lr = self.lr * 0.1
                for group in self.optimizer.param_groups:
                    group['lr'] = self.lr
            elif epoch == 20:
                self.lr = self.lr * 0.1
                for group in self.optimizer.param_groups:
                    group['lr'] = self.lr

            train_utils.train(self, epoch)
            acc, loss = train_utils.evaluate(self)
            acc = round(acc.item(), 4)
            loss = round(loss, 4)

            # Save best performance model
            if best_acc < acc:
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_acc = acc
                best_loss = loss
                best_epoch = epoch

        # Save Best model
        torch.save(best_model_wts,
                   self.classifier_path.format(exp=ittr, task=task, epoch=best_epoch, acc=round(best_acc * 100, 2)))
        # Record Metrics
        self.classifier_results.append({"Task": task, "Acc": round(best_acc * 100, 2), "Loss": best_loss})


if __name__ == '__main__':
    if os.path.exists("/Users/evanm/Documents/College Downloads/Masters Project/Baseline/UpperBound/BackBone/"):
        print("Running on Local Machine")
        home_path = "/Users/evanm/Documents/College Downloads/Masters Project/Baseline/UpperBound/"
        backbones = glob.glob(home_path + "BackBone/*.pth")
    else:
        print("Running on GPU Server")
        home_path = "/home/evan/Baseline/UpperBound/"
        backbones = glob.glob(home_path + "BackBone/*.pth")
    print("Backbones to test: ", len(backbones))
    for ittr, bb in enumerate(backbones):
        print("Running:\t", bb[-10:].replace(".pth", ""))
        exp = FineTuneClassifier(backbone=bb)

        # Train 20 classifiers
        for idx, classes in enumerate(train_utils.superclass):
            if idx != 0:
                exp.finetune_classifier(classes, ittr + 1)

        # Write results to CSV
        csv_columns = ['Task', 'Acc', 'Loss']
        csv_file = exp.home + "_PTNorm_" + bb[-10:].replace(".pth", "_") + "Results.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for data in exp.classifier_results:
                writer.writerow(data)

    for ittr, bb in enumerate(backbones):
        print("Running:\t", bb[-10:].replace(".pth", ""))
        exp = FineTuneClassifier(backbone=bb, per_task_norm=False)

        # Train 20 classifiers
        for idx, classes in enumerate(train_utils.superclass):
            if idx != 0:
                exp.finetune_classifier(classes, ittr=str(ittr + 1) + "ON")

        # Write results to CSV
        csv_columns = ['Task', 'Acc', 'Loss']
        csv_file = exp.home + "_OvNorm_" + bb[-10:].replace(".pth", "_") + "Results.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for data in exp.classifier_results:
                writer.writerow(data)

    # exp = FineTuneClassifier(backbone=backbones[0])
    #
    # # Train 20 classifiers
    # for idx, classes in enumerate(train_utils.superclass):
    #     if idx != 0:
    #         exp.finetune_classifier(classes)
    ##exp.data_dir = "/Users/evanm/Documents/College Downloads/Masters Project/CIFAR100"
    # exp.data_loaders = train_utils.CIFAR_dl_100class(exp)
    # exp.model = exp.backbone_model
    # train_utils.evaluate(self=exp)
    # model = exp.backbone_model
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc = nn.Linear(512, 5)
    # exp.model = model.to(exp.device)
    # print(exp.model)
    # for param in exp.model.parameters():
    #     print("Requires Grad:\t", param, "\t:", param.requires_grad)

    #     # Train 20 classifiers
    #     for idx, classes in enumerate(train_utils.superclass):
    #         if idx != 0:
    #             exp.finetune_classifier(classes, ittr+1)
