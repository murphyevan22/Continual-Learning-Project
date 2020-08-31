import copy
import numpy as np
import ResNet_CIFAR100
import os
import random
import torch
import train_utils
import torchvision.models as models
import torch.nn as nn


class BackBoneNet:
    def __init__(self, exp_name="", epochs=200, seed=123, data_samples=False, backbone=None, cpg_model=False,outfile=None):
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
        self.data_dir = "/Users/evanm/Documents/College Downloads/Masters Project/Data/CIFAR100"
        self.overall_log = []
        if outfile:
            self.overall_log_file = outfile
        else:
            self.overall_log_file = "overall_log.csv"

        if not os.path.exists(self.data_dir):
            # paths for remote GPU server
            # os.chdir("/home/evan/Baseline/UpperBound/")
            self.data_dir = "/home/evan/Data/CIFAR100/"
            checkpoint_path = "BackBone/"
        else:
            # os.chdir("/Users/evanm/Documents/College Downloads/Masters Project/Baseline/UpperBound/")
            checkpoint_path = "BackBone/"
        if exp_name:
            self.exp_name = exp_name

        # create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.checkpoint_path = os.path.join(checkpoint_path, 'backbone_net_Ep-{epoch}-Ac-{acc}-' + exp_name + '.pth')

        # Model Params
        # self.model = ResNet_CIFAR100.resnet18()
        self.batch_size = {"train": 256, "test": 256}
        self.data_loaders = {}
        self.lr = 0.1
        self.epochs = epochs
        self.loss_function = nn.CrossEntropyLoss()

        if backbone:
            # model =
            # model = ResNet_CIFAR100.resnet18()
            self.model = models.resnet18(pretrained=False)
            self.model.fc = nn.Linear(512, 100)
            self.model.load_state_dict(torch.load(backbone))
            #self.model = self.model

            # _, self.loss_function, _ = train_utils.setup_model(self, num_classes=100, model=model)
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        else:
            if cpg_model:
                self.model = ResNet_CIFAR100.resnet18()
            else:
                self.model = models.resnet18(pretrained=False)
                self.model.fc = nn.Linear(512, 100)

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        if data_samples:
            print("Use own data loaders")
        else:
            self.data_loaders = train_utils.CIFAR_dl_100class(self)

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
            # acc = round(acc.item(), 4)

            # Save best performance model
            if best_acc < acc:
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                best_acc = acc
                best_loss = loss
        print(f"Saving best model: Loss={best_loss}, Acc={best_acc}, Ep={best_epoch}")
        # Save Best model
        torch.save(best_model_wts, self.checkpoint_path.format(epoch=best_epoch, acc=best_acc))

        # Record Metrics
        self.overall_log.append(
            {"Experiment": self.exp_name, "Epoch": best_epoch, "Test_Acc": round(best_acc * 100, 2),
             "Test_Loss": best_loss})
        train_utils.record_overall_metrics(self, ['Experiment', 'Epoch', "Test_Acc", "Test_Loss"])

    def fine_tune(self):
        best_acc = 0.0
        self.lr = 0.01
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        print("Beginning Training for", self.epochs, " Epochs")
        for epoch in range(1, 41):
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

            # Save best performance model
            if best_acc < acc:
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                best_acc = acc
                best_loss = loss
        # Save Best model
        # torch.save(best_model_wts, self.checkpoint_path.format(epoch=best_epoch, acc=round(best_acc * 100, 2)))

        # Record Metrics
        self.overall_log.append(
            {"Experiment": self.exp_name, "Epoch": best_epoch, "Test_Acc": best_acc,
             "Test_Loss": best_loss})
        train_utils.record_overall_metrics(self, ['Experiment', 'Epoch', "Test_Acc", "Test_Loss"])


if __name__ == '__main__':
    # # for a in range(1, 6):
    # #     exp = BackBoneNet(exp_name="ittr=" + str(a), seed=a * 10)
    # #     exp.train_model()
    exp = BackBoneNet(exp_name="CPG_MODEL", cpg_model=True, outfile="CPG_MODEL.csv")
    exp.train_model()
    exp = BackBoneNet(exp_name="PyTorch_MODEL", cpg_model=False, outfile="PyTorch_MODEL.csv")
    exp.train_model()
# # exp = BackBoneNet(exp_name="WD-5e4", op=True)
# # exp.train_model()
