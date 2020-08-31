import random
import train_utils
import torch
import os
import copy
import numpy as np


class LowerBound:
    def __init__(self, task, num_classes=5, exp_name="", per_task_norm=True):
        # Seed Random Processes
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seed = 123
        torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
        torch.manual_seed(seed)  # CPU seed
        if self.device == torch.device("cuda:0"):
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
        self.data_dir = "/Users/evanm/Documents/College Downloads/Masters Project/CPG/data/cifar100_org/"

        if not os.path.exists(self.data_dir):
            # paths for remote GPU server
            self.home = "/home/evan/Baseline/LowerBound/"
            self.data_dir = "/home/evan/CPG/data/cifar100_org/"
        else:
            self.home = "/Users/evanm/Documents/College Downloads/Masters Project/Baseline/LowerBound/"

        if not os.path.exists(self.home + exp_name):
            os.mkdir(self.home + exp_name)
        os.chdir(self.home + exp_name)
        self.model_path = "models/"
        self.log_path = "metrics/"

        self.overall_log_file = self.home + exp_name.replace("/", "") + "overall_log.csv"
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.log_path += task + '.csv'
        # create folder to save model
        #self.model_path = self.model_path + "/" + self.task
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model_path = os.path.join(self.model_path, '{task}-{epoch}-{acc}.pth')

        # Model Params
        self.epochs = 200
        self.lr = 0.1
        self.batch_size = {"train": 256, "test": 256}
        # Remove this and do it all in main to ensure correct model params used
        self.model, self.loss_function, self.optimizer = train_utils.setup_model(self, num_classes=num_classes)
        self.metrics, self.overall_log = [], []

        # Get data loaders for datasets
        self.data_loaders = train_utils.CIFAR_dl_task(self, task, per_task_norm)

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
        torch.save(best_model_wts, self.model_path.format(task=self.task, epoch=best_epoch, acc=round(best_acc * 100, 2)))

        # Record Metrics
        train_utils.record_metrics(self)

        # This is not test acc, change to best acc in all experiments
        self.overall_log.append({"Task": self.task, "Epoch": best_epoch, "Test_Acc": round(best_acc * 100, 2), "Test_Loss": best_loss})
        train_utils.record_overall_metrics(self)


if __name__ == '__main__':
    for idx, classes in enumerate(train_utils.superclass):
        lb = LowerBound(task=classes, exp_name="FinalRun_pt_true", per_task_norm=False)
        lb.train_model()
    # for ittr in range(1, 6):
    #     print(f"Itteration: {ittr}")
    #     for idx, classes in enumerate(train_utils.superclass):
    #         if idx != 0:
    #             lb = LowerBound(task=classes, exp_name="PTNorm_ittr=" + str(ittr), per_task_norm=True)
    #             lb.train_model()
    #
    # for ittr in range(1, 6):
    #     print(f"Itteration: {ittr}")
    #     for idx, classes in enumerate(train_utils.superclass):
    #         if idx != 0:
    #             lb = LowerBound(task=classes, exp_name="OvNorm_ittr=" + str(ittr), per_task_norm=False)
    #             lb.train_model()
