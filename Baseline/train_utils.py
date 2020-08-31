import itertools
import json
import random
import shutil
import glob

from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import pil_loader

from torchvision import datasets, transforms
import torch
import csv
import os

task_to_classes = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}
# class2idx = {'beaver': 0, 'dolphin': 1, 'otter': 2, 'seal': 3, 'whale': 4, 'aquarium_fish': 5, 'flatfish': 6,
#              'ray': 7, 'shark': 8, 'trout': 9, 'orchid': 10, 'poppy': 11, 'rose': 12, 'sunflower': 13,
#              'tulip': 14, 'bottle': 15, 'bowl': 16, 'can': 17, 'cup': 18, 'plate': 19, 'apple': 20,
#              'mushroom': 21, 'orange': 22, 'pear': 23, 'sweet_pepper': 24, 'clock': 25, 'keyboard': 26,
#              'lamp': 27, 'telephone': 28, 'television': 29, 'bed': 30, 'chair': 31, 'couch': 32, 'table': 33,
#              'wardrobe': 34, 'bee': 35, 'beetle': 36, 'butterfly': 37, 'caterpillar': 38, 'cockroach': 39,
#              'bear': 40, 'leopard': 41, 'lion': 42, 'tiger': 43, 'wolf': 44, 'bridge': 45, 'castle': 46,
#              'house': 47, 'road': 48, 'skyscraper': 49, 'cloud': 50, 'forest': 51, 'mountain': 52, 'plain': 53,
#              'sea': 54, 'camel': 55, 'cattle': 56, 'chimpanzee': 57, 'elephant': 58, 'kangaroo': 59, 'fox': 60,
#              'porcupine': 61, 'possum': 62, 'raccoon': 63, 'skunk': 64, 'crab': 65, 'lobster': 66, 'snail': 67,
#              'spider': 68, 'worm': 69, 'baby': 70, 'boy': 71, 'girl': 72, 'man': 73, 'woman': 74, 'crocodile': 75,
#              'dinosaur': 76, 'lizard': 77, 'snake': 78, 'turtle': 79, 'hamster': 80, 'mouse': 81, 'rabbit': 82,
#              'shrew': 83, 'squirrel': 84, 'maple_tree': 85, 'oak_tree': 86, 'palm_tree': 87, 'pine_tree': 88,
#              'willow_tree': 89, 'bicycle': 90, 'bus': 91, 'motorcycle': 92, 'pickup_truck': 93, 'train': 94,
#              'lawn_mower': 95, 'rocket': 96, 'streetcar': 97, 'tank': 98, 'tractor': 99}

class_to_idx_sorted = {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6,
                       'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13,
                       'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19,
                       'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25,
                       'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29, 'dolphin': 30, 'elephant': 31,
                       'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38,
                       'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44,
                       'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50,
                       'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56,
                       'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62,
                       'porcupine': 63, 'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68,
                       'rocket': 69, 'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75,
                       'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81,
                       'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 'tank': 85, 'telephone': 86, 'television': 87,
                       'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94,
                       'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}

superclass = [
    'None',  # dummy
    'aquatic_mammals',
    'fish',
    'flowers',
    'food_containers',
    'fruit_and_vegetables',
    'household_electrical_devices',
    'household_furniture',
    'insects',
    'large_carnivores',
    'large_man-made_outdoor_things',
    'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores',
    'medium_mammals',
    'non-insect_invertebrates',
    'people',
    'reptiles',
    'small_mammals',
    'trees',
    'vehicles_1',
    'vehicles_2',
]
mean = {
    'cifar100': [0.5071, 0.4865, 0.4409],
    'two_class': [0.5071, 0.4866, 0.4409],
    'aquatic_mammals': [0.4242, 0.4783, 0.4975],
    'fish': [0.4162, 0.4626, 0.4577],
    'flowers': [0.5233, 0.4080, 0.3058],
    'food_containers': [0.5824, 0.5459, 0.5111],
    'fruit_and_vegetables': [0.6010, 0.4927, 0.3616],
    'household_electrical_devices': [0.5573, 0.5368, 0.5251],
    'household_furniture': [0.6104, 0.5407, 0.4902],
    'insects': [0.5568, 0.5380, 0.4312],
    'large_carnivores': [0.4719, 0.4395, 0.3713],
    'large_man-made_outdoor_things': [0.4848, 0.5094, 0.5116],
    'large_natural_outdoor_scenes': [0.4684, 0.4971, 0.5026],
    'large_omnivores_and_herbivores': [0.4832, 0.4650, 0.3997],
    'medium_mammals': [0.4511, 0.4291, 0.3669],
    'non-insect_invertebrates': [0.5093, 0.4760, 0.4164],
    'people': [0.5193, 0.4570, 0.4242],
    'reptiles': [0.4877, 0.4781, 0.4336],
    'small_mammals': [0.5117, 0.4798, 0.4125],
    'trees': [0.4544, 0.4833, 0.4314],
    'vehicles_1': [0.4979, 0.4858, 0.4642],
    'vehicles_2': [0.5301, 0.5280, 0.5037]
}

std = {
    'cifar100': [0.2673, 0.2564, 0.2762],
    'two_class': [0.26, 0.2517, 0.268],
    'aquatic_mammals': [0.2495, 0.2327, 0.2515],
    'fish': [0.2857, 0.2574, 0.2705],
    'flowers': [0.2892, 0.2489, 0.2613],
    'food_containers': [0.2761, 0.2782, 0.2888],
    'fruit_and_vegetables': [0.2902, 0.2837, 0.2952],
    'household_electrical_devices': [0.2933, 0.2937, 0.3028],
    'household_furniture': [0.2603, 0.2774, 0.2956],
    'insects': [0.2736, 0.2630, 0.2923],
    'large_carnivores': [0.2352, 0.2244, 0.2276],
    'large_man-made_outdoor_things': [0.2408, 0.2411, 0.2736],
    'large_natural_outdoor_scenes': [0.2349, 0.2283, 0.2731],
    'large_omnivores_and_herbivores': [0.2428, 0.2375, 0.2373],
    'medium_mammals': [0.2355, 0.2248, 0.2261],
    'non-insect_invertebrates': [0.2645, 0.2566, 0.2684],
    'people': [0.2801, 0.2735, 0.2794],
    'reptiles': [0.2619, 0.2418, 0.2526],
    'small_mammals': [0.2389, 0.2288, 0.2377],
    'trees': [0.2356, 0.2345, 0.2817],
    'vehicles_1': [0.2633, 0.2609, 0.2733],
    'vehicles_2': [0.2730, 0.2693, 0.2874],
}


# Custom Dataset
class CIFAR100_Dataset(DatasetFolder):
    def __init__(self, root, transform, num_tasks, multi_task=False, loader=pil_loader):
        self.multi_task = multi_task
        self.num_tasks = num_tasks
        super().__init__(root, loader, ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
                         transform)
        self.targets = [s[1] for s in self.samples]

    def _find_classes(self, dir):
        classes = []
        if self.multi_task:
            # Check this for samples
            for i in range(1, self.num_tasks):
                for c in task_to_classes[superclass[i]]:
                    classes.append(c)
        else:
            classes = task_to_classes[superclass[self.num_tasks]]
        print("Classes: ", classes)
        class_to_idx = {k: class_to_idx_sorted[k] for k in classes}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # tmp = path.split("\\")[-2]
        # target = self.class_to_idx[tmp]
        # print(f"{path}-->{target}")

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def CIFAR_important_samples(self, task, per_task_norm=True):
    dl, ds = {}, {}
    # Normalize data sets
    if per_task_norm:
        tsforms = {
            "train": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean[task], std=std[task])
            ]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean[task], std=std[task])
            ])
        }
    else:
        tsforms = {
            "train": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        }
    ds['train'] = datasets.ImageFolder(self.samples_dir + '{}'.format(task),
                                       tsforms['train'])
    ds['test'] = datasets.ImageFolder(self.data_dir + 'test/{}'.format(task),
                                      tsforms['test'])
    # for dset in ds.keys():
    #     rvse_map = {v: k for k, v in ds[dset].class_to_idx.items()}
    #     ds[dset].class_to_idx = class_to_idx
    #     ds[dset].targets = [class_to_idx[rvse_map[cls]] for cls in ds[dset].targets]

    dl['train'] = torch.utils.data.DataLoader(ds['train'], batch_size=self.batch_size["train"], shuffle=True,
                                              num_workers=4)
    dl['test'] = torch.utils.data.DataLoader(ds['test'], batch_size=self.batch_size["test"], shuffle=False,
                                             num_workers=4)
    for loader in dl.keys():
        print("Dataset:", loader, "\tSize:", len(dl[loader].dataset))
    return dl


def CIFAR_dl_task(self, task, per_task_norm=True, imp_samples=False):
    dl, ds = {}, {}
    # Normalize data sets
    if per_task_norm:
        tsforms = {
            "train": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean[task], std=std[task])
            ]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean[task], std=std[task])
            ])
        }
    else:
        tsforms = {
            "train": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        }
    if imp_samples:
        ds['train'] = custom_DS(self.data_dir + 'train/{}'.format(task),
                                tsforms['train'])
        ds['test'] = custom_DS(self.data_dir + 'test/{}'.format(task),
                               tsforms['test'])
    else:
        ds['train'] = datasets.ImageFolder(self.data_dir + 'train/{}'.format(task),
                                           tsforms['train'])
        ds['test'] = datasets.ImageFolder(self.data_dir + 'test/{}'.format(task),
                                          tsforms['test'])
    # for dset in ds.keys():
    #     rvse_map = {v: k for k, v in ds[dset].class_to_idx.items()}
    #     ds[dset].class_to_idx = class_to_idx
    #     ds[dset].targets = [class_to_idx[rvse_map[cls]] for cls in ds[dset].targets]
    dl['train'] = torch.utils.data.DataLoader(ds['train'], batch_size=self.batch_size["train"], shuffle=True,
                                              num_workers=4)
    dl['test'] = torch.utils.data.DataLoader(ds['test'], batch_size=self.batch_size["test"], shuffle=False,
                                             num_workers=4)
    for loader in dl.keys():
        print("Dataset:", loader, "\tSize:", len(dl[loader].dataset))
    return dl


def CIFAR_dl_100class(self, tsforms=None):
    dl, ds = {}, {}

    # Normalize data sets
    if tsforms is None:
        tsforms = {'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]), 'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])}
    print("Data: ", self.data_dir)
    ds['train'] = datasets.CIFAR100(root=self.data_dir,
                                    train=True,
                                    download=False,
                                    transform=tsforms['train'])

    ds['test'] = datasets.CIFAR100(root=self.data_dir,
                                   train=False,
                                   download=False,
                                   transform=tsforms['test'])

    # train_path = Data / CIFAR100
    # ds['train'] = CIFAR100_Dataset(train_path, tsforms['train'], 21)
    # ds['test'] = CIFAR100_Dataset(test_path, tsforms['test'], 21)

    dl['train'] = torch.utils.data.DataLoader(
        ds['train'], batch_size=self.batch_size['train'], shuffle=True, num_workers=4)

    dl['test'] = torch.utils.data.DataLoader(
        ds['test'], batch_size=self.batch_size['test'], shuffle=False, num_workers=4)

    for loader in dl.keys():
        print("Dataset:", loader, "\tSize:", len(dl[loader].dataset))

    return dl


def train(self, epoch):
    self.model.train()
    train_loss = 0.0
    correct = 0.0
    for batch_index, (images, labels) in enumerate(self.data_loaders['train']):
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
    train_loss / (len(self.data_loaders['train'].dataset))
    if hasattr(self, 'metrics'):
        self.metrics.append({"Epoch": epoch, "Train_Acc": round(train_acc.item() * 100, 2), "Train_Loss": train_loss})


def evaluate(self):
    self.model.eval()
    test_loss = 0.0
    correct = 0.0

    for (images, labels) in self.data_loaders['test']:
        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    test_acc = correct.float() / (len(self.data_loaders['test'].dataset))
    test_loss = (test_loss / (len(self.data_loaders['test'].dataset)))
    test_acc = round(test_acc.item() * 100, 2)
    test_loss = round(test_loss, 4)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}'.format(
        test_loss, test_acc))
    print()
    #test_loss = (test_loss / (len(self.data_loaders['test'].dataset)))

    if hasattr(self, 'metrics'):
        self.metrics[-1]['Test_Acc'] = test_acc
        self.metrics[-1]['Test_Loss'] = test_loss
    return test_acc, test_loss


# def setup_model(self, num_classes, model=None):
#     print("Don't use this!!!")
#     if not model:
#         print("Using Model:\tResNet18 on Device:\t", self.device)
#         model = ResNet_CIFAR100.resnet18(num_classes=num_classes)
#     model = model.to(self.device)
#
#     if not hasattr(self, 'loss_function'):
#         print("Using Loss Function:\tCross Entropy Loss")
#         loss_function = nn.CrossEntropyLoss()
#
#     if not hasattr(self, 'optimizer'):
#         print("Using Optimizer Function:\tSGD with Momentum and LR:\t", self.lr)
#         optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
#         # optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)
#
#     return model, loss_function, optimizer


def record_metrics(self):
    # Write results to CSV
    csv_columns = ['Epoch', 'Train_Acc', 'Train_Loss', "Test_Acc", "Test_Loss"]
    with open(self.log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for data in self.metrics:
            writer.writerow(data)


def record_overall_metrics(self, csv_columns=None):
    # Write results to CSV
    if not csv_columns:
        csv_columns = ['Task', 'Epoch', "Test_Acc", "Test_Loss"]
    with open(self.overall_log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        if os.stat(self.overall_log_file).st_size == 0:
            writer.writeheader()
        for data in self.overall_log:
            writer.writerow(data)


def save_model(model, path):
    pass


def get_retained_samples(self, percent, num_tasks):
    samples = {}
    qty = int(500 * (percent / 100))
    print(f"Retaining {percent}% of 500 samples ({qty})")
    print(f"Path to records: {self['samples_records_dir']}")
    print(f"Path to data: {self['data_dir']}")

    print(f"Starting from {1} to {num_tasks}")
    # for task_id in range(start, num_tasks + 1):
    for task_id in range(1, num_tasks + 1):
        task = superclass[task_id]

        print(f"Getting important samples for {task}")
        path = self['samples_records_dir'] + f"{task}.json"
        with open(path) as f:
            json_obj = json.load(f)
            samples[task] = {}
            for subclass in json_obj.keys():
                if len(json_obj[subclass]) < qty:
                    print(f"Not enough samples! Qty: {qty} --> {len(json_obj[subclass])}")
                    qty = len(json_obj[subclass])

            for subclass in json_obj.keys():
                samples[task][subclass] = dict(itertools.islice(json_obj[subclass].items(), qty))
    return samples


def get_random_samples(self, percent, num_tasks):
    samples = {}
    # Qty for CIFAR100 DS
    qty = int(500 * (percent / 100))
    print(f"Retaining {percent}% of 500 samples ({qty})")
    print(f"Path to data: {self['data_dir']}")

    print(f"Starting from {1} to {num_tasks}")
    # for task_id in range(start, num_tasks + 1):
    for task_id in range(1, num_tasks + 1):
        task = superclass[task_id]
        print(f"Getting random samples for {task}")

        samples[task] = {}
        task_dir = glob.glob(self['data_dir'] + f"train/{task}/*")
        for subclass in task_dir:
            subclass = subclass.replace("\\", "/")
            samples[task][subclass.split('/')[-1]] = random.sample(
                [img.replace('.png', '') for img in glob.glob(f"{subclass}/*")], qty)
    return samples


def update_samples_CIFAR100(self, num_tasks=20):
    # Remove Existing Samples
    if os.path.exists(self['samples_dir']):
        print("Removing existing samples")
        shutil.rmtree(self['samples_dir'])
    os.mkdir(self['samples_dir'])
    if self['random_samples']:
        print("Using Random Samples")
        samples = get_random_samples(self, self['retain_percent'], num_tasks)
    else:
        print("Using Important Samples")
        samples = get_retained_samples(self, self['retain_percent'], num_tasks)

    print("Updating important samples in directory")
    for task in samples.keys():
        for sub_task in samples[task].keys():
            if isinstance(samples[task][sub_task], list):
                images = samples[task][sub_task]
            else:
                images = samples[task][sub_task].keys()
            for sample in images:
                sample = sample.replace("\\", "/")
                sample = sample.split('/')[-1]
                # dest = f"{self['samples_dir']}{task}/{sub_task}/"
                dest = f"{self['samples_dir']}/{sub_task}/"
                source = f"{self['data_dir']}train/{task}/{sub_task}/{sample}.png"
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(source, dest)
    print("Important samples updated!")


# def get_retained_samples(self, percent=50):
#     samples = {}
#     qty = int(500 * (percent / 100))
#     print(f"Retaining {percent}% of 500 samples ({qty})")
#     print(f"Path to records: {self['samples_records_dir']}")
#     print(f"Path to data: {self['data_dir']}")
#     for idx, task in enumerate(superclass):
#         if idx != 0:
#             # qty = int(len(self.data_loaders['train'].dataset/5) * (percent / 100))
#             # print(f"Retaining {percent}% of {len(self.data_loaders['train'].dataset)/5} samples ({qty})")
#             print(f"Getting important samples for {task}")
#             path = self['samples_records_dir'] + f"{task}.json"
#             with open(path) as f:
#                 json_obj = json.load(f)
#                 samples[task] = {}
#                 for subclass in json_obj.keys():
#                     if len(json_obj[subclass]) < qty:
#                         print(f"Not enough samples! Qty: {qty} --> {len(samples[subclass])}")
#                         qty = len(samples[subclass])
#
#                 for subclass in json_obj.keys():
#                     samples[task][subclass] = dict(itertools.islice(json_obj[subclass].items(), qty))
#     return samples
#
#
# def get_random_samples(self, percent=50):
#     samples = {}
#     # Qty for CIFAR100 DS
#     qty = int(500 * (percent / 100))
#     print(f"Retaining {percent}% of 500 samples ({qty})")
#     print(f"Path to data: {self['data_dir']}")
#     for idx, task in enumerate(superclass):
#         if idx != 0:
#             print(f"Getting random samples for {task}")
#
#             samples[task] = {}
#             task_dir = glob.glob(self['data_dir'] + f"train/{task}/*")
#             for subclass in task_dir:
#                 subclass = subclass.replace("\\", "/")
#                 samples[task][subclass.split('/')[-1]] = random.sample(
#                     [img.replace('.png', '') for img in glob.glob(f"{subclass}/*")], qty)
#     return samples


class custom_DS(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(custom_DS, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0].split("/")[-1].replace(".png", "")
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def samples_data_loader(self, tsforms=None):
    dl, ds = {}, {}
    # Normalize data sets
    if tsforms is None:
        tsforms = {'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]), 'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])}
    print("Data: ", self.data_dir)
    # ds['train'] = datasets.ImageFolder(self.samples_dir, tsforms['train'])
    ds['train'] = CIFAR100_Dataset(self.samples_dir, tsforms['train'], num_tasks=21, multi_task=True)

    ds['test'] = datasets.CIFAR100(root=self.data_test_dir,
                                   train=False,
                                   download=False,
                                   transform=tsforms['test'])
    if ds['train'].class_to_idx != ds['train'].class_to_idx:
        raise Exception("Class mappings not the same in train and test datasets")
    else:
        print("Class mappings correct!")
    # for dset in ds.keys():
    #     rvse_map = {v: k for k, v in ds[dset].class_to_idx.items()}
    #     ds[dset].class_to_idx = class_to_idx
    #     ds[dset].targets = [class_to_idx[rvse_map[cls]] for cls in ds[dset].targets]
    dl['train'] = torch.utils.data.DataLoader(
        ds['train'], batch_size=self.batch_size['train'], shuffle=True, num_workers=4)

    dl['test'] = torch.utils.data.DataLoader(
        ds['test'], batch_size=self.batch_size['test'], shuffle=False, num_workers=4)

    for loader in dl.keys():
        print("Dataset:", loader, "\tSize:", len(dl[loader].dataset))

    return dl
