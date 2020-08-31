import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import pil_loader
from collections import Counter

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


class2idx = {'beaver': 0, 'dolphin': 1, 'otter': 2, 'seal': 3, 'whale': 4, 'aquarium_fish': 5, 'flatfish': 6,
             'ray': 7, 'shark': 8, 'trout': 9, 'orchid': 10, 'poppy': 11, 'rose': 12, 'sunflower': 13,
             'tulip': 14, 'bottle': 15, 'bowl': 16, 'can': 17, 'cup': 18, 'plate': 19, 'apple': 20,
             'mushroom': 21, 'orange': 22, 'pear': 23, 'sweet_pepper': 24, 'clock': 25, 'keyboard': 26,
             'lamp': 27, 'telephone': 28, 'television': 29, 'bed': 30, 'chair': 31, 'couch': 32, 'table': 33,
             'wardrobe': 34, 'bee': 35, 'beetle': 36, 'butterfly': 37, 'caterpillar': 38, 'cockroach': 39,
             'bear': 40, 'leopard': 41, 'lion': 42, 'tiger': 43, 'wolf': 44, 'bridge': 45, 'castle': 46,
             'house': 47, 'road': 48, 'skyscraper': 49, 'cloud': 50, 'forest': 51, 'mountain': 52, 'plain': 53,
             'sea': 54, 'camel': 55, 'cattle': 56, 'chimpanzee': 57, 'elephant': 58, 'kangaroo': 59, 'fox': 60,
             'porcupine': 61, 'possum': 62, 'raccoon': 63, 'skunk': 64, 'crab': 65, 'lobster': 66, 'snail': 67,
             'spider': 68, 'worm': 69, 'baby': 70, 'boy': 71, 'girl': 72, 'man': 73, 'woman': 74, 'crocodile': 75,
             'dinosaur': 76, 'lizard': 77, 'snake': 78, 'turtle': 79, 'hamster': 80, 'mouse': 81, 'rabbit': 82,
             'shrew': 83, 'squirrel': 84, 'maple_tree': 85, 'oak_tree': 86, 'palm_tree': 87, 'pine_tree': 88,
             'willow_tree': 89, 'bicycle': 90, 'bus': 91, 'motorcycle': 92, 'pickup_truck': 93, 'train': 94,
             'lawn_mower': 95, 'rocket': 96, 'streetcar': 97, 'tank': 98, 'tractor': 99}

idx2class = {v: k for k, v in class2idx.items()}
# class_to_idx_sorted = {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6,
#                        'beetle': 7,
#                        'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14,
#                        'camel': 15,
#                        'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 'chair': 20, 'chimpanzee': 21,
#                        'clock': 22,
#                        'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28,
#                        'dinosaur': 29,
#                        'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35,
#                        'hamster': 36,
#                        'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42,
#                        'lion': 43,
#                        'lizard': 44, 'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49,
#                        'mouse': 50,
#                        'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56,
#                        'pear': 57,
#                        'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63,
#                        'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70,
#                        'sea': 71,
#                        'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78,
#                        'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84,
#                        'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90,
#                        'trout': 91,
#                        'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97,
#                        'woman': 98,
#                        'worm': 99}
CIFAR100 = [
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
    'vehicles_2'
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


def save_model(model, args, test_acc, test_loss, train_acc, train_loss, save_path, task_id, epoch):
    """Saves model to file."""

    ckpt = {
        'args': args,
        'epoch': epoch,
        'tasks_seen': task_id,
        'train_accuracy': train_acc,
        'train_loss': train_loss,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'model': model.state_dict(),
    }

    # Save to file.
    torch.save(ckpt, save_path + '.pth')


def load_checkpoint(path):
    ckpt = torch.load(path)
    # model = ckpt['model']
    # model = model.cuda()
    return ckpt


# Custom Dataset
class CIFAR100_Dataset(DatasetFolder):
    def __init__(self, root, transform, num_tasks, multi_task=False, loader=pil_loader):
        self.multi_task = multi_task
        self.num_tasks = num_tasks
        super().__init__(root, loader,
                         ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
                         transform)
        self.targets = [s[1] for s in self.samples]

    def _find_classes(self, dir):
        classes = []
        # Handle class indices when loading multiple tasks
        if self.multi_task:
            for i in range(1, self.num_tasks):
                for c in task_to_classes[CIFAR100[i]]:
                    classes.append(c)
        else:
            classes = task_to_classes[CIFAR100[self.num_tasks]]
        class_to_idx = {k: class2idx[k] for k in classes}
        return classes, class_to_idx

    def __getitem__(self, index):
        # Get path to image and target label
        path, target = self.samples[index]
        # Load the image
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # Return both the image and the corresponding target
        return sample, target


def CIFAR_SuperClass_loader(task, batch_size, per_task_norm=False):
    """Return train and test data loaders for given superclass"""

    dl, ds = {}, {}
    train_path = f'../Data/cifar100_org/train/{task}'
    test_path = f'../Data/cifar100_org/test/{task}'
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
    # ds['train'] = datasets.ImageFolder(train_path, tsforms['train'])
    # ds['test'] = datasets.ImageFolder(test_path, tsforms['test'])
    num_task = CIFAR100.index(task)
    print(f"Task: {task}({num_task})")
    ds['train'] = CIFAR100_Dataset(train_path, tsforms['train'], num_task)
    ds['test'] = CIFAR100_Dataset(test_path, tsforms['test'], num_task)

    dl['train'] = torch.utils.data.DataLoader(ds['train'], batch_size=batch_size, shuffle=True,
                                              num_workers=4)
    dl['test'] = torch.utils.data.DataLoader(ds['test'], batch_size=min(500, len(ds['test'])), shuffle=False,
                                             num_workers=4)
    # for loader in dl.keys():
    #     print("Dataset:", loader, "\tSize:", len(dl[loader].dataset))
    return dl['train'], dl['test']


def MultiClassCrossEntropy(logits, labels, T, device):
    with torch.no_grad():
        # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
        outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
        labels = torch.softmax(labels / T, dim=1)

        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        outputs = -torch.mean(outputs, dim=0, keepdim=False)

        outputs = outputs.to(device)
    return outputs


def calculate_loss(preds, targets, loss_type, device, T=2.0, cls_scale=1, dist_scale=1):
    """
       Calculate Loss accoriding to the LwF paper
    """
    # with torch.autograd.detect_anomaly():
    preds = preds.to(device)
    targets = targets.to(device)

    if loss_type == "CrossEntropy":
        # print("CLS loss scale =", cls_scale)
        loss = nn.CrossEntropyLoss()
        return loss(preds, targets) * cls_scale

    elif loss_type == "Distillation":
        # print("Dist loss scale =", dist_scale)
        # with torch.no_grad():
        # https://discuss.pytorch.org/t/is-this-loss-function-for-knowledge-distillation-correct/69863
        #prob_t = F.softmax(targets / T, dim=1)
        #log_prob_s =F.log_softmax(preds / T, dim=1)
        #loss = -((F.log_softmax(preds / T, dim=1)) * (F.softmax(targets / T, dim=1))).sum(dim=1).mean()

        #loss = -(F.softmax(targets / T, dim=1) * F.log_softmax(preds / T, dim=1)).sum(dim=1).mean()
        #loss = F.kl_div(F.log_softmax((preds / T), dim=1), F.softmax((targets / T), dim=1),
                     #   reduction='batchmean')
        # loss = F.kl_div(F.log_softmax((preds / T), dim=1), F.softmax((targets / T), dim=1),
        #                 reduction='sum')
        #loss = nn.KLDivLoss(reduction='sum')(F.log_softmax(preds / T, dim=1), F.softmax(targets / T, dim=1))
        loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(preds / T, dim=1), F.softmax(targets / T, dim=1))
        #loss = nn.MSELoss(reduction="sum")(F.log_softmax((preds / T), dim=1), F.softmax((targets / T), dim=1))

        #loss = -torch.mean(torch.sum(preds * F.log_softmax(targets, dim=1), dim=1))
        return loss * dist_scale

        #     loss = F.kl_div(F.log_softmax((preds / T), dim=1), F.softmax((targets / T), dim=1),
        #                     reduction='batchmean', size_average=False)
        # #loss = MultiClassCrossEntropy(preds, targets, T,device)
        # return loss * dist_scale
        # return F.kl_div(F.log_softmax(preds / T), F.softmax(targets / T)) * scale

        # dist_loss = -torch.mean(torch.sum(preds * F.log_softmax(targets, dim=1), dim=1))

        # a = torch.isnan(preds)
        # b = torch.isnan(targets)

        # torch.autograd.set_detect_anomaly(True)
        # preds = F.softmax(preds, dim=1)
        # targets = F.softmax(targets, dim=1)
        #
        # preds = preds.pow(1 / T)
        # targets = targets.pow(1 / T)
        #
        # sum_preds = torch.sum(preds, dim=1)
        # sum_targets = torch.sum(targets, dim=1)
        #
        # sum_preds_ref = torch.transpose(sum_preds.repeat(preds.size(1), 1), 0, 1)
        # sum_preds_ref = sum_preds_ref.to(device)
        #
        # sum_targets_ref = torch.transpose(sum_targets.repeat(targets.size(1), 1), 0, 1)
        # sum_targets_ref = sum_targets_ref.to(device)
        #
        # preds = preds / sum_preds_ref
        # targets = targets / sum_targets_ref
        #
        # loss = torch.sum(-1 * preds * torch.log(targets), dim=1)
        # batch_size = loss.size()[0]
        # loss = torch.sum(loss, dim=0) / batch_size
        #
        # return loss * dist_scale


def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


# def distillation_loss(y, teacher_scores, T, scale):
#     """Computes the distillation loss (cross-entropy).
#        xentropy(y, t) = kl_div(y, t) + entropy(t)
#        entropy(t) does not contribute to gradient wrt y, so we skip that.
#        Thus, loss value is slightly different, but gradients are correct.
#        \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
#        scale is required as kl_div normalizes by nelements and not batch size.
#     """
#     return F.kl_div(F.log_softmax(preds / T), F.softmax(targets / T)) * scale


def record_preds(predictions):
    pred_counts = dict(Counter(predictions))
    class_counts = {}
    for idx in pred_counts.keys():
        class_counts[idx2class[idx]] = pred_counts[idx]

    print(class_counts)

    return class_counts
