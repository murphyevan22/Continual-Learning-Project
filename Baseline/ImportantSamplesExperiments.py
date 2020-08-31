from BackBoneNet import *
import train_utils
import sys
import torch.nn as nn

if __name__ == '__main__':
    seed = 123
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)  # CPU seed
    if device == torch.device("cuda:0"):
        print("Cuda Seeding..")
        torch.cuda.manual_seed_all(seed)  # GPU seed
    random.seed(seed)  # python seed for image transformation
    np.random.seed(seed)
    # backbone_path = "/Users/evanm/Documents/College Downloads/Masters Project/Baseline/UpperBound/BackBone/backbone_net_Ep-199-Ac-77.02-ittr=4.pth"
    # important_samples_info= {
    #     'data_dir': "/Users/evanm/Documents/College Downloads/Masters Project/Data/cifar100_org/",
    #     'data_test_dir': "/Users/evanm/Documents/College Downloads/Masters Project/Data/CIFAR100/",
    #     'samples_dir': "/Users/evanm/Documents/College Downloads/Masters Project/Baseline/RandomSamples/",
    #      'samples_records_dir': "/Users/evanm/Documents/College Downloads/Masters "
    #                             "Project/Baseline/ImportantSamplesRecords/"
    #  }
    # random_samples_info = {
    #     'data_dir': "/Users/evanm/Documents/College Downloads/Masters Project/Data/cifar100_org/",
    #     'data_test_dir': "/Users/evanm/Documents/College Downloads/Masters Project/Data/CIFAR100/",
    #     'samples_dir': "/Users/evanm/Documents/College Downloads/Masters Project/Baseline/ImportantSamples/",
    #     'samples_records_dir': "/Users/evanm/Documents/College Downloads/Masters "
    #                            "Project/Baseline/ImportantSamplesRecords/"}
    backbone_path = "BackBone/backbone_net_Ep-82-Ac-60.35-FinalRun.pth"

    important_samples_info = {
        'data_dir': "../Data/cifar100_org/",
        'data_test_dir': "../Data/CIFAR100/",
        'samples_dir': "ImportantSamples/",
        'samples_records_dir': "ImportantSamplesRecords/",
        'random_samples': False}
    random_samples_info = {
        'data_dir': "../Data/cifar100_org/",
        'data_test_dir': "../Data/CIFAR100/",
        'samples_dir': "ImportantSamples/",
        'samples_records_dir': "ImportantSamplesRecords/",
        'random_samples': True}
    ep = 200
    # for idx, pc in enumerate([25, 50, 75, 100]):
    for idx, pc in enumerate([10, 20, 30, 50, 75]):
        important_samples_info['retain_percent'] = pc
        random_samples_info['retain_percent'] = pc
        exp_name = f"IMPT-{pc}_pc_retain"
        print("* " * 50)
        print("Using Important samples")
        print(f"Using {pc}% of data for training")
        print("* " * 50)

        train_utils.update_samples_CIFAR100(important_samples_info)

        print("Exp 1: Training RN18 from scratch")
        exp1 = BackBoneNet(exp_name=exp_name + "-EXP-1", epochs=ep, data_samples=True)
        exp1.data_dir = important_samples_info['data_dir']
        exp1.samples_dir = important_samples_info['samples_dir']
        exp1.data_test_dir = important_samples_info['data_test_dir']
        exp1.data_loaders = train_utils.samples_data_loader(exp1)
        exp1.train_model()

        print("Exp 2: FT Backbone (classifier only)")
        exp2 = BackBoneNet(exp_name=exp_name + "-EXP-2", epochs=ep, data_samples=True, backbone=backbone_path)
        exp2.data_dir = important_samples_info['data_dir']
        exp2.samples_dir = important_samples_info['samples_dir']
        exp2.data_test_dir = important_samples_info['data_test_dir']
        exp2.data_loaders = train_utils.samples_data_loader(exp2)

        for param in exp2.model.parameters():
            param.requires_grad = False
        exp2.model.fc = nn.Linear(512, 100)
        exp2.model = exp2.model.to("cuda:0")
        exp2.optimizer = torch.optim.SGD(exp2.model.parameters(), lr=exp2.lr, momentum=0.9, weight_decay=5e-4)
        exp2.fine_tune()

        print("Exp 3: FT Backbone (full network)")
        exp3 = BackBoneNet(exp_name=exp_name + "-EXP-3", epochs=ep, data_samples=True, backbone=backbone_path)
        exp3.data_dir = important_samples_info['data_dir']
        exp3.samples_dir = important_samples_info['samples_dir']
        exp3.data_test_dir = important_samples_info['data_test_dir']
        exp3.data_loaders = train_utils.samples_data_loader(exp3)
        # exp3.train_model()
        exp3.fine_tune()

        ## Random Samples

        exp_name = f"RAND-{pc}_pc_retain"
        print("* " * 50)
        print("Using Random samples")
        print(f"Using {pc}% of data for training")
        print("* " * 50)

        # random_samples = train_utils.get_random_samples(self=random_samples_info, percent=pc)
        train_utils.update_samples_CIFAR100(random_samples_info)

        print("Exp 1: Training RN18 from scratch")
        exp4 = BackBoneNet(exp_name=exp_name + "-EXP-1", epochs=ep, data_samples=True)
        exp4.data_dir = random_samples_info['data_dir']
        exp4.samples_dir = random_samples_info['samples_dir']
        exp4.data_test_dir = random_samples_info['data_test_dir']
        exp4.data_loaders = train_utils.samples_data_loader(exp4)
        exp4.train_model()

        print("Exp 2: FT Backbone (classifier only)")
        exp5 = BackBoneNet(exp_name=exp_name + "-EXP-2", epochs=ep, data_samples=True, backbone=backbone_path)
        exp5.data_dir = random_samples_info['data_dir']
        exp5.samples_dir = random_samples_info['samples_dir']
        exp5.data_test_dir = random_samples_info['data_test_dir']
        exp5.data_loaders = train_utils.samples_data_loader(exp5)

        for param in exp5.model.parameters():
            param.requires_grad = False
        exp5.model.fc = nn.Linear(512, 100)
        exp5.model = exp5.model.to("cuda:0")
        exp5.optimizer = torch.optim.SGD(exp5.model.parameters(), lr=exp5.lr, momentum=0.9, weight_decay=5e-4)
        exp5.fine_tune()

        print("Exp 3: FT Backbone (full network)")
        exp6 = BackBoneNet(exp_name=exp_name + "-EXP-3", epochs=ep, data_samples=True, backbone=backbone_path)
        exp6.data_dir = random_samples_info['data_dir']
        exp6.samples_dir = random_samples_info['samples_dir']
        exp6.data_test_dir = random_samples_info['data_test_dir']
        exp6.data_loaders = train_utils.samples_data_loader(exp6)
        # exp3.train_model()
        exp6.fine_tune()
    # train_utils.get_retained_samples(important_samples_info, 50)
