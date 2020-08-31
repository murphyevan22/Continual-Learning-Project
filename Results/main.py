# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import glob

import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import csv

shuffled = ['None', 'large_carnivores', 'large_omnivores_and_herbivores', 'household_furniture',
            'household_electrical_devices', 'fish', 'non-insect_invertebrates', 'medium_mammals', 'flowers',
            'aquatic_mammals', 'fruit_and_vegetables', 'vehicles_2', 'reptiles', 'large_man-made_outdoor_things',
            'vehicles_1', 'people', 'insects', 'large_natural_outdoor_scenes', 'trees', 'food_containers',
            'small_mammals']
original = ['None', 'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
            'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things',
            'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores', 'medium_mammals',
            'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1',
            'vehicles_2']


def PackNet_Results():
    # Change dir
    # os.chdir('C:/Users/evanm/Documents/Masters/Thesis/CPG/Experiment1CPG')

    # Eval Files
    files = [f'PN/Eval_prune_0.{n}.txt' for n in range(15, 100, 10)]
    files.append("PN/Eval_prune_0.05.txt")

    PN_Acc = []
    # PN_Acc = pd.DataFrame()
    for f in files:
        with open(f) as file:
            for line in file:
                js = json.loads(line)
                PN_Acc.append(
                    pd.DataFrame.from_dict(js, orient='index', dtype=float, columns=[f.replace(".txt", "")[-2:]]))

    PN_Acc = pd.concat(PN_Acc, axis=1, names=[range(15, 100, 10), 5])
    PN_Acc = PN_Acc * 100
    print(PN_Acc.describe())
    print("05:", PN_Acc["05"].tolist())
    for a in range(15, 100, 10):
        print(str(a) + ":", PN_Acc[str(a)].tolist())
    # Display Results
    # **{"color": ['green', 'red']}
    PN_Acc.plot(kind="line")
    # plt.savefig("graphs/Fintune vs Baseline", bbox_inches='tight')
    plt.show()


def task_ordering():
    experiment = {}
    path = 'CPG/TaskOrdering/*.log'
    files = glob.glob(path)
    for f in files:
        with open(f) as file:
            name = f[41:-4]
            experiment[name] = {}
            csv_reader = csv.reader(file, delimiter=',')
            for idx, row in enumerate(csv_reader):
                experiment[name][idx + 1] = float(row[2][-5:])
                # print(row[2][-5:])

    experiment = pd.DataFrame.from_dict(experiment)
    experiment.head(20)
    # Display Results
    experiment.plot(kind="barh", )
    plt.title("Task Ordering Experiment")
    plt.savefig("graphs/Task Ordering", bbox_inches='tight')
    plt.show()


def rn18_vs_vgg():
    experiment = {"ResNet18": {}, "Vgg16": {}}
    path = 'C:/Users/evanm/Documents/College Downloads/Masters Project/CPG/Results/'

    # ResNet18
    with open(path + 'RN18_Server/cifar100_inference_rn18.log') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for idx, row in enumerate(csv_reader):
            experiment['ResNet18'][original[idx + 1]] = float(row[2][-5:])

    # Vgg16
    with open(path + 'VGG_basic/cifar100_inference.log') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for idx, row in enumerate(csv_reader):
            experiment['Vgg16'][original[idx + 1]] = float(row[2][-5:])

    experiment = pd.DataFrame.from_dict(experiment)
    experiment.head(20)
    # Display Results
    experiment.plot(kind="barh", figsize=(20, 15), **{"color": ['blue', 'red']})
    plt.title("CPG with RN18 vs Vgg16")
    plt.savefig("graphs/RN18vsVGG16", bbox_inches='tight')
    plt.show()


# colours = ['aqua', 'darkgoldenrod', 'darkseagreen', 'lightcoral', 'lightsalmon', 'deepskyblue',
#            'lime', 'royalblue', 'purple', 'red', 'rosybrown', 'turquoise', 'violet',
#            'indianred', 'indigo', 'orange', 'magenta', 'maroon', 'darkcyan',
#            'cadetblue', 'chartreuse', 'chocolate', 'coral', 'black', 'blanchedalmond', 'forestgreen']
mammels_class = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
fish_class = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']

from matplotlib.patches import Patch


def show_forgetting():
    experiment = {}
    path = 'LWF/*.txt'
    files = glob.glob(path)

    LWF_outputs = []
    # PN_Acc = pd.DataFrame()
    # for f in files[0]:
    f = files[1]
    f = "LWF/LwF_Results_CLS_SCALE_0.20.txt"
    print(f)
    with open(f) as file:
        for line in file:
            js = json.loads(line)
            js = pd.DataFrame.from_dict(js, dtype=int, orient='index')
            # LWF_outputs.append(js)
    scale = 0.2
    #scale = float(f[-8:-4])

    # LWF_outputs = pd.concat(LWF_outputs, axis=1, ignore_index=True)
    LWF_outputs = js
    LWF_outputs = LWF_outputs.T
    print(LWF_outputs.describe())

    aquatic = pd.DataFrame(LWF_outputs['aquatic_mammals'][:10])
    fish = pd.DataFrame(LWF_outputs['fish'][:15])

    # for idx, row in aquatic.iterrows():
    #     print(row)
    #     row = dict(row)['aquatic_mammals']
    #     size = len(row)
    #     row = pd.DataFrame.from_dict(row, orient="index")
    #     print(row.values)
    #     row.plot(kind="bar", figsize=(8, 8), **{'legend': None, "color": colours[:size]})
    #     plt.xticks(rotation='horizontal')
    #     plt.ylabel('Classes')
    #     plt.xlabel('Number of Predictions')
    #     plt.title(f"Epoch {idx+1}")
    #     plt.show()
    # colours = {'aquarium_fish': 'blue', 'beaver': 'r', 'dolphin': 'r', 'flatfish': 'blue', 'otter': 'r', 'ray': 'blue',
    #            'seal': 'r', 'shark': 'blue', 'trout': 'blue', 'whale': 'r'}
    rows = []
    for idx, row in fish.iterrows():
        row = dict(row)['fish']
        for k in (mammels_class + fish_class):
            if k not in row:
                row[k] = 0
        # for k in fish_class:
        #     if k not in row:
        #         row[k] = 0

        # cols = sorted(row.keys())
        # row = pd.DataFrame.from_dict(row, columns=cols)
        row = pd.DataFrame([row])
        row.index += 1
        row.columns = mammels_class + fish_class
        rows.append(list(row.iloc[0].values))
        # row = row.T
        # row.plot.bar(color=colours)

        # #plt.bar(row.columns,row.values , figsize=(8, 8), legend=None, colours=colours)

        # plt.show()
    # Display Results
    rows = pd.DataFrame(rows, columns=row.columns)
    rows.index += 1
    # # **{"color": ['green', 'red']}
    # for idx, row in rows.iterrows():
    rows.plot(kind="bar", color=['b', 'b', 'b', 'b', 'b', 'r', 'r', 'r', 'r', 'r'])
    plt.xticks(rotation='horizontal')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Predictions')
    plt.title(f"Cross Entropy Loss Scale Value: {scale}")
    # plt.legend(labels=["Task 1", "Task 2"], ncol=2)
    plt.legend([
        Patch(facecolor="b"),
        Patch(facecolor="r")
    ], ["Task 1", "Task 2"])
    plt.savefig(f"Graphs/LwF_scale_{scale}.png", bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == '__main__':
    # PackNet_Results()
    # task_ordering()
    # rn18_vs_vgg()
    show_forgetting()
