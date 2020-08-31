#!/bin/bash
#!C:\Users\evanm\anaconda3\pkgs\m2-bash-4.3.042-5\Library\usr\bin\bash.exe

cd ../
echo "Running Imagenet Pruning on RN18..."
#./scripts/run_imagenet_pruning.sh 0 resnet18 0.5 1
echo "Finished Imagenet Pruning on RN18!"

echo "Running Baseline Finetuning on RN18..."

#./scripts/run_baseline_finetuning.sh 0 CIFAR100 all 1e-3 20 20 1 resnet18

echo "Finished Baseline Finetuning on RN18!"

echo "Running Sequence on RN18..."
ARCH="resnet18"
#python main.py --arch $ARCH --init_dump
 ./scripts/run_sequence.sh CIFAR100

echo "Finished Sequence on RN18!"

