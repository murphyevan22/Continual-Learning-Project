#!/bin/bash


DATASETS=( "$@" )
NAME=${DATASETS[-1]}
LOG="logs/cifar100_inference${NAME}.log"
echo "LOGFILE: ${LOG}"
unset DATASETS[-1]
printf ' ->%s\n' "${DATASETS[@]}"
#baseline_cifar100_acc="logs/baseline_cifar100_acc${NAME}.txt"
baseline_cifar100_acc='logs/baseline_cifar100_acc.txt'

GPU_ID=0
NETWORK_WIDTH_MULTIPLIER=1.0
ARCH='resnet18'
setting="scratch_mul_1.5_$NAME"

for TASK_ID in `seq 1 10`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_cifar100_main_normal.py \
        --arch $ARCH \
        --dataset ${DATASETS[TASK_ID]} --num_classes 5 \
        --load_folder checkpoints/CPG/$SETTING/$ARCH/${DATASETS[10]}/gradual_prune \
        --mode inference \
        --baseline_acc_file $baseline_cifar100_acc \
        --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
        --max_allowed_network_width_multiplier 1.5 \
        --log_path $LOG
done
