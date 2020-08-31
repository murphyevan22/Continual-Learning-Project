#!/bin/bash


DATASETS=(
    'None'                # dummy
    'aquatic_mammals'
    'fish'
    'flowers'
    'food_containers'
    'fruit_and_vegetables'
    'household_electrical_devices'
    'household_furniture'
    'insects'
    'large_carnivores'
    'large_man-made_outdoor_things'
    'large_natural_outdoor_scenes'
    'large_omnivores_and_herbivores'
    'medium_mammals'
    'non-insect_invertebrates'
    'people'
    'reptiles'
    'small_mammals'
    'trees'
    'vehicles_1'
    'vehicles_2'
)

GPU_ID=0
NETWORK_WIDTH_MULTIPLIER=1.0
#ARCH='resnet18'
ARCH='resnet18_better_with_wd_5e4'

#SETTING='scratch_mul_1.5__task_aquatic_mammals_pos_9'
SETTING='scratch_mul_1.5'
NUM_TASKS=10

for TASK_ID in `seq 1 $NUM_TASKS`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_cifar100_main_normal.py \
        --arch $ARCH \
        --dataset ${DATASETS[TASK_ID]} --num_classes 5 \
        --load_folder checkpoints/CPG/experiment1/$SETTING/$ARCH/${DATASETS[$NUM_TASKS]}/gradual_prune \
        --mode inference \
        --baseline_acc_file logs/baseline_cifar100_acc.txt \
        --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
        --max_allowed_network_width_multiplier 1.5 \
        --log_path logs/cifar100_inference.log
done
