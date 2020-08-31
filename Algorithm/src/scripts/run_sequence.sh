#!/bin/bash
# Adds tasks in sequence using the iterative pruning + re-training method.
# Usage:
# ./scripts/run_sequence.sh ORDER PRUNE_STR LOADNAME GPU_IDS RUN_TAG EXTRA_FLAGS
# ./scripts/run_sequence.sh csf 0.75,0.75,-1 ../checkpoints/imagenet/imagenet_pruned_0.5_final.pt 3 nobias_1 
CIFAR=(
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
# This is hard-coded to prevent silly mistakes.
declare -A DATASETS
DATASETS["i"]="imagenet"
DATASETS["p"]="places"
DATASETS["s"]="stanford_cars_cropped"
DATASETS["c"]="cubs_cropped"
DATASETS["f"]="flowers"
DATASETS["CIFAR100"]="CIFAR100"

declare -A NUM_OUTPUTS
NUM_OUTPUTS["imagenet"]="1000"
NUM_OUTPUTS["places"]="365"
NUM_OUTPUTS["stanford_cars_cropped"]="196"
NUM_OUTPUTS["cubs_cropped"]="200"
NUM_OUTPUTS["flowers"]="102"
NUM_OUTPUTS["CIFAR100"]="5"

ORDER=$1
#LOADNAME="START"
LOADNAME="../checkpoints/imagenet/resnet18.pt"
GPU_IDS=0
RUN_TAG=$5
EXTRA_FLAGS=$6
TRAIN_EPOCHS=50
NUM_TASKS=20

for task_id in `seq 1 $NUM_TASKS`; do

  dataset=${CIFAR[task_id]}
  EXP_NAME="CIFAR100"
  #mkdir ../checkpoints/$dataset/$ORDER/
  #mkdir ../logs/$dataset/$ORDER/
  mkdir -p ../checkpoints/$EXP_NAME/$dataset/
  mkdir -p ../logs/$EXP_NAME/$dataset/

  # Get model to add dataset to.
  if [ $task_id -eq 1 ]
  then
    loadname=$LOADNAME
  else
    loadname=$prev_pruned_savename'_final.pt'
    if [ ! -f $loadname ]; then
        echo 'Final file not found! Using postprune'
        loadname=$prev_pruned_savename'_postprune.pt'
    fi
  fi

  # Prepare tags and savenames.
  #tag=$ORDER'_'$PRUNE_STR
  tag=$ORDER
  ft_savename=../checkpoints/$EXP_NAME/$dataset/$tag
  pruned_savename=../checkpoints/$EXP_NAME/$dataset/$tag'_pruned'
  logname=../logs/$EXP_NAME/$dataset/$tag

#if [ $task_id -eq 1 ]
#  then
#    echo "TASK 1"
  ##############################################################################
  # Initial Train on current dataset.
  ##############################################################################
  CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode train $EXTRA_FLAGS \
    --dataset $dataset --num_outputs ${NUM_OUTPUTS[$ORDER]} \
    --loadname $loadname --train_epochs $TRAIN_EPOCHS\
    --lr 0.1 --lr_decay_every 75 --lr_decay_factor 0.1 --save_prefix $ft_savename
#  else
#    echo "New FT Savename value"
#      ft_savename=../checkpoints/$EXP_NAME/$prev_ds/$tag'_pruned_final'
#fi
#    ##############################################################################
#  # Train on current dataset.
#  ##############################################################################
#  CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode train $EXTRA_FLAGS \
#    --dataset $dataset --num_outputs ${NUM_OUTPUTS[$ORDER]} \
#    --loadname $loadname \
#    --lr 1e-3 --lr_decay_every 10 --lr_decay_factor 0.1 --finetune_epochs 20 \
#    --save_prefix $ft_savename | tee $logname'.txt'

##############################################################################
  # Prune on current dataset.
  ##############################################################################
#  CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode prune $EXTRA_FLAGS \
#    --dataset $dataset --loadname $ft_savename'.pt' \
 #   --post_prune_epochs 10 \
  #  --lr 1e-3 --lr_decay_every 10 --lr_decay_factor 0.1 \
 #   --save_prefix $pruned_savename | tee $logname'_pruned.txt'

  ##############################################################################
  # Prune on current dataset.
  ##############################################################################
  CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode prune $EXTRA_FLAGS \
    --dataset $dataset --loadname $ft_savename'.pt' \
    --post_prune_epochs 25 --num_outputs ${NUM_OUTPUTS[$ORDER]}\
    --lr 1e-2 --lr_decay_every 10 --lr_decay_factor 0.1 \
    --save_prefix $pruned_savename

  prev_pruned_savename=$pruned_savename

  echo "Final Eval.... $pruned_savename"
  CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode eval $EXTRA_FLAGS \
    --dataset $dataset --loadname $pruned_savename'_final.pt' \
    --save_prefix $pruned_savename | tee $logname'_pruned_EVAL.txt'

  #exit 1
  #prev_ds=$dataset
  ./scripts/run_eval.sh $task_id

done
