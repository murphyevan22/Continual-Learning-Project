#!/bin/bash


dataset=(
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
for one_shot_prune_perc in $(seq 0.05 0.1 0.95); do
echo "Running for Pruning % = $one_shot_prune_perc"
GPU_ID=0
#one_shot_prune_perc=0.6
arch='resnet18'
finetune_epochs=100
prune_epochs=30
WEIGHT_DECAY=5e-4
LOGFILE="logfile_prune_$one_shot_prune_perc.txt"
LOGFILE_EVAL="Eval_prune_$one_shot_prune_perc.txt"
for task_id in `seq 1 20`; do

  # Finetune tasks
  if [ "$task_id" != "1" ]
  then
      CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
          --arch $arch \
          --dataset ${dataset[task_id]} --num_classes 5 \
          --lr 1e-2 \
          --weight_decay $WEIGHT_DECAY \
          --save_folder checkpoints/PackNet/experiment1/$arch/$one_shot_prune_perc/${dataset[task_id]}/scratch \
          --load_folder checkpoints/PackNet/experiment1/$arch/$one_shot_prune_perc/${dataset[task_id-1]}/one_shot_prune \
          --logfile checkpoints/PackNet/experiment1/$arch/$LOGFILE \
          --epochs $finetune_epochs \
          --mode finetune
  else
      CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
          --arch $arch \
          --dataset ${dataset[task_id]} --num_classes 5 \
          --lr 1e-2 \
          --weight_decay $WEIGHT_DECAY \
          --save_folder checkpoints/PackNet/experiment1/$arch/$one_shot_prune_perc/${dataset[task_id]}/scratch \
          --logfile checkpoints/PackNet/experiment1/$arch/$LOGFILE \
          --epochs $finetune_epochs \
          --mode finetune
  fi

  # Prune tasks
  CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
      --arch $arch \
      --dataset ${dataset[task_id]} --num_classes 5 \
      --lr 1e-3 \
      --weight_decay $WEIGHT_DECAY \
      --save_folder checkpoints/PackNet/experiment1/$arch/$one_shot_prune_perc/${dataset[task_id]}/one_shot_prune \
      --load_folder checkpoints/PackNet/experiment1/$arch/$one_shot_prune_perc/${dataset[task_id]}/scratch \
      --logfile checkpoints/PackNet/experiment1/$arch/$LOGFILE \
      --epochs $prune_epochs \
      --mode prune \
      --one_shot_prune_perc $one_shot_prune_perc
done

# Evaluate tasks
echo "Evaluating on Tasks..."
for history_id in `seq 1 20`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
        --arch $arch \
        --dataset ${dataset[history_id]} --num_classes 5 \
        --load_folder checkpoints/PackNet/experiment1/$arch/$one_shot_prune_perc/${dataset[20]}/one_shot_prune \
        --logfile checkpoints/PackNet/experiment1/$arch/$LOGFILE_EVAL \
        --mode inference
done

done