#!/bin/bash
#!C:\Users\evanm\anaconda3\pkgs\m2-bash-4.3.042-5\Library\usr\bin\bash.exe
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
SEEN=$1
current_ds=${CIFAR[$SEEN]}
echo "Current Dataset: $current_ds"
#cd ../
echo "Evaluate on all tasks seen"
for task_id in `seq 1 $SEEN`; do

  dataset=${CIFAR[task_id]}

  echo "Evaluate on $dataset ($task_id)"
  CUDA_VISIBLE_DEVICES=0 python main.py --mode eval \
    --dataset $dataset --loadname ../checkpoints/CIFAR100/$current_ds/CIFAR100_pruned_final.pt \
    --save_prefix 'testing_eval'| tee ../logs/CIFAR100/$current_ds/$dataset'_final_eval.txt'

done

