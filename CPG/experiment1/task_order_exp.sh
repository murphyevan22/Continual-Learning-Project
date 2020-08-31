#!/bin/bash
#!C:\Users\evanm\anaconda3\pkgs\m2-bash-4.3.042-5\Library\usr\bin\bash.exe


#dataset_original=(
#    'None'                # dummy
#    'aquatic_mammals'
#    'fish'
#    'flowers'
#    'food_containers'
#    'fruit_and_vegetables'
#    'household_electrical_devices'
#    'household_furniture'
#    'insects'
#    'large_carnivores'
#    'large_man-made_outdoor_things'
#    'large_natural_outdoor_scenes'
#    'large_omnivores_and_herbivores'
#    'medium_mammals'
#    'non-insect_invertebrates'
#    'people'
#    'reptiles'
#    'small_mammals'
#    'trees'
#    'vehicles_1'
#    'vehicles_2'
#)
#dataset=(
#    'None'                # dummy
#    'aquatic_mammals'
#    'fish'
#    'flowers'
#    'food_containers'
#    'fruit_and_vegetables'
#    'household_electrical_devices'
#    'household_furniture'
#    'insects'
#    'large_carnivores'
#    'large_man-made_outdoor_things'
#    'large_natural_outdoor_scenes'
#    'large_omnivores_and_herbivores'
#    'medium_mammals'
#    'non-insect_invertebrates'
#    'people'
#    'reptiles'
#    'small_mammals'
#    'trees'
#    'vehicles_1'
#    'vehicles_2'
#)
dataset_original=(
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
)
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
)

# now loop through the above array
for x in 1 3 6 9
do
  echo "**********************************************************************"
  test_class="${dataset_original[$x]}"

for i in 1 3 6 9
do
  # Name Experiment
  name="_task_${dataset_original[$x]}""_pos_$i"
  echo "Experiment: $name"


  # Change ordering of tasks each itteration
  dataset=("${dataset_original[@]}")
  temp_class="${dataset_original[$i]}"
  dataset[$x]="$temp_class"
  echo "Moving test_class ($test_class) to position $i, for class: $temp_class"
  dataset[$i]="$test_class"

  #echo "Running Baseline Script..."
  #./experiment1/baseline_cifar100.sh "${dataset[@]}" ${name}

  echo "Running Scratch Mul Script..."
  ./experiment1/CPG_cifar100_scratch_mul_1.5.sh "${dataset[@]}" "${name}"

  echo "Running Inference Script..."
  ./experiment1/inference_cifar_task_ordering.sh "${dataset[@]}" "${name}"

done
done