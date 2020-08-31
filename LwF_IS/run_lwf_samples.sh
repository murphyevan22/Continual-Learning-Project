#!/bin/bash

TASKS=20
EPOCHS=50

echo "Running LwF on CIFAR100 (ResNet18)"
for pc in 15 25 45 ; do
  EXP="LwF_Retain_PC_$pc"
  echo "Begining exp: $EXP"
  python main.py --num_tasks $TASKS --num_epochs $EPOCHS --outfile $EXP'.csv' --retain_percent $pc
  echo "Finished exp: $EXP"
done

echo "Running LwF on CIFAR100 (ResNet18)"
for pc in 15 25 45 ; do
  EXP="LwF_Retain_PC_Rand_$pc"
  echo "Begining exp: $EXP"
  python main.py --num_tasks $TASKS --num_epochs $EPOCHS --outfile $EXP'.csv' --random_samples True --retain_percent $pc
  echo "Finished exp: $EXP"
done

#echo "Running LwF on CIFAR100 (ResNet18)"
#for pc in $(seq 5 10 60) ; do
#  EXP="LwF_Retain_PC_Best_wts_$pc"
#  echo "Begining exp: $EXP"
#  python main.py --num_epochs $EPOCHS --outfile $EXP'.csv' --select_best_weights True --retain_percent $pc
#  echo "Finished exp: $EXP"
#
#done
#
