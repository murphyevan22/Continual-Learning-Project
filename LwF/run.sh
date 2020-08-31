#!/bin/bash

TASKS=10
EPOCHS=50

echo "Running LwF on CIFAR100 (ResNet18)"
for SCALE in 3 4 5; do
  EXP="LwF_Results_METRIC_CLS_SCALE_$SCALE"
  echo "Begining exp: $EXP"
  python main.py --num_epochs $EPOCHS --num_tasks $TASKS --outfile $EXP'.csv' --cls_loss_scale $SCALE
  echo "Finished exp: $EXP"
done


#echo "Running LwF on CIFAR100 (ResNet18)"
#for LR in 1 0.1 0.01 ; do
#  EXP="LwF_results_LR_$LR"
#  echo "Begining exp: $EXP"
#  python main.py --num_epochs $EPOCHS --outfile $EXP'.csv' --init_lr $LR
#  echo "Finished exp: $EXP"
#done
#
##
#TASKS=10
###EPOCHS=50
#echo "Running LwF on CIFAR100 (ResNet18)"
#for EP in $(seq 10 10 50) ; do
#  EXP="LwF_Results_EP__$EP"
#  echo "Begining exp: $EXP"
#  python main.py --num_epochs $EP --num_tasks $TASKS --outfile $EXP'.csv' --init_lr 0.01
#  echo "Finished exp: $EXP"
#done
#
#python show_forgetting.py --num_epochs 50 --outfile 'final_forgetting_exp.csv' --init_lr 0.01
#echo "Running LwF on CIFAR100 (ResNet18)"
#for ITTR in 1 2 3 4 5 ; do
#  EXP="LwF_show_forgetting_ittr_$ITTR"
#  echo "Begining exp: $EXP"
#  python show_forgetting.py --num_epochs $EPOCHS --outfile $EXP'.csv' --init_lr 0.01
#  echo "Finished exp: $EXP"
#
#done

#echo "Running LwF on CIFAR100 (ResNet18)"
#for SCALE in $(seq 0.6 0.1 1.2) ; do
#  EXP="LwF_Results_METRIC_CLS_SCALE_$SCALE"
#  echo "Begining exp: $EXP"
#  python main.py --num_epochs $EPOCHS --num_tasks $TASKS --outfile $EXP'.csv' --cls_loss_scale $SCALE
#  echo "Finished exp: $EXP"
#done

#python main.py --num_epochs $EPOCHS --num_tasks $TASKS --outfile 'bst_sts_test.csv' --cls_loss_scale 0.05




#echo "Running LwF on CIFAR100 (ResNet18)"
#for SCALE in $(seq 0.1 0.1 1) ; do
#  EXP="LwF_Results_DIST_SCALE_$SCALE"
#  echo "Begining exp: $EXP"
#  python main.py --num_epochs $EPOCHS --num_tasks $TASKS --outfile $EXP'.csv' --dist_loss_scale $SCALE
#  echo "Finished exp: $EXP"
#done
#
#
#echo "Running LwF on CIFAR100 (ResNet18)"
#for SCALE in $(seq 0.1 0.1 1) ; do
#  EXP="LwF_Results_REG_SCALE_BEST_wts_$SCALE"
#  echo "Begining exp: $EXP"
#  python main.py --num_epochs $EPOCHS --num_tasks $TASKS --outfile $EXP'.csv' --select_best_weights True --cls_loss_scale $SCALE
#  echo "Finished exp: $EXP"
#done
#


#echo "Running LwF on CIFAR100 (ResNet18)"
#for ALPHA in $(seq 0 0.1 1) ; do
#  EXP="LwF_Results_AlPHA_$ALPHA"
#  echo "Begining exp: $EXP"
#  python main.py --num_epochs $EPOCHS --outfile $EXP'.csv' --loss_alpha $ALPHA
#  echo "Finished exp: $EXP"
#
#done
#
#echo "Running LwF on CIFAR100 (ResNet18)"
#for LR in 1 0.5 0.1 0.05 0.01 0.001 ; do
#  EXP="LwF_Results_LR_$LR"
#  echo "Begining exp: $EXP"
#  python main.py --num_epochs $EPOCHS --outfile $EXP'.csv' --init_lr $LR
#  echo "Finished exp: $EXP"
#
#done
#
#echo "Running LwF on CIFAR100 (ResNet18)"
#for T in $(seq 0 0.5 3); do
#  EXP="LwF_Results_T_$T"
#  echo "Begining exp: $EXP"
#  python main.py --num_epochs $EPOCHS --outfile $EXP'.csv' --T $T
#  echo "Finished exp: $EXP"
#
#done
#
#echo "Running LwF on CIFAR100 (ResNet18)"
#for ALPHA in $(seq 0 0.1 1) ; do
#  EXP="LwF_Results_AlPHA_ES_$ALPHA"
#  echo "Begining exp: $EXP"
#  python main.py --num_epochs $EPOCHS --outfile $EXP'.csv' --loss_alpha $ALPHA --early_stop True
#  echo "Finished exp: $EXP"
#
#done


##TASKS=20
#echo "Running LwF on CIFAR100 (ResNet18)"
#for TASKS in $(seq 2 20) ; do
#  EXP="LwF_show_forgetting_T$TASKS"
#  echo "Begining exp: $EXP"
#  python show_forgetting.py --num_epochs $EPOCHS --outfile $EXP'.csv' --num_tasks $TASKS
#  echo "Finished exp: $EXP"
#
#done