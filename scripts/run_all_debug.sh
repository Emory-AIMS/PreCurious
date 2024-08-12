#!/bin/bash
#SBATCH --job-name=all_debug
#SBATCH --output=all_debug.log
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=a100-8-gm320-c96-m1152

TASK_NAME='PreCurious'
DATA='enron'

PEFT_PRE=full
LR_PRE=1e-4
PEFT=adapter
LR=1e-4

cd ../examples

model="gpt2"
python train_all.py \
    --dataset_name $DATA \
    --model_name_or_path $model \
    --per_device_eval_batch_size 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --task_name $TASK_NAME \
    --peft $PEFT --learning_rate $LR \
    --num_train_epochs 5 --only_eval_last \
    --peft_pre $PEFT_PRE --learning_rate_pre $LR_PRE \
    --pre_aux_ratio 1 --pre_num_epochs 3 \
    --do_ref_model --tune_ref --debug

python train_all.py \
    --dataset_name $DATA \
    --model_name_or_path $model \
    --per_device_eval_batch_size 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --task_name $TASK_NAME \
    --peft $PEFT --learning_rate $LR \
    --num_train_epochs 5 --only_eval_last \
    --attack warm \
    --peft_pre $PEFT_PRE --learning_rate_pre $LR_PRE \
    --pre_aux_ratio 1 --pre_num_epochs 3 \
    --do_ref_model --debug