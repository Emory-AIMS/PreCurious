#!/bin/bash
#SBATCH --job-name=all_main
#SBATCH --output=all_main.log
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=a100-8-gm320-c96-m1152

# srun --mem=20G --gpus=1 --nodes=1 --ntasks=1 --partition a10g-8-gm192-c192-m768

DATA='enron'

PEFT_PRE=full
LR_PRE=1e-4
PEFT=adapter
LR=1e-4

cd ../examples

# MODELS="gpt2 gpt2-medium gpt2-large"
MODELS="gpt2"
for model in $MODELS; do
    python train_all.py \
        --dataset_name $DATA \
        --model_name_or_path $model \
        --per_device_eval_batch_size 2 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --peft $PEFT --learning_rate $LR \
        --num_train_epochs 5 --only_eval_last \
        --attack warm \
        --peft_pre $PEFT_PRE --learning_rate_pre $LR_PRE \
        --pre_aux_ratio 1 --pre_num_epochs 3 \
        --do_ref_model

    python train_all.py \
        --dataset_name $DATA \
        --model_name_or_path $model \
        --per_device_eval_batch_size 2 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --peft $PEFT --learning_rate $LR \
        --num_train_epochs 5 --only_eval_last \
        --peft_pre $PEFT_PRE --learning_rate_pre $LR_PRE \
        --pre_aux_ratio 1 --pre_num_epochs 3 \
        --do_ref_model --tune_ref
done