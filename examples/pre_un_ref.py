#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import datasets
import argparse
import logging
import math
import os
import copy 
import sys
import json
from utils import (
    count_parameters,
    setup_peft,
    load_model_only,
    load_tokenizer,
    add_canary,
    load_raw_data,
    preprocess_data,
    load_aux_data,
    set_dropout,
    eval_ppl,
    get_exposure,
    get_fit_canary_loss,
    loss_to_ppl

)
from datetime import datetime
from collections import deque

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import transformers
from accelerate import Accelerator, DistributedType
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    GPT2LMHeadModel,
)
from transformers.utils.versions import require_version
import datasets
import numpy as np
import random

import attack_loop
from datasets import load_from_disk

transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    # basic
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--do_ref_model",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    # parser.add_argument(
    #     "--learning_rate",
    #     type=float,
    #     default=5e-5,
    #     help="Initial learning rate (after the potential warmup period) to use.",
    # )
    # parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    # parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
       choices=[],
    )
    parser.add_argument('--local_home', type=str, default='/local/scratch2/rliu51/')
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )

    # logging
    parser.add_argument("--debug", action="store_true", help='if debug')

    # canary exposure
    parser.add_argument("--add_canary", action="store_true", help = "If true, then add canaries in the dataset.")
    parser.add_argument("--canary_rep", default=None, type = int, help = "The repetition of each canary")
    parser.add_argument("--canary_len", default = 5, type = int, help = "The len of digit of canaries")

    # PEFT
    '''
        addictive-adapter: adapter
        addictive-soft prompt: (prompt), prefix
        addictive-others: ia3
        selective: head, layers, bitfit
        re-parameterize: lora
        hybrid: compactor
    '''
    # parser.add_argument(
    #     "--peft",
    #     type=str,
    #     default=None,
    #     choices=['full', 'head', 'layers', 'bitfit', 'adapter', 'prefix', 'compacter', 'lora', 'ia3']
    # )
    parser.add_argument(
        "--peft_pre",
        type=str,
        default='none',
        choices=['full', 'layers', 'bitfit', 'lora', 'ia3', 'head', 'adapter']
    )
    parser.add_argument(
        "--adapter_reduction",
        type=int,
        default=16,
        help="whether to add adapter or not",
    )
    parser.add_argument("--lora_r", default=16, type=int, help='rank for lora')
    parser.add_argument("--train_layer_n_only",default=None, nargs="+", help = "If true, only true the indicated layer indexes of the model.")

    # pre curious
    parser.add_argument("--attack", default=None, type=str)
    parser.add_argument("--aux_dir", default=None, type=str, help='aux directory, setup for dup and dedup')
    parser.add_argument("--mode", default=None, type=str)
    parser.add_argument("--p", type=float, default=None)

    parser.add_argument("--pre_num_epochs", type=int, default=1)
    parser.add_argument("--pre_aux_ratio", type=float, default=1, help='the portion of auxiliary dataset')
    parser.add_argument("--sim_k", type=int, default=3)
    parser.add_argument(
        "--learning_rate_pre",
        type=float,
        default=None,
        help="learning_rate_pre / learning_rate ",
    )
    parser.add_argument("--dropout_pre", default=None, type=float, help='suggest 0.2')
    parser.add_argument("--weight_decay_pre", default=0.0, type=float, help='suggest 0.01')
    parser.add_argument("--clip_pre", default=None, type=float, help='default 1.0')
    parser.add_argument("--drop_head", action="store_true", help="by default not drop pre-trained head")

    parser.add_argument('--split_dir', type=str, default='../pre_data/', help='folder for saving data split')
    parser.add_argument('--data_seed', default=1234, type=int)

    parser.add_argument('--save_ckpt', default=[], type=int, nargs="+", help="mid epoch for saving ckpt")

    parser.add_argument('--task_name', default='PreCurious')

    args = parser.parse_args()

    # Sanity checks
    assert args.dataset_name is not None
    assert args.model_name_or_path is not None

    if args.aux_dir == 'none':
        args.aux_dir = None
    args.output_dir = os.path.join('../cache/output/ckpt_pre/')
 
    return args

def train_pre_adv(
        model,
        args,
        accelerator,
        aux_dataloader,
        eval_dataloader,
        pre_metric,
    ):
    curious_loss_fn = getattr(attack_loop, f'loss_{args.attack}')

    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay_pre,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate_pre)

    # Prepare everything with our `accelerator`.
    model, optimizer, aux_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, aux_dataloader, eval_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(aux_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.pre_num_epochs * num_update_steps_per_epoch
    num_steps_per_log = math.ceil(num_update_steps_per_epoch/5)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running Pre-Curious training *****")
    logger.info(f"  Num examples = {len(aux_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.pre_num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_loss = 1000000
    logger.info(torch.cuda.is_available())
    logger.info(model.device)

    pre_metric['val_ppl'] = []
    pre_metric['gen_gap'] = []
    pre_metric['loss'] = []
    for epoch in range(args.pre_num_epochs):
        model.train()
        running_loss = deque()
        logger.info(f"pre-curious training epoch {epoch}")
        for step, batch in enumerate(aux_dataloader):
            loss = curious_loss_fn(args, model, batch, running_loss)
            accelerator.backward(loss / args.gradient_accumulation_steps)
            if step % args.gradient_accumulation_steps == 0 or step == len(aux_dataloader) - 1:
                if args.clip_pre:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_pre)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

            if completed_steps % num_steps_per_log == 0:
                pre_loss = np.mean(running_loss)
                logger.info(f"pre_loss: {pre_loss:.4f}")
            
            if completed_steps >= args.max_train_steps:
                break
        
        perplexity = eval_ppl(model, args, eval_dataloader=eval_dataloader)
        pre_metric['val_ppl'].append(perplexity)
        perplexity_aux = eval_ppl(model, args, eval_dataloader=aux_dataloader)
        pre_metric['gen_gap'].append(perplexity - perplexity_aux)
        pre_metric['loss'].append(np.mean(running_loss))
        if epoch + 1 in args.save_ckpt:
            trainable_keys = [name for name, param in model.named_parameters() if param.requires_grad]
            trainable_params = {name: param for name, param in model.state_dict().items() if name in trainable_keys}
            ckpt = {
                'model_state_dict': trainable_params,
                'aux_loss': pre_metric['loss'][-1],
                'val_ppl': pre_metric['val_ppl'][-1],
                'gen_gap': pre_metric['gen_gap'][-1],
            }
            model_path = args.model_folder + f'/ckpt_{epoch+1}.pth'
            torch.save(ckpt, model_path)
            logger.info(f'Saving model to {model_path}')
            
def main():

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    
    ################## logging ##################
    args = parse_args()

    peft_name = args.peft_pre

    att_name = args.attack
    
    folder_name = f"pre_{args.model_name_or_path}_{args.dataset_name}_aux{args.pre_aux_ratio}_{peft_name}_{att_name}_pe{args.pre_num_epochs}_f{args.mode}_p{args.p}_plr{args.learning_rate_pre}_s{args.seed}"

    if args.add_canary:
        folder_name = f"canary_{str(args.canary_rep)}_{str(args.canary_len)}" + folder_name

    if args.debug:
        folder_name = 'DEBUG' + folder_name

    if args.aux_dir:
        folder_name = folder_name + 'DEDUP' + args.aux_dir.split('/')[-1].split('_')[-2]
    if len(args.save_ckpt) != 0:
        folder_name = 'SAVE' + folder_name
    
    directory = "{}{}".format(args.output_dir,folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    log_file = os.path.join(directory, "stdout")
    metric_file = os.path.join(directory, "metric.json")
    model_folder = os.path.join(directory, "model")
    if os.path.exists(log_file):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = f"{log_file.replace('.log', '')}_{timestamp}.log"
        metric_file = f"{metric_file.replace('.json', '')}_{timestamp}.json"
        model_folder = f"{model_folder}_{timestamp}"
    args.model_folder = model_folder
    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    file_handler = logging.FileHandler(log_file, mode='w')
    logger.addHandler(file_handler)
    if accelerator.is_local_main_process:
        logger.info("Logging to {}".format(log_file))

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the model saving path
    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process:
       logger.info(str(args))


    ############### set up the tokenizer ###############
    tokenizer = load_tokenizer(args)

    ################## dataset ##################
    raw_datasets = load_raw_data(args, logger)

    # #### add canary to train set if need ####
    # if args.add_canary:
    #     raw_datasets, fitting_canaries_ids, canary_ids = add_canary(args, tokenizer, raw_datasets, directory, logger)
    # else:
    #     fitting_canaries_ids = None
    #     canary_ids = None

    #### Preprocessing the datasets ####
    train_dataset, eval_dataset = preprocess_data(args, raw_datasets, tokenizer, accelerator, logger)
    if args.debug:
        args.num_train_epochs = 1
        subset_ids = list(range(20))
        train_dataset = Subset(train_dataset, subset_ids)
        subset_ids = list(range(10))
        eval_dataset = Subset(eval_dataset, subset_ids)

    ###### split for aux dataset ####
    aux_dataset, _train_dataset, _non_dataset = load_aux_data(args, train_dataset, logger)
    if args.aux_dir:
        aux_dataset = load_from_disk(args.aux_dir)
        aux_dataset = Subset(aux_dataset, np.arange(0, len(aux_dataset)))
        aux_dataset = aux_dataset.dataset

    # dataloader
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    # )
    # non_dataloader = DataLoader(
    #     non_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    # )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    aux_dataloader = DataLoader(
        aux_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )

    all_metric = {'pre': {}}

    ################## pre-curious ##################
    assert args.attack is not None
    # assert args.peft_pre == 'full'
    model = load_model_only(args, tokenizer, logger)

    if args.peft_pre in ['full', 'head']:
        pass
    else:
        # add adapters (adapter/lora/etc)
        setup_peft(model, args.peft_pre, args)
        # inverse the trainable and frozen part
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.requires_grad = False
                logger.info(f'Freezing PEFT parameters {n}')
            else:
                p.requires_grad = True
        if args.peft_pre == 'adapter':
            seed_check=model.transformer.h[0].output_adapters.adapters.adapter.adapter_down[0].weight.sum()
            logger.info(f'SEED_CHECK={seed_check}')
        elif args.peft_pre == 'lora':
            seed_check=model.transformer.h[1].attn.c_attn.loras.lora.lora_A.sum()
            logger.info(f'SEED_CHECK={seed_check}')
    count_parameters(args, logger, model, prefix='pre')
    if args.dropout_pre:
        set_dropout(model, args.dropout_pre)
    train_pre_adv(
        model,
        args,
        accelerator,
        aux_dataloader,
        eval_dataloader,
        all_metric['pre']
    )
    # pre-curious recover 
    if args.dropout_pre is not None:
        set_dropout(model, 0.1) # default dropout is 0.1

    with open(metric_file, 'w') as f:
        json.dump(all_metric, f)
        logger.info(f"Saving metrics to {metric_file}")

    if len(args.save_ckpt) == 0:
        trainable_keys = [name for name, param in model.named_parameters() if param.requires_grad]
        trainable_params = {name: param for name, param in model.state_dict().items() if name in trainable_keys}
        ckpt = {
            'model_state_dict': trainable_params,
            'aux_loss': all_metric['pre']['loss'][-1],
            'val_ppl': all_metric['pre']['val_ppl'][-1],
            'gen_gap': all_metric['pre']['gen_gap'][-1],
        }
        model_path = args.model_folder + f'/ckpt_{args.pre_num_epochs}.pth'
        torch.save(ckpt, model_path)
        logger.info(f'Saving model to {model_path}')

if __name__ == "__main__":
    main()
