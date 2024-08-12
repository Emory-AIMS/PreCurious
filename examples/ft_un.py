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
import re
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
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
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
    parser.add_argument('--task_name', type=str, default='PreCurious')
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
    parser.add_argument("--log_per_step", type=int, default=2)

    # 添加参数
    parser.add_argument('--prefix_len', type=int, default=100, help='Length of the prefix.')
    parser.add_argument('--prefix_mode', type=str, default='aux_dup', choices=['train_dup', 'aux_dup'], help='Mode of the prefix.')
    parser.add_argument('--prefix_num', type=int, default=100, help='Number of prefixes.')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum frequency for filtering.')

    parser.add_argument('--suffix_len', type=int, default=512, help='Length of the suffix.')
    parser.add_argument('--sampling_k', type=int, default=50, help='Top-k sampling parameter. Set None for greedy.')
    parser.add_argument('--num_gen', type=int, default=1000, help='Number of generations.')
    parser.add_argument('--threshold_token', type=int, default=50, help='Token threshold for filtering.')
    parser.add_argument('--k', type=int, default=100, help='Top-k filter for final outputs.')

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
    parser.add_argument(
        "--peft",
        type=str,
        default=None,
        choices=['full', 'head', 'layers', 'bitfit', 'adapter', 'prefix', 'compacter', 'lora', 'ia3']
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
    parser.add_argument("--drop_head", action="store_true", help="by default not drop pre-trained head")
    parser.add_argument("--bad", action="store_true", help="add additional bad tricks")
    parser.add_argument("--bad_scale", type=float, default=10, help="add additional bad tricks")
    parser.add_argument("--bad_key", type=str, default='.mlp.c_proj.bias', help="add additional bad tricks")
    parser.add_argument("--untied_head", action="store_true")
    parser.add_argument("--fresh_head", action="store_true", help="random init dropped head, otherwise use old head before pre-curious")
    parser.add_argument("--pre_aux_ratio", type=float, default=1, help='the portion of auxiliary dataset')

    # defense # TODO: add dpsgd
    parser.add_argument("--dropout_def", default=None, type=float)
    parser.add_argument("--weight_decay_def", default=0.0, type=float)
    parser.add_argument("--clip_def", default=None, type=float)

    parser.add_argument("--only_eval_last", action="store_true")
    parser.add_argument("--load_ref_file", type=str, default=None, help="load a reference model, none for benign; ends with ckpt.pth")
    parser.add_argument("--load_ref_name", type=str, default='benign')
    parser.add_argument("--load_init_file", type=str, default=None, help="load model intialization, none for benign; ends with ckpt.pth")
    parser.add_argument("--load_init_name", type=str, default='benign') 

    parser.add_argument('--split_dir', type=str, default='../pre_data/', help='folder for saving data split')
    parser.add_argument('--data_seed', default=1234, type=int)
    parser.add_argument('--save_ft', action="store_true", help='save final ft ckpt')
 
    args = parser.parse_args()

    args.output_dir = os.path.join('../cache/output/ckpt_ft/')

    # Sanity checks
    assert args.dataset_name is not None
    assert args.model_name_or_path is not None

    if args.load_ref_file == 'none':
        args.load_ref_file = None
    if args.load_init_file == 'none':
        args.load_init_file = None

    # 使用正则表达式来匹配 "aux" 后面的数字
    if args.load_ref_file is not None:
        match = re.search(r'aux(\d+\.\d+)', args.load_ref_file)
        assert match
        args.pre_aux_ratio = float(match.group(1))
    
    if args.load_init_file is not None:
        match = re.search(r'aux(\d+\.\d+)', args.load_init_file)
        assert match
        pre_aux_ratio = float(match.group(1))
        assert pre_aux_ratio == args.pre_aux_ratio

    return args

def priv_measure(
        model,
        accelerator,
        args,
        eval_dataloader,
        train_dataloader,
        non_dataloader,
        epoch,
        fitting_canaries_ids=None,
        canary_ids=None,
        model_ref=None,
):
    logger.info(f"************* epoch {epoch} priv_measure *************")

    model.eval()
    if args.do_ref_model:
        model_ref.eval()
    
    # init metric dict
    fpr_list = [0.1, 0.01, 0.001, 0.0001]
    metric_unit = {
            'roc_curve': {
                'fpr': [],
                'tpr': [],
                'roc_auc': None
            },
            'pr_curve': {
                'p': [],
                'r': [],
                'pr_auc': None,
            }
        }
    for fpr in fpr_list:
        metric_unit[f'fpr{fpr}'] = {
                            'threshold': None,
                            'recall': None,
                            }

    mia_metric = {
        'mem': {
            'losses': [],
            'diff_losses_ref': [],
            'diff_losses_aug_avg': [],
        },
        'non_mem': {
            'losses': [],
            'diff_losses_ref': [],
            'diff_losses_aug_avg': [],
        },
        'loss_mia': copy.deepcopy(metric_unit),
    }
    if args.do_ref_model:
        mia_metric['ref_model_mia'] = copy.deepcopy(metric_unit)

    ppl_metric = {
        'val_ppl': None,
        'train_ppl': None,
        'gen_gap': None
    }
    
    if args.add_canary:
        logger.info("running canary eval")
        canary_loss, fitting_loss = get_fit_canary_loss(model,fitting_canaries_ids,canary_ids)        
        exposure = get_exposure(fitting_loss,canary_loss)
        logger.info(exposure)

    # eval on eval
    perplexity = eval_ppl(model, args, eval_dataloader)
    ppl_metric['val_ppl'] = perplexity
    logger.info(f'EPOCH {epoch}, val_ppl = {perplexity}')

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # loop on non
    with torch.no_grad():
        for step, batch in enumerate(non_dataloader):
            labels = batch["labels"][..., 1:].contiguous()

            # target model
            # for label = torch.Size([4, 1024])
            # for logits = torch.Size([4, 1024, 50275])
            logits = model(**batch).logits[..., :-1, :].contiguous()
            sample_loss = loss_fn(logits.transpose(1, 2), labels).mean(1).tolist()
            mia_metric['non_mem']['losses'].extend(sample_loss)
            
            # ref model
            if args.do_ref_model:
                logits = model_ref(**batch).logits[..., :-1, :].contiguous()
                sample_loss_ref = loss_fn(logits.transpose(1, 2), labels).mean(1).tolist()
                mia_metric['non_mem']['diff_losses_ref'].extend((np.array(sample_loss) - np.array(sample_loss_ref)).tolist())

    accelerator.wait_for_everyone()
    assert len(mia_metric['non_mem']['losses']) == len(non_dataloader.dataset) 

    # choose the threshold on non-mem
    sorted_loss = sorted(mia_metric['non_mem']['losses'])
    if args.do_ref_model:
        sorted_ratio = sorted(mia_metric['non_mem']['diff_losses_ref'])

    len_list = len(mia_metric['non_mem']['losses'])
    for fpr in fpr_list:
        mia_metric['loss_mia'][f'fpr{fpr}']['threshold'] = sorted_loss[int(fpr*len_list)]
        if args.do_ref_model:
            mia_metric['ref_model_mia'][f'fpr{fpr}']['threshold'] = sorted_ratio[int(fpr*len_list)]
        
    ################################################    
    #run threshold on training samples
    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            labels = batch["labels"][..., 1:].contiguous()

            # target model
            outputs = model(**batch)
            logits = outputs.logits[..., :-1, :].contiguous()
            sample_loss = loss_fn(logits.transpose(1, 2), labels).mean(1).tolist()
            mia_metric['mem']['losses'].extend(sample_loss)

            # ref model
            if args.do_ref_model:
                logits = model_ref(**batch).logits[..., :-1, :].contiguous()
                sample_loss_ref = loss_fn(logits.transpose(1, 2), labels).mean(1).tolist()
                mia_metric['mem']['diff_losses_ref'].extend((np.array(sample_loss) - np.array(sample_loss_ref)).tolist())

    accelerator.wait_for_everyone()

    # train ppl
    perplexity_train = loss_to_ppl(mia_metric['mem']['losses'])
    gen_gap = perplexity - perplexity_train
    ppl_metric['train_ppl'] = perplexity_train
    ppl_metric['gen_gap'] = gen_gap
    logger.info(f'EPOCH {epoch}, train_ppl = {perplexity_train}')
    logger.info(f'EPOCH {epoch}, gen_gap = {gen_gap}')

    # attack predict
    len_list_train = len(mia_metric['mem']['losses'])
    len_list_non = len(mia_metric['non_mem']['losses'])

    def fpr_get_tpr(att_name, score_name, fpr):
        ''' inner func '''
        guess_cor = sum([1 for sample in mia_metric['mem'][score_name] if sample < mia_metric[att_name][f'fpr{fpr}']['threshold']])
        mia_metric[att_name][f'fpr{fpr}']['recall'] = guess_cor / len_list_train

        logger.info(f"{att_name} recall@FPR{fpr} = {mia_metric[att_name][f'fpr{fpr}']['recall']} ({guess_cor}/{len_list_train})")

    # if accelerator.is_local_main_process: # TODO: fix when use multiple GPUs
    for fpr in fpr_list:
        fpr_get_tpr('loss_mia', 'losses', fpr)
        if args.do_ref_model:
            fpr_get_tpr('ref_model_mia', 'diff_losses_ref', fpr)


    ## AUC
    def log_auc(y_true, y_scores, att_name):
        # roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        mia_metric[att_name]['roc_curve']['fpr'] = fpr.tolist()
        mia_metric[att_name]['roc_curve']['tpr'] = tpr.tolist()
        mia_metric[att_name]['roc_curve']['roc_auc'] = roc_auc

        logger.info(f'{att_name}, roc_auc = {roc_auc:.4f}')

    y_true = [1] * len_list_train + [0] * len_list_non
    y_scores = list(map(lambda x: -x, mia_metric['mem']['losses'] + mia_metric['non_mem']['losses']))
    log_auc(y_true, y_scores, 'loss_mia')

    if args.do_ref_model:
        y_scores_ref_model = list(map(lambda x: -x, mia_metric['mem']['diff_losses_ref'] + mia_metric['non_mem']['diff_losses_ref']))
        log_auc(y_true, y_scores_ref_model, 'ref_model_mia')

    return mia_metric, ppl_metric


def train_ft(
        model,
        model_ref,
        args,
        accelerator,
        train_dataloader,
        eval_dataloader,
        non_dataloader,
        fitting_canaries_ids,
        canary_ids,
        ft_metric,
        aux_dataset,
        train_dataset
    ):
        

    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay_def,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, model_ref, optimizer, train_dataloader, eval_dataloader, non_dataloader = accelerator.prepare(
        model, model_ref, optimizer, train_dataloader, eval_dataloader, non_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_steps_per_log = math.ceil(num_update_steps_per_epoch/5)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
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

    mia_m, ppl_m = priv_measure(
            model=model,
            accelerator=accelerator,
            args=args,
            eval_dataloader=eval_dataloader,
            train_dataloader=train_dataloader,
            non_dataloader=non_dataloader,
            epoch=-1,
            fitting_canaries_ids=fitting_canaries_ids,
            canary_ids=canary_ids,
            model_ref=model_ref,
        )
    ft_metric[args.num_train_epochs] = {
            'mia_metric': mia_m,
            'ppl_metric': ppl_m
        }

    for epoch in range(args.num_train_epochs):
        model.train()
        running_loss = deque()
        logger.info(f"training epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            running_loss.append(loss.item())
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.clip_def:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_def)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1
            
            if completed_steps % num_steps_per_log == 0:
                ft_loss = np.mean(running_loss)
                logger.info(f"ft_loss: {ft_loss:.4f}")
            
            if completed_steps >= args.max_train_steps:
                break   

        if ((epoch + 1) % args.log_per_step == 0 and not args.only_eval_last) or epoch == args.num_train_epochs - 1:
            mia_m, ppl_m = priv_measure(
                model=model,
                accelerator=accelerator,
                args=args,
                eval_dataloader=eval_dataloader,
                train_dataloader=train_dataloader,
                non_dataloader=non_dataloader,
                epoch=epoch,
                fitting_canaries_ids=fitting_canaries_ids,
                canary_ids=canary_ids,
                model_ref=model_ref,
            )
            ft_metric[epoch] = {
                'mia_metric': mia_m,
                'ppl_metric': ppl_m
            }
            if epoch == args.num_train_epochs - 1 and args.save_ft:
                if args.peft in ['adapter', 'lora', 'ia3', 'prefix', 'compacter']:
                    model.save_adapter(args.model_folder+f'/adapter_{epoch+1}', args.peft)
                    ckpt = {
                        'train_ppl': ft_metric[epoch]['ppl_metric']['train_ppl'],
                        'val_ppl': ft_metric[epoch]['ppl_metric']['val_ppl'],
                        'gen_gap': ft_metric[epoch]['ppl_metric']['gen_gap'],
                        }
                elif args.peft in ['head', 'bitfit', 'full']:
                    trainable_keys = [name for name, param in model.named_parameters() if param.requires_grad]
                    trainable_params = {name: param for name, param in model.state_dict().items() if name in trainable_keys}
                    ckpt = {
                        'model_state_dict': trainable_params,
                        'train_ppl': ft_metric[epoch]['ppl_metric']['train_ppl'],
                        'val_ppl': ft_metric[epoch]['ppl_metric']['val_ppl'],
                        'gen_gap': ft_metric[epoch]['ppl_metric']['gen_gap'],
                        }
                    model_path = args.model_folder + f'/ckpt_{epoch+1}.pth'
                    torch.save(ckpt, model_path)
                    logger.info(f'Saving ft model to {model_path}')
                else:
                    raise ValueError
            
def main():

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    
    ################## logging ##################
    args = parse_args()

    if args.load_ref_file is None:
        ref_model_name = 'benign'
    else:
        ref_model_name = args.load_ref_name

    if args.load_init_file is None:
        init_model_name = 'benign'
    else:
        init_model_name = args.load_init_name
    

    folder_name = f"un_ft_{args.peft}_{args.model_name_or_path}_{args.dataset_name}_aux{args.pre_aux_ratio}_lr{args.learning_rate}_epoch_{args.num_train_epochs}_def{args.dropout_def}{args.weight_decay_def}{args.clip_def}_head{args.drop_head}{args.untied_head}_init{init_model_name}"

    if args.add_canary:
        folder_name = f"canary_{str(args.canary_rep)}_{str(args.canary_len)}" + folder_name

    if args.save_ft:
        folder_name = 'SAVE' + folder_name

    if args.debug:
        folder_name = 'DEBUG' + folder_name
    
    directory = "{}{}".format(args.output_dir,folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    log_file = os.path.join(directory, "stdout")
    metric_file = os.path.join(directory, "metric.json")
    model_folder = os.path.join(directory, "model")
    if os.path.exists(log_file):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = f"{log_file}_{timestamp}.log"
        metric_file = f"{metric_file}_{timestamp}.json"
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

    #### add canary to train set if need ####
    if args.add_canary:
        raw_datasets, fitting_canaries_ids, canary_ids = add_canary(args, tokenizer, raw_datasets, directory, logger)
    else:
        fitting_canaries_ids = None
        canary_ids = None

    #### Preprocessing the datasets ####
    train_dataset, eval_dataset = preprocess_data(args, raw_datasets, tokenizer, accelerator, logger)
    if args.debug:
        args.num_train_epochs = 1
        subset_ids = list(range(20))
        train_dataset = Subset(train_dataset, subset_ids)
        aux_dataset = Subset(train_dataset, subset_ids)
        non_dataset = Subset(train_dataset, subset_ids)
        subset_ids = list(range(10))
        eval_dataset = Subset(eval_dataset, subset_ids)
    else:
        ###### split for aux dataset ####
        aux_dataset, train_dataset, non_dataset = load_aux_data(args, train_dataset, logger, must_exist=True)

    # dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    non_dataloader = DataLoader(
        non_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    aux_dataloader = DataLoader(
        aux_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )


    all_metric = {'pre': args.load_ref_file,
                'init': args.load_init_file,
                'ft': {}}

    ############### load model and model_ref ##############
    model = load_model_only(args, tokenizer, logger)
    model_ref = copy.deepcopy(model)

    if args.drop_head:
        lm_head = copy.deepcopy(model.lm_head)
    if args.load_ref_file:
        ckpt = torch.load(args.load_ref_file)
        model_ref.load_state_dict(ckpt['model_state_dict'])
        logger.info("Advanced model_ref loaded")
    else:
        logger.info("Basic pre-trained model_ref loaded")

    if args.load_init_file:
        ckpt = torch.load(args.load_init_file)
        if args.bad:
            BAD_PARAM = args.bad_key
            for name in ckpt['model_state_dict'].keys():
                if BAD_PARAM in name:
                    ckpt['model_state_dict'][name] = args.bad_scale * ckpt['model_state_dict'][name]
                    logger.info(f'Popping {name} from PreCurious model initialization...')
        if args.drop_head:
            # Remove 'transformer.wte.weight' and 'lm_head.weight' from the state dict
            ckpt['model_state_dict'].pop('transformer.wte.weight', None)
            ckpt['model_state_dict'].pop('lm_head.weight', None)
            
            # Load the state dict with missing keys (strict=False to ignore missing keys)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            # load all state_dict in ckpt
            try:
                model.load_state_dict(ckpt['model_state_dict'])
            except:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info("PreCuriopus model intialization loaded")
    else:
        if args.bad:
            ckpt = {"model_state_dict": model.state_dict()}
            BAD_PARAM = args.bad_key
            for name in ckpt['model_state_dict'].keys():
                if BAD_PARAM in name:
                    ckpt['model_state_dict'][name] = args.bad_scale * ckpt['model_state_dict'][name]
                    logger.info(f'Popping {name} from PreCurious model initialization...')
            model.load_state_dict(ckpt['model_state_dict'])
        logger.info("Benign pre-trained model intialization loaded")

    ################## setup ft model ###############
    setup_peft(model, args.peft, args)
    count_parameters(args, logger, model, prefix='ft')
    print([name for name, p in model.named_parameters() if p.requires_grad])
    if args.untied_head:
        model.lm_head = lm_head
    if args.fresh_head:
        model.lm_head = torch.nn.Linear(model.config.n_embd, model.config.vocab_size, bias=False)
    if args.dropout_def:
        set_dropout(model, args.dropout_def)
    count_parameters(args, logger, model, prefix='ft')

    ################## fine-tuning ##################
    train_ft(
        model,
        model_ref,
        args,
        accelerator,
        train_dataloader,
        eval_dataloader,
        non_dataloader,
        fitting_canaries_ids,
        canary_ids,
        all_metric['ft'],
        aux_dataset,
        train_dataset
    )
    with open(metric_file, 'w') as f:
        json.dump(all_metric, f)
        logger.info(f"Saving metrics to {metric_file}")
    logger.info(directory)


if __name__ == "__main__":
    main()
