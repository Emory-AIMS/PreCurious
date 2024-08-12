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
import torch.nn.functional as F
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
    parser.add_argument("--pre_aux_ratio", type=float, default=1, help='the portion of auxiliary dataset')

    # defense # TODO: add dpsgd
    parser.add_argument("--dropout_def", default=None, type=float)
    parser.add_argument("--weight_decay_def", default=0.0, type=float)
    parser.add_argument("--clip_def", default=None, type=float)

    parser.add_argument("--load_ref_file", type=str, default=None, help="load a reference model, none for benign; ends with ckpt.pth")
    parser.add_argument("--load_ref_name", type=str, default=None)
    parser.add_argument("--load_ft_file", type=str, default=None)
    parser.add_argument("--load_ft_name", type=str, default=None)
    parser.add_argument("--load_init_file", type=str, default=None)
    parser.add_argument("--load_init_name", type=str, default=None)
    parser.add_argument("--ds_epoch", type=int, default=None)
    parser.add_argument("--ds_temp", type=float, default=1)

    parser.add_argument('--split_dir', type=str, default='../pre_data/', help='folder for saving data split')
    parser.add_argument('--data_seed', default=1234, type=int)
 
    args = parser.parse_args()

    # Sanity checks
    assert args.dataset_name is not None
    assert args.model_name_or_path is not None

    if args.load_ref_file == 'none' or args.load_ref_name == 'none':
        args.load_ref_file = None
        args.load_ref_name = 'benign'
    if args.load_init_file == 'none':
        args.load_init_file = None
    if args.load_ft_file == 'none' or args.load_ft_name == 'none':
        args.load_ft_file = None
        args.load_ft_name = 'benign'

    # 使用正则表达式来匹配 "aux" 后面的数字
    if args.load_ref_file is not None:
        match = re.search(r'aux(\d+\.\d+)', args.load_ref_file)
        assert match
        args.pre_aux_ratio = float(match.group(1))

    assert args.do_ref_model

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
            
def main():

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    
    ################## logging ##################
    args = parse_args()

    if args.load_ref_file is None:
        ref_model_name = 'benign'
    else:
        ref_model_name = args.load_ref_name

    if args.load_ft_file is None:
        ft_model_name = 'none'
    else:
        ft_model_name = args.load_ft_name

    folder_name = f"mia_{ft_model_name}_ref{ref_model_name}_{args.dataset_name}_{args.peft}_{args.ds_epoch}"
    
    directory = "{}{}".format(args.output_dir,folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    metric_file = os.path.join(directory, "metric.json")
    if os.path.exists(metric_file):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        metric_file = f"{metric_file}_{timestamp}.json"

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process:
       logger.info(str(args))


    ############### set up the tokenizer ###############
    tokenizer = load_tokenizer(args)

    ################## dataset ##################
    raw_datasets = load_raw_data(args, logger)

    #### Preprocessing the datasets ####
    train_dataset, eval_dataset = preprocess_data(args, raw_datasets, tokenizer, accelerator, logger)
    if args.debug:
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

    ############### load model and model_ref ##############
    model = load_model_only(args, tokenizer, logger)
    model_ref = copy.deepcopy(model)

    if args.load_ref_file:
        ckpt = torch.load(args.load_ref_file)
        try:
            model_ref.load_state_dict(ckpt['model_state_dict'])
        except:
            assert 'full' not in args.load_ref_file
            setup_peft(model_ref, args.peft, args)   # TODO: assert same init for adapter
            model_ref.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info(f"Advanced model_ref loaded from {args.load_ref_file}")
    else:
        logger.info(f"Basic pre-trained model_ref loaded from {args.load_ref_file}")

    ################## setup ft model ###############
    setup_peft(model, args.peft, args)
    if args.load_init_file is not None:
        ckpt = torch.load(args.load_init_file)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info(f'Init model loaded from {args.load_init_file}')
    if args.load_ft_file:
        if args.peft in ['adapter', 'lora', 'ia3', 'prefix', 'compacter']:
            model.load_adapter(args.load_ft_file)
            model.train_adapter(args.peft)    
        else:
            ckpt = torch.load(args.load_ft_file)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            logger.info(f"ft model loaded from {args.load_ft_file}")
    else:
        logger.info(f"target ft model loaded from {args.load_ft_file}")

    # Prepare everything with our `accelerator`.
    model, model_ref, train_dataloader, non_dataloader, eval_dataloader, aux_dataloader = accelerator.prepare(
        model, model_ref, train_dataloader, non_dataloader, eval_dataloader, aux_dataloader
    )

    ################## priv measure ##################
    priv_metric = {
        'basic': {
            'mia': {},
            'ppl': {}
        },
        'ds': {
            'mia': {},
            'ppl': {}
        }
    }
    mia, ppl = priv_measure(
        model,
        accelerator,
        args,
        eval_dataloader,
        train_dataloader,
        non_dataloader,
        epoch=999,
        model_ref=model_ref
    )
    priv_metric['basic']['mia'] = mia
    priv_metric['basic']['ppl'] = ppl

    with open(metric_file, 'w') as f:
        json.dump(priv_metric, f)
        logger.info(f"Saving ds metrics to {metric_file}")

if __name__ == "__main__":
    main()