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
import string

from scipy.stats import skewnorm
from scipy.stats import kstest

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
        # required=True,
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
        "--tune_ref", action="store_true",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
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

    # canary exposure
    parser.add_argument("--add_canary", action="store_true", help = "If true, then add canaries in the dataset.")
    parser.add_argument("--canary_rep", default=None, type = int, help = "The repetition of each canary")
    parser.add_argument("--canary_len", default = 5, type = int, help = "The len of digit of canaries")
    parser.add_argument("--num_cand", type=int, default=50000)

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
        "--peft_pre",
        type=str,
        default='none',
        choices=['full', 'layers', 'bitfit', 'lora', 'ia3', 'head']
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
    parser.add_argument("--untied_head", action="store_true")
    parser.add_argument("--fresh_head", action="store_true", help="random init dropped head, otherwise use old head before pre-curious")
    parser.add_argument("--attack", default=None, type=str)
    parser.add_argument("--aux_dir", default=None, type=str, help='aux directory, setup for dup and dedup')
    parser.add_argument("--train_dir", default=None, type=str, help='aux directory, setup for dup and dedup')
    parser.add_argument("--mode", default=None, type=str)
    parser.add_argument("--p", type=float, default=None)

    parser.add_argument("--pre_num_epochs", type=int, default=1)
    parser.add_argument("--pre_aux_ratio", type=float, default=1, help='the portion of auxiliary dataset')
    parser.add_argument("--sim_k", type=int, default=3)

    # defense # TODO: add dpsgd
    parser.add_argument("--dropout_def", default=None, type=float)
    parser.add_argument("--weight_decay_def", default=0.0, type=float)
    parser.add_argument("--clip_def", default=None, type=float)

    parser.add_argument("--only_eval_last", action="store_true")

    parser.add_argument(
        "--learning_rate_pre",
        type=float,
        default=None,
        help="learning_rate_pre / learning_rate ",
    )
    parser.add_argument("--dropout_pre", default=None, type=float, help='suggest 0.2')
    parser.add_argument("--weight_decay_pre", default=0.0, type=float, help='suggest 0.01')
    parser.add_argument("--clip_pre", default=None, type=float, help='default 1.0')

    parser.add_argument('--split_dir', type=str, default='../pre_data/', help='folder for saving data split')
    parser.add_argument('--data_seed', default=1234, type=int)

    parser.add_argument('--save_ckpt', default=[], type=int, nargs="+", help="mid epoch for saving ckpt")

    parser.add_argument("--load_ref_file", type=str, default=None, help="load a reference model, none for benign; ends with ckpt.pth")
    parser.add_argument("--load_ref_name", type=str, default=None)
    parser.add_argument("--load_ft_file", type=str, default=None)
    parser.add_argument("--load_ft_name", type=str, default=None)
    parser.add_argument("--load_init_file", type=str, default=None)
    parser.add_argument("--load_init_name", type=str, default=None)

    # 添加参数
    parser.add_argument('--prefix_len', type=int, default=100, help='Length of the prefix.')
    parser.add_argument('--fix_prefix', type=str, default=" ", help='Default prefix.')
    parser.add_argument('--prefix_mode', type=str, default='aux_dup', choices=['train_dup', 'other_mode'], help='Mode of the prefix.')
    parser.add_argument('--prefix_num', type=int, default=100, help='Number of prefixes.')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum frequency for filtering.')

    parser.add_argument('--suffix_len', type=int, default=512, help='Length of the suffix.')
    parser.add_argument('--sampling_k', type=int, default=None, help='Top-k sampling parameter. Set None for greedy. default is 50')
    parser.add_argument('--num_gen', type=int, default=1000, help='Number of generations.')
    parser.add_argument('--threshold_token', type=int, default=50, help='Token threshold for filtering.')
    parser.add_argument('--k', type=int, default=100, help='Top-k filter for final outputs.')

    args = parser.parse_args()

    # Sanity checks
    assert args.dataset_name is not None
    assert args.model_name_or_path is not None

    if args.aux_dir == 'none':
        args.aux_dir = None
    
    args.load_ref_file = None if args.load_ref_file == 'none' else args.load_ref_file
    args.load_ft_file = None if args.load_ft_file == 'none' else args.load_ft_file

    args.output_dir = os.path.join('../cache/output/logs/')
 
    return args

            
def main():

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    
    ################## logging ##################
    args = parse_args()

    att_name = args.attack

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

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


    #### Preprocessing the datasets ####
    train_dataset, eval_dataset = preprocess_data(args, raw_datasets, tokenizer, accelerator, logger)
    if args.debug:
        args.num_train_epochs = 1
        subset_ids = list(range(20))
        train_dataset = Subset(train_dataset, subset_ids)
        subset_ids = list(range(10))
        eval_dataset = Subset(eval_dataset, subset_ids)

    ###### split for aux dataset ####
    aux_dataset, train_dataset, non_dataset = load_aux_data(args, train_dataset, logger)

    if args.aux_dir:
        from datasets import load_from_disk
        aux_dataset = load_from_disk(args.aux_dir)
        aux_dataset = Subset(aux_dataset, np.arange(0, len(aux_dataset)))
        aux_dataset = aux_dataset.dataset
    if args.train_dir:
        from datasets import load_from_disk
        train_dataset = load_from_disk(args.train_dir)
        train_dataset = Subset(train_dataset, np.arange(0, len(train_dataset)))
        train_dataset = train_dataset.dataset

    # fixauc
    if len(non_dataset) != len(train_dataset):
        assert len(non_dataset) > len(train_dataset)
        idx_list = list(range(len(train_dataset)))
        non_dataset = Subset(non_dataset, idx_list[:len(train_dataset)])

    # dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    # non_dataloader = DataLoader(
    #     non_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    # )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    # aux_dataloader = DataLoader(
    #     aux_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    # )

    # ################## pre-curious ##################
    # assert args.peft_pre == 'full'
    model = load_model_only(args, tokenizer, logger)
    model_ref = copy.deepcopy(model)

    setup_peft(model, args.peft, args)
    if args.load_ft_file:
        if args.peft in ['adapter', 'lora', 'ia3', 'prefix', 'compacter']:
            if args.load_init_file is not None:
                ckpt = torch.load(args.load_init_file)
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
            try:
                model.load_adapter(args.load_ft_file)
                model.train_adapter(args.peft)    
            except:
                ckpt = torch.load(args.load_ft_file)
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            ckpt = torch.load(args.load_ft_file)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            logger.info(f"ft model loaded from {args.load_ft_file}")

    setup_peft(model_ref, args.peft_pre, args)
    if args.load_ref_file:
        if args.peft_pre in ['adapter', 'lora', 'ia3', 'prefix', 'compacter']:
            model_ref.load_adapter(args.load_ref_file)
            model_ref.train_adapter(args.peft) 
        else:
            ckpt = torch.load(args.load_ref_file)
            model_ref.load_state_dict(ckpt['model_state_dict'], strict=False)
            logger.info(f"ft model loaded from {args.load_ref_file}")
    else:
        args.load_ref_name = 'benign'

    model, model_ref, train_dataloader, eval_dataloader = accelerator.prepare(model, model_ref, train_dataloader, eval_dataloader)

    ################## data extraction ##################
    # args.prefix_len = 300
    # args.prefix_mode = 'aux_dup'
    # args.prefix_num = 100
    # args.min_freq = 2
    
    # args.suffix_len = 512
    # args.sampling_k = 50

    # args.num_gen = 1000
    # args.threshold_token = 50
    # args.k = 100

    from collections import Counter
    from torch.utils.data import Dataset
    import math
    import time
    import torch.nn as nn
    
    # priv profile with data extra
    prefix_L_max=100
    prefix_file = f'../cache/datasets/enron/mask_train_prefix_{prefix_L_max}_aux_phone_email2.json'
    with open(prefix_file, 'r') as f:
        prefix_dict = json.load(f)
    prefix_list = prefix_dict['prefix_ids']
    secret_list = [tokenizer.decode(x) for x in prefix_dict['secret_ids']]
    args.prefix_num = len(prefix_list)
    
    file_name = f'../cache/results/EnronMask_exposure_{args.dataset_name}_{args.load_ft_name}_{args.load_init_name}_{args.seed}'

    # # eval ppl 
    # perplexity = eval_ppl(model, args, eval_dataloader)
    # logger.info(f'eval_ppl={perplexity}')

    cnt_correct = 0
    cnt_tot = 0

    uni_tot = len(set(secret_list))
    uni_counter = Counter(secret_list)
    uni_dict = {}
    start = time.time()

    exp_dict = {'uni_cnt': [], 'exposure': [], 'valid_exp': [], 'exposure_ref': []}

    def get_exposure(fitting, main):
        fitting_params = skewnorm.fit(fitting)
        ks = kstest(fitting, 'skewnorm', fitting_params)
        cdf = skewnorm.cdf(main, fitting_params[0], fitting_params[1], fitting_params[2])
        if cdf == 0.0:
            exposure = 0.0
        else:
            exposure = -1.0*np.log2(cdf)
        return exposure

    def get_fit_canary_loss(model, fitting_id, main_id, model_ref=None):
        loss_list = []
        loss_ref_list = []
        loss_main_ref = None

        main_id = torch.tensor(main_id).cuda()
        loss_main = model(input_ids=main_id, labels=main_id)['loss'].item()
        if model_ref:
            loss_ref = model_ref(input_ids=main_id, labels=main_id)['loss'].item()
            loss_main_ref = np.exp(loss_main - loss_ref)

        for sample in fitting_id:
            sample = torch.tensor(sample).cuda()
            loss = model(input_ids=sample, labels=sample).loss.item()
            loss_list.append(loss)
            if model_ref:
                loss_ref = model_ref(input_ids=sample, labels=sample).loss.item()
                loss = np.exp(loss - loss_ref)
                loss_ref_list.append(loss)

        return loss_main, loss_list, loss_main_ref, loss_ref_list


    def gen_canary(canary_len,tokenizer):
        raw_sample = random.choices([str(i) for i in range(10)], k=canary_len)
        raw_sample = " ".join(raw_sample)
        
        tokenized = tokenizer.tokenize(raw_sample)
        ids = tokenizer.convert_tokens_to_ids(tokenized)
        assert len(ids) == canary_len
        
        raw_sample = "the secret number is " + raw_sample
        toked =  tokenizer(raw_sample)
        toked['labels'] = toked['input_ids'].copy()
        return raw_sample, toked

    def random_char(letter, cand_size):
        if letter.islower():
            return random.choice(string.ascii_lowercase[:cand_size])
        elif letter.isupper():
            return random.choice(string.ascii_uppercase[:cand_size])
        elif letter.isdigit():
            return random.choice(string.digits[:cand_size])
        else:
            return letter

    cand_len = 10
    cand_size = None
    f = open(file_name + '.txt', 'w')
    f.write('\t'.join(['secret', 'uni_cnt', 'valid_exp', 'exposure', 'exposure_ref']) + '\n')
    if cand_size is not None:
        print(math.log(math.pow(cand_size+1, cand_len), 2))
    PREFIX = True
    for i, main_ids in enumerate(prefix_dict['secret_ids']):
        exp_dict['uni_cnt'].append(uni_counter[secret_list[i]])
        with torch.no_grad():
            # generate candidates
            if PREFIX:
                main_prefix = prefix_dict['prefix_ids'][i][-5:]
            else:
                main_prefix = []

            main_text = tokenizer.decode(main_ids)
            if cand_size is not None:
                valid = math.log(math.pow(cand_size+1, cand_len), 2)
            elif '@' in main_text:
                valid = math.log(math.pow(26, cand_len), 2)
            else:
                valid = math.log(math.pow(10, cand_len), 2)
            exp_dict['valid_exp'].append(valid)

            fitting_ids = []
            for j in range(args.num_cand):
                fitting_j_text = ''.join([random_char(char, cand_size) for char in main_text[:cand_len]])
                if fitting_j_text != main_text:
                    fitting_ids.append(main_prefix + tokenizer.encode(fitting_j_text + main_text[cand_len:]))

            main_ids = main_prefix + main_ids

            print(fitting_j_text, main_text)

            # get losses
            main_loss, fitting_loss, main_loss_ref, fitting_loss_ref = get_fit_canary_loss(model, fitting_ids, main_ids)

            # get exposure
            exposure = get_exposure(fitting_loss, main_loss)
            print(exposure)
            exp_dict['exposure'].append(exposure)
            f.write('\t'.join([
                main_text, 
                str(exp_dict['uni_cnt'][-1]), 
                str(exp_dict['valid_exp'][-1]), 
                str(exp_dict['exposure'][-1]),
                    ]
                ) + '\n'
            )
            f.flush()

    with open(file_name + '.json', 'w') as f:
        json.dump(exp_dict, f)

if __name__ == "__main__":
    main()
