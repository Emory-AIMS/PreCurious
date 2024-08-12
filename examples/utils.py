import os
import sys
import time
import gc
import random

from datetime import datetime

import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Subset, random_split

from itertools import chain
from tqdm.auto import tqdm
import lora_utils

from datasets import load_dataset, load_from_disk
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
# import adapters
from transformers.adapters import (
    PrefixTuningConfig,
    CompacterConfig,
    LoRAConfig,
    IA3Config,
    AdapterConfig,
)


def memReport():
  for obj in gc.get_objects():
    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
      print(type(obj), obj.size())

def reorder(x, index):
    """original x is reordered in terms of index to get x,
    this function is to recover original index
    Args:
    x: list
    index: numpy array, index[i] == j means the ith element
        in x was located at j originally
    """

    assert(len(x) == len(index))
    new_x = [0 for _ in range(len(x))]

    for i, j in enumerate(index):
        new_x[j] = x[i]

    return new_x

def get_criterion(hparams):
  loss_reduce = False
  crit = nn.CrossEntropyLoss(ignore_index=hparams.pad_id, size_average=False, reduce=loss_reduce)
  if hparams.cuda:
    crit = crit.cuda()
  return crit

def get_performance(crit, trans_logits, noise_logits, labels, hparams, x_len, sum_loss=True):
  # average over length
  x_len_t = torch.tensor(x_len, dtype=torch.float, requires_grad=False, device=hparams.device)
  x_len_t = x_len_t - 1
  batch_size = len(x_len)
  mask = (labels == hparams.pad_id)
  if hparams.bt:
    trans_logits = trans_logits.view(-1, hparams.src_vocab_size)
    trans_loss = crit(trans_logits, labels)
    trans_loss = trans_loss.view(batch_size, -1).sum(-1)
    _, trans_preds = torch.max(trans_logits, dim=1)
    trans_acc = torch.eq(trans_preds, labels).int().masked_fill_(mask, 0).sum().item()
  else:
    trans_loss = torch.zeros((batch_size), requires_grad=False, device=hparams.device)
    trans_acc = 0

  if hparams.noise_flag:
    noise_logits = noise_logits.view(-1, hparams.src_vocab_size)
    noise_loss = crit(noise_logits, labels)
    noise_loss = noise_loss.view(batch_size, -1).sum(-1)
    _, preds = torch.max(noise_logits, dim=1)
    acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum().item()
  else:
    noise_loss = torch.zeros((batch_size), requires_grad=False, device=hparams.device)
    acc = 0

  if hparams.avg_len:
    noise_loss = noise_loss / x_len_t
    trans_loss = trans_loss / x_len_t

  trans_loss = trans_loss.sum()
  noise_loss = noise_loss.sum()
  loss = trans_loss + hparams.noise_weight * noise_loss
  #loss = noise_loss.sum()
  return loss, trans_loss, noise_loss, acc, trans_acc


def save_checkpoint(extra, model, optimizer, hparams, path):
  print("Saving model to '{0}'".format(path))
  torch.save(extra, os.path.join(path, "extra.pt"))
  torch.save(model, os.path.join(path, "model.pt"))
  torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
  torch.save(hparams, os.path.join(path, "hparams.pt"))
  torch.save(model.state_dict(), os.path.join(path, "model.dict"))

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "w")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

def set_lr(optim, lr):
  for param_group in optim.param_groups:
    param_group["lr"] = lr

def init_param(p, init_type="uniform", init_range=None):
  if init_type == "xavier_normal":
    init.xavier_normal(p)
  elif init_type == "xavier_uniform":
    init.xavier_uniform(p)
  elif init_type == "kaiming_normal":
    init.kaiming_normal(p)
  elif init_type == "kaiming_uniform":
    init.kaiming_uniform(p)
  elif init_type == "uniform":
    #assert init_range is not None and init_range > 0
    init.uniform_(p, -init_range, init_range)
  else:
    raise ValueError("Unknown init_type '{0}'".format(init_type))


def get_attn_subsequent_mask(seq, pad_id=0):
  """ Get an attention mask to avoid using the subsequent info."""

  assert seq.dim() == 2
  batch_size, max_len = seq.size()
  sub_mask = torch.triu(
    torch.ones(max_len, max_len), diagonal=1).unsqueeze(0).repeat(
      batch_size, 1, 1).type(torch.ByteTensor)
  if seq.is_cuda:
    sub_mask = sub_mask.cuda()
  return sub_mask

def grad_clip(params, grad_bound=None):
  """Clipping gradients at L-2 norm grad_bound. Returns the L-2 norm."""

  params = list(filter(lambda p: p.grad is not None, params))
  total_norm = 0
  for p in params:
    if p.grad is None:
      continue
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm ** 2
  total_norm = total_norm ** 0.5

  if grad_bound is not None:
    clip_coef = grad_bound / (total_norm + 1e-6)
    if clip_coef < 1:
      for p in params:
        p.grad.data.mul_(clip_coef)
  return total_norm

def count_parameters(args, logger, model, prefix=''):
    num_total_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if num_total_params == 0:
        percentage_trainable = 0.0
    else:
        percentage_trainable = (num_trainable_params / num_total_params) * 100
    logger.info(f"{percentage_trainable:.4f} % trainable params ({num_trainable_params/1e6} / {num_total_params/1e6})")

# for attack
def flip_random(args, aux_dataset):
    for i in range(len(aux_dataset)):
        seq_len = len(aux_dataset[i]['input_ids'])
        rand_ids = random.sample(range(seq_len), k=int(args.flip_p * seq_len))
        assert len(rand_ids) > 0
        for ids in rand_ids:
            # aux_dataset[i]['input_ids'][ids] = random.randint(0, args.vocab_size)
            print(aux_dataset[i]["labels"][ids])
            aux_dataset[i]['labels'][ids] = random.randint(0, args.vocab_size)
            print(aux_dataset[i]["labels"][ids])
    return aux_dataset

def load_tokenizer(args):
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    args.vocab_size = tokenizer.vocab_size
    return tokenizer

def setup_peft(model, peft, args):
    if peft == 'full':
        for params in model.parameters():
            params.requires_grad = True
    elif peft == 'head':
        for params in model.parameters():
            params.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = True
    elif peft == 'layers':
        for params in model.parameters():
            params.requires_grad = False
        for n in args.train_layer_n_only:
            for params in model.transformer.h[n].parameters():
                params.requires_grad = True
    elif peft == 'bitfit':
        for name, params in model.named_parameters():
            if '.bias' in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
    elif peft == 'adapter':
        # add new adapter
        # adapters.init(model)
        config = AdapterConfig(
                        ln_after= False,
                        ln_before= False,
                        mh_adapter= False,
                        output_adapter= True,
                        adapter_residual_before_ln= False,
                        non_linearity= "relu",
                        original_ln_after= True,
                        original_ln_before= True,
                        reduction_factor= args.adapter_reduction,
                        residual_before_ln= True
        )
        model.add_adapter(peft, config)
        # activate adapter for training
        model.train_adapter("adapter")
    elif peft == 'prefix':
        config = PrefixTuningConfig(flat=True, prefix_length=30)
        model.add_adapter(peft, config=config)
        model.train_adapter(peft)
    elif peft == 'compacter':
        config = CompacterConfig()
        model.add_adapter(peft, config=config)
        model.train_adapter(peft)
    elif peft == 'lora':
        config = LoRAConfig(r=args.lora_r, alpha=16)
        model.add_adapter(peft, config=config)
        model.train_adapter(peft)
        # model.merge_adapter("lora")
        # model.reset_adapter()
    elif peft == 'loradp':
        lora_utils.make_lora_gpt2(model, args.lora_r)
    elif peft == 'ia3':
        config = IA3Config()
        model.add_adapter("ia3", config=config)
        model.train_adapter("ia3")
    else:
        raise ValueError

# reset the dropout rate
def set_dropout(model, dropout_rate):
    """
    Recursively traverse all submodules of `model` and set the dropout rate.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
    
# model_utils
def load_model(args, tokenizer, logger, peft):
    '''
    Load pretrained model and tokenizer
    In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    '''
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError('config_name or model_name_or_path should be set')
    
    # if dropout:
    #     for k, v in vars(config).items():
    #         if 'pdrop' in k:
    #             setattr(config, k, dropout)
    #             logger.info(f"Modifying {k} from {v} to {getattr(config, k)}")

    # args.vocab_size = tokenizer.vocab_size

    if args.model_name_or_path:
        model = GPT2LMHeadModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    #### PEFT ####
    setup_peft(model, peft, args)

    model_ref = copy.deepcopy(model) # use the backbone model as ref model if no pre-curious

    return model, model_ref

def load_model_only(args, tokenizer, logger):
    '''
    Load pretrained model and tokenizer
    In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    '''
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError('config_name or model_name_or_path should be set')
    
    # if dropout:
    #     for k, v in vars(config).items():
    #         if 'pdrop' in k:
    #             setattr(config, k, dropout)
    #             logger.info(f"Modifying {k} from {v} to {getattr(config, k)}")

    # args.vocab_size = tokenizer.vocab_size

    if args.model_name_or_path:
        model = GPT2LMHeadModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    return model

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

def add_canary(args, tokenizer, raw_datasets, directory, logger):
    if 'ptb' in args.dataset_name:
        dict_key = 'sentence'
    elif 'pubmed' in args.dataset_name:
        dict_key = 'abstract'
    else:
        dict_key='text'
    logger.info("before canary len ", len(raw_datasets['train'][dict_key]))
    canary, canary_ids = gen_canary(args.canary_len, tokenizer)
    for j in range(args.canary_rep):
        raw_datasets['train']=raw_datasets['train'].add_item({dict_key:canary})

    raw_datasets['train'] = raw_datasets['train'].shuffle(seed=args.seed)
    logger.info("after canary len ", len(raw_datasets['train'][dict_key]))
    # save the canaries in csv

    file = open(f'./{directory}/canaries.txt', 'w+')
    file.write(canary)
    file.write('\n')
    file.close()

    file = open(f'./{directory}/fitting_canaries.txt', 'w+')

    fitting_canaries_ids = []
    for i in range(5000):
        fit , fit_ids = gen_canary(args.canary_len,tokenizer)
        if fit != canary:
            fitting_canaries_ids.append(fit_ids)
            file.write(fit)
            file.write('\n')
    logger.info(len(fitting_canaries_ids))
    return raw_datasets, fitting_canaries_ids, canary_ids

def load_raw_data(args, logger):
    if 'enron' in args.dataset_name:
        raw_datasets = load_dataset('csv',
                                data_files={
                                    'train': '../cache/datasets/enron/cleaned_short_train_scrubbed.csv' ,
                                    'validation': '../cache/datasets/enron/cleaned_short_test_scrubbed.csv'
                                    }
                                )
    elif 'pubmed' in args.dataset_name:
        raw_datasets = load_from_disk('../cache/datasets/pubmed/pubmed_tokenized_sub1234')
    elif 'ag_news' in args.dataset_name:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        logger.info('Use original test dataset as new validation dataset')
        raw_datasets['validation'] = raw_datasets['test']
    else:
        # wikitext, ptb_text_only
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            logger.info('Spliting original trainset into train & validation')
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    return raw_datasets

def preprocess_data(args, raw_datasets, tokenizer, accelerator, logger):
    # First we tokenize all the texts.
    if 'pubmed' in args.dataset_name:
        return raw_datasets['train'], raw_datasets['validation']

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.

        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    logger.info(f"Original train size = {len(train_dataset)}")
    logger.info(f"eval size = {len(eval_dataset)}")

    return train_dataset, eval_dataset

class auxDataset(torch.utils.data.Dataset):
    def __init__(self, aux_ds):
       self.aux_ds = aux_ds
    
    def __len__(self):
       return len(self.aux_ds)
    
    def __getitem__(self, index):
       return self.aux_ds[index]
    
class auxDataset_random_label(auxDataset):
    def __init__(self, aux_ds, flip_p=None, vocab_size=None):
        super().__init__(aux_ds)
        self.flip_p = flip_p
        self.vocab_size = vocab_size
           
    def __getitem__(self, index):
        seq_len = len(self.aux_ds[index]["input_ids"])
        rand_ids = random.sample(range(seq_len), k=int(self.flip_p * seq_len))
        labels = self.aux_ds[index]["labels"]
        for ids in rand_ids:
            labels[ids] = random.randint(0, self.vocab_size-1)
           
        return {"input_ids": self.aux_ds[index]["input_ids"],
              "attention_mask": self.aux_ds[index]["attention_mask"],
              "labels": labels}

class auxDataset_random_xy(auxDataset):
    def __init__(self, aux_ds, flip_p=None, vocab_size=None):
        super().__init__(aux_ds)
        self.flip_p = flip_p
        self.vocab_size = vocab_size
           
    def __getitem__(self, index):
        seq_len = len(self.aux_ds[index]["input_ids"])
        rand_ids = random.sample(range(seq_len), k=int(self.flip_p * seq_len))
        labels = self.aux_ds[index]["labels"]
        input_ids = self.aux_ds[index]["input_ids"]
        for ids in rand_ids:
            input_ids[ids] = random.randint(0, self.vocab_size-1)
            labels[ids] = random.randint(0, self.vocab_size-1)
           
        return {"input_ids": self.aux_ds[index]["input_ids"],
              "attention_mask": self.aux_ds[index]["attention_mask"],
              "labels": labels}

class auxDataset_random_sim(auxDataset):
    def __init__(self, aux_ds, flip_p=None, vocab_size=None):
        super().__init__(aux_ds)
        self.flip_p = flip_p
        self.vocab_size = vocab_size
           
    def __getitem__(self, index):
        seq_len = len(self.aux_ds[index]["input_ids"])
        rand_ids = random.sample(range(seq_len), k=int(self.flip_p * seq_len))
        labels = self.aux_ds[index]["labels"]
        for ids in rand_ids:
            labels[ids] = random.randint(0, self.vocab_size-1)
           
        return {"input_ids": self.aux_ds[index]["input_ids"],
              "attention_mask": self.aux_ds[index]["attention_mask"],
              "labels": labels}
    
class PreDataset(torch.utils.data.Dataset):
    def __init__(self, vic_dataset, aux_dataset=None):
        self.vic_dataset = vic_dataset
        self.aux_dataset = aux_dataset
        
        # The length of the combined dataset is the sum of both individual datasets
        self.total_len = len(vic_dataset) + len(aux_dataset) if len(aux_dataset) > 0 else len(vic_dataset)
        
        # The point where the vic_dataset ends and the aux_dataset starts
        self.split_point = len(vic_dataset)
        
        # Half point of the VIC dataset
        self.vic_half = len(vic_dataset) // 2

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # If the index is within the range of the VIC dataset
        if idx < self.split_point:
            item = self.vic_dataset[idx]
            item["target"] = 1
        else: # assert self.aux_dataset is not None
            item = self.aux_dataset[idx - self.split_point]
            item["target"] = -1 # TODO:
            # item["mem"] = -1
        return item


def split_aux_data_target(args, train_dataset, logger):
    '''
    original train dataset
        - 33.3%: for train dataset
        - 33.3%: for aux dataset
        - 33.3%: for non dataset
    '''
    logger.info(f"Preparing datasets (aux size / train size  = {args.pre_aux_ratio})")
    org_train_len = len(train_dataset)
    len_non = int(org_train_len/3)
    len_train = int(org_train_len/3)
    len_aux = int(args.pre_aux_ratio * len_train)
    assert args.vic_p < 1
    len_vic_half = int(args.vic_p * len_train)

    logger.info(f"org size = {org_train_len}")
    logger.info(f"aux size = {len_aux}")
    logger.info(f"train size = {len_train}")
    logger.info(f"non size = {len_non}")
    logger.info(f"vic size (mem+nonmem) = {2 * len_vic_half}")

    idx_file_name = args.output_dir + '_'.join([args.dataset_name, str(len_train), str(len_aux), str(len_non), str(args.seed)]) + '_target.json'
    if os.path.exists(idx_file_name):
        logger.info(f"Loading existing splits from {idx_file_name} ...")
        with open(idx_file_name, 'r') as f:
           idx_dict = json.load(f)
    else:
        logger.info(f"Generating new splits {idx_file_name} ...")
        idx_shuffle = list(range(len(train_dataset)))
        random.shuffle(idx_shuffle)
        idx_dict = {'train': idx_shuffle[:len_train],
                    'aux': idx_shuffle[len_train: len_train+len_aux],
                    'non': idx_shuffle[len_train+len_aux: len_aux + len_non + len_train]}
        vic_idx_in_train = idx_dict['train'][:len_vic_half]
        vic_idx_in_non = idx_dict['non'][:len_vic_half]
        idx_dict['vic'] = vic_idx_in_train + vic_idx_in_non
        with open(idx_file_name, 'w') as f:
           json.dump(idx_dict, f)
    
    train_ds = Subset(train_dataset, idx_dict['train'])
    aux_ds = Subset(train_dataset, idx_dict['aux']) if len_aux > 1 else None
    non_ds = Subset(train_dataset, idx_dict['non'])
    vic_ds = Subset(train_dataset, idx_dict['vic'])
    mem_ds = Subset(train_dataset, idx_dict['train'][:len_vic_half])
    non_mem_ds = Subset(train_dataset, idx_dict['non'][:len_vic_half])
    unt_mem_ds = Subset(train_dataset, idx_dict['train'][len_vic_half:])
    unt_non_mem_ds = Subset(train_dataset, idx_dict['non'][len_vic_half:])
    pre_ds = PreDataset(vic_ds, aux_ds)

    return pre_ds, train_ds, non_ds, mem_ds, non_mem_ds, unt_mem_ds, unt_non_mem_ds

def load_aux_data(args, train_dataset, logger, must_exist=False):
    '''
    original train dataset
        - 33.3%: for train dataset
        - 33.3%: for aux dataset
        - 33.3%: for non dataset
    '''
    logger.info(f"Preparing datasets (aux size / train size  = {args.pre_aux_ratio})")
    org_train_len = len(train_dataset)
    len_non = int(org_train_len/3)
    len_train = int(org_train_len/3)
    len_aux = int(args.pre_aux_ratio * len_train)
    len_rest = org_train_len - (len_aux + len_non + len_train)

    logger.info(f"org size = {org_train_len}")
    logger.info(f"aux size = {len_aux}")
    logger.info(f"train size = {len_train}")
    logger.info(f"non size = {len_non}")

    idx_file_name = args.split_dir + '_'.join([args.dataset_name, str(len_train), str(len_aux), str(len_non), str(args.data_seed)]) + '.json'
    if os.path.exists(idx_file_name):
        logger.info("Loading existing splits...")
        with open(idx_file_name, 'r') as f:
            idx_dict = json.load(f)
    elif must_exist == False:
        logger.info("Generating new splits...")
        idx_shuffle = list(range(len(train_dataset)))
        random.seed(args.data_seed)
        random.shuffle(idx_shuffle)
        idx_dict = {'train': idx_shuffle[:len_train],
                    'aux': idx_shuffle[len_train: len_train+len_aux],
                    'non': idx_shuffle[len_train+len_aux: len_aux + len_non + len_train]}
        with open(idx_file_name, 'w') as f:
           json.dump(idx_dict, f)
    else:
        print(idx_file_name)
        raise ValueError
    
    train_ds = Subset(train_dataset, idx_dict['train'])
    aux_ds = Subset(train_dataset, idx_dict['aux'])
    non_ds = Subset(train_dataset, idx_dict['non'])

    return aux_ds, train_ds, non_ds
    
def split_aux_data(args, train_dataset, logger):
    '''
    original train dataset
        - 33.3%: for train dataset
        - 33.3%: for aux dataset
        - 33.3%: for non dataset
    '''
    logger.info(f"Preparing datasets (aux size / train size  = {args.pre_aux_ratio})")
    org_train_len = len(train_dataset)
    len_non = int(org_train_len/3)
    len_train = int(org_train_len/3)
    len_aux = int(args.pre_aux_ratio * len_train)
    len_rest = org_train_len - (len_aux + len_non + len_train)

    logger.info(f"org size = {org_train_len}")
    logger.info(f"aux size = {len_aux}")
    logger.info(f"train size = {len_train}")
    logger.info(f"non size = {len_non}")

    idx_file_name = args.output_dir + '_'.join([args.dataset_name, str(len_train), str(len_aux), str(len_non), str(args.seed)]) + '.json'
    if os.path.exists(idx_file_name):
        logger.info("Loading existing splits...")
        with open(idx_file_name, 'r') as f:
           idx_dict = json.load(f)
    else:
        logger.info("Generating new splits...")
        idx_shuffle = list(range(len(train_dataset)))
        random.shuffle(idx_shuffle)
        idx_dict = {'train': idx_shuffle[:len_train],
                    'aux': idx_shuffle[len_train: len_train+len_aux],
                    'non': idx_shuffle[len_train+len_aux: len_aux + len_non + len_train]}
        with open(idx_file_name, 'w') as f:
           json.dump(idx_dict, f)
    
    train_ds = Subset(train_dataset, idx_dict['train'])
    aux_ds = Subset(train_dataset, idx_dict['aux'])
    non_ds = Subset(train_dataset, idx_dict['non'])

    return aux_ds, train_ds, non_ds

from scipy.stats import skewnorm
from scipy.stats import kstest

def get_exposure(fitting, main):

    fitting_params = skewnorm.fit(fitting)
    ks = kstest(fitting, 'skewnorm', fitting_params)

    cdf = skewnorm.cdf(main, fitting_params[0], fitting_params[1], fitting_params[2])

    
    if cdf == 0.0:
        exposure = 0.0
    else:
        exposure = -1.0*np.log2(cdf)
    
    
    return exposure

def get_fit_canary_loss(model,fitting_id, main_id):
    loss_list = []
    for k, v in main_id.items():
        main_id[k] = torch.tensor(v).cuda()
                  
    loss_main = np.exp(model(**main_id)['loss'].item())

    for sample in fitting_id:
        for k, v in sample.items():
            sample[k] = torch.tensor(v).cuda()
        
        output = model(**sample)
        loss_list.append(np.exp(output.loss.item()))

    return loss_main,loss_list

import math

def eval_ppl(
        model,
        args,
        eval_dataloader,
        accelerator=None,
        ):
    losses = []
    model.eval()
    if accelerator is not None:
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        # losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
        losses.append(loss.repeat(args.per_device_eval_batch_size))

    losses = torch.cat(losses)
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    return perplexity


def loss_to_ppl(losses):
    try:
        return math.exp(np.mean(losses))
    except OverflowError:
        return float("inf")

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__