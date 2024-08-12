import os
import numpy as np
import struct
import multiprocessing as mp
from datasets import Dataset

class Arguments:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

import re
import json

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

args = Arguments(
    dataset_name='enron', #'pubmed', #'enron', #'ptb_text_only',
    preprocessing_num_workers=None,
    overwrite_cache=False,
    block_size=1024,
    dataset_config_name=None,
    validation_split_percentage=5,
    tokenizer_name=None,
    use_slow_tokenizer=False,
    model_name_or_path='gpt2',
    pre_aux_ratio=1.0,
    split_dir='../cache/datasets/',
    data_seed=1234,
    tokenize=True,
    save_dir='../cache/datasets',
    dedup_length=40,
    save=True,
)

from utils import load_raw_data, preprocess_data, load_tokenizer, load_aux_data
from accelerate import Accelerator

accelerator = Accelerator()

raw_datasets = load_raw_data(args, logger)
tokenizer = load_tokenizer(args)

train_dataset, eval_dataset = preprocess_data(args, raw_datasets, tokenizer, accelerator, logger)

aux_dataset, train_dataset, non_dataset = load_aux_data(args, train_dataset, logger, must_exist=True)

def get_mask(ds, name, save_ds):
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    phone_number_pattern = re.compile(r'(\d{3})-(\d{3})-(\d{4})')

    email_phone_file = open(f'{args.save_dir}/{args.dataset_name}/{name}_phone_email', "w")

    dedup_ds = []

    for sublist in ds:
        dedup_seq = {'input_ids': [], 'attention_mask': [], 'labels': []}
        text = tokenizer.decode(sublist['input_ids'])

        dummy_token = tokenizer.encode(" ")

        # match
        email_matches = list(email_pattern.finditer(text))
        phone_matches = list(phone_number_pattern.finditer(text))

        # check
        if not email_matches and not phone_matches:
            dedup_seq["input_ids"] = sublist["input_ids"]
            dedup_seq["attention_mask"] = sublist["attention_mask"]
            dedup_seq["labels"] = sublist["labels"]
            dedup_ds.append(dedup_seq)
            continue

        # calculate location
        all_matches = sorted(email_matches + phone_matches, key=lambda match: match.start())

        # remove subseq
        # print("== text before")
        # print(text)
        # print("== detected secretes")
        for match in reversed(all_matches):
            start, end = match.span()
            mask_text = text[start: end]
            email_phone_file.write(mask_text + "\n")
            print(mask_text)
            dummy_mask = " "
            text = text[:start] + dummy_mask + text[end:]
        # print("== text after")
        # print(text)

        text_ids = tokenizer.encode(text)
        if len(text_ids) < args.block_size:
            text_ids.extend(dummy_token * (args.block_size - len(text_ids)))
        elif len(text_ids) > args.block_size:
            text_ids = text_ids[:args.block_size]
        assert len(text_ids) == args.block_size
        dedup_seq["input_ids"] = text_ids
        dedup_seq["attention_mask"] = sublist["attention_mask"]
        dedup_seq["labels"] = dedup_seq["input_ids"]
        dedup_ds.append(dedup_seq)

    dedup_ds = {key: [dic[key] for dic in dedup_ds] for key in dedup_ds[0]}
    if save_ds:
        dedup_ds = Dataset.from_dict(dedup_ds)
        dedup_ds.save_to_disk(f'{args.save_dir}/{args.dataset_name}_dedup/{name}')


def get_prefix2(ds, name, L_max=512):
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    phone_number_pattern = re.compile(r'(\d{3})-(\d{3})-(\d{4})')

    file_name=f'{args.save_dir}/{args.dataset_name}_dedup/{name}_phone_email2'
    prefix_file = open(file_name, "w")

    prefix_dict = {
        'prefix_ids': [],
        'secret_ids': []
    }

    for sublist in ds:
        text = tokenizer.decode(sublist['input_ids'])

        # match
        email_matches = list(email_pattern.finditer(text))
        phone_matches = list(phone_number_pattern.finditer(text))

        # check
        if not email_matches and not phone_matches:
            continue

        # calculate location
        all_matches = sorted(email_matches + phone_matches, key=lambda match: match.start())

        # remove subseq
        aux_text = text
        for match in reversed(all_matches):
            start, end = match.span()
            dummy_mask = " " * len(text[start:end])
            aux_text = aux_text[:start] + dummy_mask + aux_text[end:]

        for match in reversed(all_matches):
            start, end = match.span()
            prefix_text = aux_text[:start][-L_max:]
            prefix_file.write(prefix_text + "\n")
            prefix_dict['prefix_ids'].append(tokenizer.encode(prefix_text))
            prefix_dict['secret_ids'].append(tokenizer.encode(text[start:end]))

    with open(file_name + '.json', 'w') as f:
        json.dump(prefix_dict, f)

# save masked train
get_mask(train_dataset, name='mask_train', save_ds=True)
# save prefix for masked train
# L_max=100
# get_prefix2(train_dataset, name=f'mask_train_prefix_{L_max}_aux', L_max=L_max)