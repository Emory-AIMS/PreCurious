import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

loss_fn = nn.CrossEntropyLoss(reduction="none")
loss_fn_mean = nn.CrossEntropyLoss()
loss_fn_kl = nn.KLDivLoss()

def loss_warm(args, model, batch, running_loss):
    outputs = model(**batch)
    loss = outputs.loss
    running_loss.append(loss.item())
    return loss

def loss_mask_random(args, model, batch, running_loss):
    labels = batch["labels"][..., 1:].contiguous()
    logits = model(**batch).logits[..., :-1, :].contiguous()
    sample_loss = loss_fn(logits.transpose(1, 2), labels).mean(1)

    running_loss.append(sample_loss.mean().item())
    
    # random mask
    loss_mask = (torch.rand(labels.shape) < args.p).to(labels.device)
    sample_loss = sample_loss * loss_mask

    return sample_loss.mean()

def loss_mask_loss(args, model, batch, running_loss):
    ''' high loss mask '''
    labels = batch["labels"][..., 1:].contiguous()
    logits = model(**batch).logits[..., :-1, :].contiguous()
    sample_loss = loss_fn(logits.transpose(1, 2), labels).mean(1)

    running_loss.append(sample_loss.mean().item())

    # Compute the 30th percentile value
    percentile_value = np.quantile(running_loss, 1 - args.p)
    
    # Create a mask where True indicates that the loss is below the 30th percentile
    loss_mask = sample_loss >= percentile_value
    sample_loss = sample_loss * loss_mask

    return sample_loss.mean()


def loss_mask_conf(args, model, batch, running_loss):
    ''' low conf mask '''
    labels = batch["labels"][..., 1:].contiguous()
    logits = model(**batch).logits[..., :-1, :].contiguous()
    sample_loss = loss_fn(logits.transpose(1, 2), labels).mean(1)

    running_loss.append(sample_loss.mean().item())

    # Extract the predicted probabilities of the next token for each sample
    probs = torch.softmax(logits, dim=-1)
    predicted_probs = probs.gather(2, labels[..., None]).squeeze(-1)
    
    # Calculate the confidence as the max predicted probability for each token in the sequence
    entro = Normal(predicted_probs.mean(dim=-1), predicted_probs.std(dim=-1)).entropy()
    
    # Determine the samples with low confidence on its prediction for the next token
    entro_thre = torch.quantile(entro, 1 - args.p)
    low_confidence_mask = entro > entro_thre

    loss_mask = low_confidence_mask
    sample_loss = sample_loss * loss_mask

    return sample_loss.mean()

def loss_ascd_random(args, model, batch, running_loss):
    # outputs = model(**batch)
    # loss = outputs.loss
    # running_loss.append(loss.item())
    # if random.random() < args.p:
    #     loss = - loss
    # return loss
    labels = batch["labels"][..., 1:].contiguous()
    logits = model(**batch).logits[..., :-1, :].contiguous()
    sample_loss = loss_fn(logits.transpose(1, 2), labels).mean(1)

    running_loss.append(sample_loss.mean().item())
    
    # random mask
    loss_mask = (torch.rand(labels.shape[0]) < args.p).to(labels.device)
    sample_loss[loss_mask] = - sample_loss[loss_mask]

    return sample_loss.mean()

def loss_ascd_loss(args, model, batch, running_loss):
    ''' small loss ascd '''
    labels = batch["labels"][..., 1:].contiguous()
    logits = model(**batch).logits[..., :-1, :].contiguous()
    sample_loss = loss_fn(logits.transpose(1, 2), labels).mean(1)

    running_loss.append(sample_loss.mean().item())

    # Compute the 30th percentile value
    percentile_value = np.quantile(running_loss, args.p)
    
    # Create a mask where True indicates that the loss is below the 30th percentile
    loss_mask = sample_loss <= percentile_value
    sample_loss[loss_mask] = - sample_loss[loss_mask]

    return sample_loss.mean()

def loss_ascd_conf(args, model, batch, running_loss):
    ''' high conf ascd '''
    labels = batch["labels"][..., 1:].contiguous()
    logits = model(**batch).logits[..., :-1, :].contiguous()
    sample_loss = loss_fn(logits.transpose(1, 2), labels).mean(1)

    running_loss.append(sample_loss.mean().item())

    # Extract the predicted probabilities of the next token for each sample
    probs = torch.softmax(logits, dim=-1)
    predicted_probs = probs.gather(2, labels[..., None]).squeeze(-1)
    
    # Calculate the confidence as the max predicted probability for each token in the sequence
    entro = Normal(predicted_probs.mean(dim=-1), predicted_probs.std(dim=-1)).entropy()
    
    # Determine the samples with low confidence on its prediction for the next token
    entro_thre = torch.quantile(entro, args.p)
    high_confidence_mask = entro < entro_thre

    loss_mask = high_confidence_mask
    sample_loss[loss_mask] = -sample_loss[loss_mask]

    return sample_loss.mean()

def loss_flip(args, model, batch, running_loss):
    ''' flip y with args.mode '''
    labels = batch['labels'] # labels.shape = [batch_size, sequence_length]
    logits = model(**batch).logits # logits.shape = [batch_size, sequence_length, vocab_size]

    # assert args.flip_p is not None
    if args.mode[:3] == 'rnd':
        mask = torch.rand(labels.shape, device=logits.device) < args.flip_p  # Random mask
    elif args.mode[:4] == 'conf': 
        ''' high conf flip '''
        probs = F.softmax(logits, dim=-1)
        # Compute the entropy
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # entropy.shape = [batch_size, sequence_length]
        # Optional: Compute the mean entropy across tokens for each sample
        mean_entropy_per_sample = torch.mean(entropy, dim=-1)  # mean_entropy_per_sample.shape = [batch_size]
        threshold = np.quantile(mean_entropy_per_sample, 1 - args.p)
        mask = mean_entropy_per_sample > threshold
    else:
        raise ValueError

    if args.mode.split('_')[-1] == 'sim':
        _, indices = torch.topk(logits, args.sim_k, dim=-1)
        new_labels = indices[..., -1]
    elif args.mode.split('_')[-1] == 'unsim':
        _, indices = torch.topk(-logits, args.sim_k, dim=-1)
        new_labels = indices[..., -1]
    elif args.mode.split('_')[-1] == 'rnd':
        vocab_size = logits.size(-1)
        new_labels = torch.randint(0, vocab_size, labels.shape, device=logits.device)

    # Apply the mask to change only flip_p proportion of the tokens
    labels = torch.where(mask, new_labels, labels)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fn_mean(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    running_loss.append(loss.item())

    return loss

def loss_rdrop(args, model, batch, running_loss):
    '''
    assertations:
        args.attack == 'rdrop'
        args.dropout_pre is not None
    '''
    def clone_merge_tensor(tensor):
        # double batch size 
        return torch.cat([tensor]*2, dim=0)
    
    real_bz = batch['labels'].shape[0]

    input_ids = clone_merge_tensor(batch['input_ids'])
    attention_mask = clone_merge_tensor(batch['attention_mask'])
    labels = clone_merge_tensor(batch['labels']) # labels.shape = [batch_size, sequence_length]

    logits = model(input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels).logits # logits.shape = [batch_size * 2, sequence_length, vocab_size]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    score1 = logits[real_bz:, :]
    score2 = logits[:real_bz, :]
    loss_kl1 = loss_fn_kl(F.log_softmax(score1, dim=1), F.softmax(score2, dim=1))
    loss_kl2 = loss_fn_kl(F.log_softmax(score2, dim=1), F.softmax(score1, dim=1))
    loss_kl = (loss_kl1 + loss_kl2) / 2

    loss = loss + loss_kl

    running_loss.append(loss.item())

    return loss


import hashlib
import time
import math
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader

#### for data extraction ####
def data_extract(
        args,
        model,
        model_ref,
        accelerator,
        aux_dataset,
        train_dataset,

):
    model, model_ref = accelerator.prepare(model, model_ref)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    class PrefixDataset(Dataset):
        def __init__(self, prefix_list, device):
            self.prefix_list = torch.tensor(prefix_list)
            self.device = device

        def __len__(self):
            return len(self.prefix_list)

        def __getitem__(self, index):
            return {"input_ids": self.prefix_list[index].to(self.device),
              "attention_mask": torch.ones(self.prefix_list[index].shape).to(self.device),
              "labels": self.prefix_list[index].to(self.device)}
        
    def get_batch(input_ids):
        return {"input_ids": input_ids,
              "attention_mask": torch.ones(input_ids.shape).to(input_ids.device),
              "labels": input_ids}

    def calculate_mia(tokenize_input, model, model_ref):
        with torch.no_grad():  # 不计算梯度以节省内存和计算资源
            batch = get_batch(tokenize_input)
            labels = batch["labels"][..., 1:].contiguous()
            logits = model(**batch).logits[..., :-1, :].contiguous()
            sample_loss = loss_fn(logits.transpose(1, 2), labels).mean(1)
            perplexity = torch.exp(sample_loss)

            logits2 = model_ref(**batch).logits[..., :-1, :].contiguous()
            sample_loss2 = loss_fn(logits2.transpose(1, 2), labels).mean(1)
            perplexity2 = torch.exp(sample_loss2)

        return (perplexity2 - perplexity).cpu().tolist()

    def get_prefix(ds, L, max_len, min_freq=2):
        if L > 0:
            # Hash and count all L-length subsequences
            subsequence_counts = Counter()
            mapping = {}
            for item in ds:
                item = item['input_ids']
                for i in range(len(item) - L + 1):
                    subsequence = item[i:i+L]
                    hash_value = hashlib.md5(str(subsequence).encode()).hexdigest()
                    subsequence_counts[hash_value] += 1
                    mapping[hash_value] = subsequence

            # Filter and sort subsequences by frequency
            frequent_subsequences = [seq for seq, count in subsequence_counts.items() if count > min_freq]
            frequent_subsequences.sort(key=lambda x: subsequence_counts[x], reverse=True)

            # Return top n frequent subsequences
            if len(frequent_subsequences) > max_len:
                frequent_subsequences = frequent_subsequences[:max_len]
            else:
                import random
                # Randomly select a key
                frequent_subsequences = random.choices([seq for seq, count in subsequence_counts.items()], k=max_len)
            return [mapping[x] for x in frequent_subsequences]
        else:
            dummpy_prefix = tokenizer.encode(" ", return_tensors="pt").tolist()
            return dummpy_prefix * max_len
            
    if args.prefix_mode == 'aux_dup':
        prefix_list = get_prefix(aux_dataset, args.prefix_len, args.prefix_num, args.min_freq)
        prefix_ds = PrefixDataset(prefix_list, model.device)
        prefix_dl = DataLoader(prefix_ds, batch_size=args.per_device_train_batch_size*2)
    elif args.prefix_mode == 'train_dup':
        prefix_list = get_prefix(train_dataset, args.prefix_len, args.prefix_num, args.min_freq)
        prefix_ds = PrefixDataset(prefix_list, model.device)
        prefix_dl = DataLoader(prefix_ds, batch_size=args.per_device_train_batch_size*2)
    else:
        raise ValueError

    output_list = []
    mia_score_list = []

    start = time.time()
    for batch in tqdm(prefix_dl):
        batch_outputs = model.generate(batch['input_ids'], 
                                    max_length=args.suffix_len,
                                    do_sample=args.sampling_k is not None,
                                    top_k=args.sampling_k,
                                    num_return_sequences=math.ceil(args.num_gen / args.prefix_num))
        
        output_list.extend(batch_outputs.cpu().tolist())
        torch.cuda.empty_cache()  # 清理 GPU 缓存

        # calculate the perplexity of batch_outputs on model and model_ref, and append the difference of [ppl_ref - ppl] into mia_score_list
        mia_score_list.extend(calculate_mia(batch_outputs, model, model_ref))
        torch.cuda.empty_cache()  # 清理 GPU 缓存

    print(f'time = {time.time() - start}/s, expected {int(10000/args.num_gen) * (time.time() - start) / 3600}/h for 10K samples')
    print(len(output_list), len(mia_score_list))

    # given two list output_list and mia_score_list, please filter by ranking the mia_score_list and only keep output_list elements with top-k mia_Score
    def filter_sequences(token_ids_list, threshold=0.5):
        """
        过滤掉具有异常多重复token的序列。

        :param token_ids_list: 包含多个序列的列表,每个序列是token ID的列表。
        :param threshold: 重复token的比例阈值。
        :return: 过滤后的序列列表。
        """
        filtered_list = []

        for sequence in token_ids_list:
            unique_tokens = set(sequence[1])
            if len(unique_tokens) / len(sequence[1]) >= threshold:
                filtered_list.append([sequence[0], sequence[1]])

        return filtered_list
    
    def create_hash_table(ds, L):
        hash_table = set()
        for sequence in ds:
            sequence = sequence['input_ids']
            for i in range(len(sequence) - L + 1):
                substring = sequence[i:i+L]
                hash_value = hashlib.md5(str(substring).encode()).hexdigest()
                hash_table.add(hash_value)
        return hash_table

    def get_ratio_token(gen_seq, ds, c):
        hash_table = create_hash_table(ds, c)
        cnt = 0
        cnt_tot = 1e-6
        for seq in gen_seq:
            for i in range(len(seq) - args.threshold_token + 1):
                substring = seq[i:i+args.threshold_token]
                hash_value = hashlib.md5(str(substring).encode()).hexdigest()
                if hash_value in hash_table:
                    cnt += 1
                cnt_tot += 1
        return cnt / cnt_tot

    # 步骤 1: 联合两个列表
    combined_list = list(zip(mia_score_list, output_list))

    # filter out sequences with many repeated tokens
    combined_list = filter_sequences(combined_list, threshold=0.1)

    # 步骤 2: 根据 mia_score 排序
    combined_list.sort(key=lambda x: x[0], reverse=True)  # 假设更高的 mia_score 是更好的

    # 步骤 3: 选择 top-k
    top_k_combined = combined_list[:args.k]

    # 步骤 4: 分离输出
    top_k_outputs = [item[1] for item in top_k_combined]
    ratio_token = get_ratio_token(top_k_outputs, train_dataset, args.threshold_token)
    return ratio_token