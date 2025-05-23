import numpy as np
import torch
import math
import random
import torch.utils.data as data_utils
import copy
import pandas as pd
from torch.utils.data import WeightedRandomSampler

def map_io_flag(tranxs):
    flag = tranxs[4]
    if flag == "OUT":
        return 1
    elif flag == "IN":
        return 2
    else:
        return 0

def convert_timestamp_to_position(block_timestamps):
    position = [0]
    if len(block_timestamps) <= 1:
        return position
    last_ts = block_timestamps[1]
    idx = 1
    for b_ts in block_timestamps[1:]:
        if b_ts != last_ts:
            last_ts = b_ts
            idx += 1
        position.append(idx)
    return position
    

def is_phishing(eoa, phisher_account):      
    if eoa in phisher_account:
        return 1
    return 0

class BERT4ETHDataloader:

    def __init__(self, args, vocab, eoa2seq):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.eoa2seq = eoa2seq
        self.vocab = vocab
        self.seq_list , self.labels = self.preprocess(eoa2seq)

    def preprocess(self, eoa2seq):
        self.masked_lm_prob = self.args.masked_lm_prob
        # self.rng = random.Random(self.args.dataloader_random_seed)
        self.rng = random.Random()
        self.sliding_step = round(self.args.max_seq_length * 0.6)

        phisher_account = pd.read_csv("/home/phinn/BERT4ETH/ZIPZAP/data/phisher_account.txt",names = ["account"])
        phisher_account = set(phisher_account.account.values)
        

        # preprocess
        length_list = []
        for eoa in eoa2seq.keys():
            seq = eoa2seq[eoa]
            length_list.append(len(seq))
         
        length_list = np.array(length_list)
        print("Median:", np.median(length_list))
        print("Mean:", np.mean(length_list))
        print("Seq num:", np.sum(length_list))

        # clip
        labels = []
        max_num_tokens = self.args.max_seq_length - 1
        seqs = []
        idx = 0
        for eoa, seq in eoa2seq.items():
            if len(seq) <= max_num_tokens:
                seqs.append([[eoa, 0, 0, 0, 0, 0]])
                seqs[idx] += seq
                idx += 1
                labels.append(is_phishing(eoa,phisher_account))
            elif len(seq) > max_num_tokens:
                beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * self.sliding_step))
                beg_idx.append(0)

                if len(beg_idx) > 500:
                    beg_idx = list(np.random.permutation(beg_idx)[:500])
                    for i in beg_idx:
                        seqs.append([[eoa, 0, 0, 0, 0, 0]])
                        seqs[idx] += seq[i:i + max_num_tokens]
                        idx += 1
                        labels.append(is_phishing(eoa,phisher_account))

                else:
                    for i in beg_idx[::-1]:
                        seqs.append([[eoa, 0, 0, 0, 0, 0]])
                        seqs[idx] += seq[i:i + max_num_tokens]
                        idx += 1
                        labels.append(is_phishing(eoa,phisher_account))
    
        return seqs , labels

    def get_train_loader(self):
        dataset = BERT4ETHTrainDataset(self.args, self.vocab, self.seq_list, self.labels)
        labels = dataset.labels
         # Chuyển labels thành tensor
        
        labels = torch.tensor(labels).view(-1)
        class_sample_count = torch.bincount(labels)
        
        weight_per_class = 1. / class_sample_count.float()

        weights = weight_per_class[labels] 

        sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),  
        replacement=True
    )      
        
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           sampler=sampler,shuffle=False, 
                                           pin_memory=True)
        return dataloader

    def get_eval_loader(self):
        dataset = BERT4ETHEvalDataset(self.args, self.vocab, self.seq_list, self.labels)
        labels = dataset.labels
         # Chuyển labels thành tensor
        
        labels = torch.tensor(labels).view(-1)
        class_sample_count = torch.bincount(labels)
        
        weight_per_class = 1. / class_sample_count.float()

        weights = weight_per_class[labels]  

        sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights), 
        replacement=True
    )      
        
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.eval_batch_size,
                                           sampler=sampler, shuffle=False, 
                                           pin_memory=True)

        return dataloader


class BERT4ETHTrainDataset(data_utils.Dataset):

    def __init__(self, args, vocab, seq_list, labels):
        # mask_prob, mask_token, max_predictions_per_seq):
        self.args = args
        self.seq_list = seq_list
        self.vocab = vocab
        seed = args.dataloader_random_seed
        # self.rng = random.Random(seed)
        self.rng = random.Random()
        self.labels = labels

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):

        # only one index as input
        tranxs = copy.deepcopy(self.seq_list[index])
        address = tranxs[0][0]

        # MAP discrete feature to int
        address_id = self.vocab.convert_tokens_to_ids([address])
        label = [self.labels[index]]
        tokens = list(map(lambda x: x[0], tranxs))
        input_ids = self.vocab.convert_tokens_to_ids(tokens)

        block_timestamps = list(map(lambda x: x[2], tranxs))
        values = list(map(lambda x: x[3], tranxs))
        io_flags = list(map(map_io_flag, tranxs))
        counts = list(map(lambda x: x[5], tranxs))
        positions = convert_timestamp_to_position(block_timestamps)
        input_mask = [1] * len(input_ids)

        max_seq_length = self.args.max_seq_length

        assert len(input_ids) <= max_seq_length
        assert len(counts) <= max_seq_length
        assert len(values) <= max_seq_length
        assert len(io_flags) <= max_seq_length
        assert len(positions) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        counts += [0] * (max_seq_length - len(counts))
        values += [0] * (max_seq_length - len(values))
        io_flags += [0] * (max_seq_length - len(io_flags))
        positions += [0] * (max_seq_length - len(positions))
        input_mask += [0] * (max_seq_length - len(input_mask))


        assert len(input_ids) == max_seq_length
        assert len(counts) == max_seq_length
        assert len(values) == max_seq_length
        assert len(io_flags) == max_seq_length
        assert len(positions) == max_seq_length
        assert len(input_mask) == max_seq_length

        return torch.LongTensor(address_id), \
               torch.LongTensor(label) ,\
               torch.LongTensor(input_ids), \
               torch.LongTensor(counts), \
               torch.LongTensor(values), \
               torch.LongTensor(io_flags), \
               torch.LongTensor(positions), \
               torch.LongTensor(input_mask)

class BERT4ETHEvalDataset(data_utils.Dataset):

    def __init__(self, args, vocab, seq_list , labels):
        # mask_prob, mask_token, max_predictions_per_seq):
        self.args = args
        self.seq_list = seq_list
        self.vocab = vocab
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.labels = labels
    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):

        # only one index as input
        tranxs = self.seq_list[index]
        address = tranxs[0][0]

        # MAP discrete feature to int
        address_id = self.vocab.convert_tokens_to_ids([address])
        label = [self.labels[index]]
        tokens = list(map(lambda x: x[0], tranxs))
        input_ids = self.vocab.convert_tokens_to_ids(tokens)

        block_timestamps = list(map(lambda x: x[2], tranxs))
        values = list(map(lambda x: x[3], tranxs))
        io_flags = list(map(map_io_flag, tranxs))
        counts = list(map(lambda x: x[5], tranxs))
        positions = convert_timestamp_to_position(block_timestamps)
        input_mask = [1] * len(input_ids)

        max_seq_length = self.args.max_seq_length

        assert len(input_ids) <= max_seq_length
        assert len(counts) <= max_seq_length
        assert len(values) <= max_seq_length
        assert len(io_flags) <= max_seq_length
        assert len(positions) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        counts += [0] * (max_seq_length - len(counts))
        values += [0] * (max_seq_length - len(values))
        io_flags += [0] * (max_seq_length - len(io_flags))
        positions += [0] * (max_seq_length - len(positions))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(counts) == max_seq_length
        assert len(values) == max_seq_length
        assert len(io_flags) == max_seq_length
        assert len(positions) == max_seq_length
        assert len(input_mask) == max_seq_length

        return torch.LongTensor(address_id), \
               torch.LongTensor(label),\
               torch.LongTensor(input_ids), \
               torch.LongTensor(counts), \
               torch.LongTensor(values), \
               torch.LongTensor(io_flags), \
               torch.LongTensor(positions), \
               torch.LongTensor(input_mask)

