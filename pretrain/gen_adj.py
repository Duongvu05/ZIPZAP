from config import args
from modeling import BERT4ETH
import numpy as np
from sklearn.model_selection import train_test_split
import torch

import torch.nn.functional as F
import torch.nn as nn
import os
# ===== Built-in imports =====

import pickle as pkl


# ===== Third-party imports =====
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.data import Data

# ===== Local imports =====
from utils import AverageMeterSet, fix_random_seed_as
from vocab import FreqVocab

class BERT4ETH_PR_DATA(nn.Module):
    def __init__(self, args, vocab, eoa2seq):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.eoa2seq = eoa2seq
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.a0_weight = torch.rand([1,5],device = self.device,requires_grad= True)

    @staticmethod
    def cal_n_gram(eoa2seq, n_gram = 5):
        new_eoa2seq = {}
        for eoa in eoa2seq.keys():
            seq = copy.deepcopy(eoa2seq[eoa])
            for i in range(len(seq)):
                n_gram_seq = []
                trx = seq[i]
                block_timestamp = trx[2]
                for j in range(0,-n_gram, -1):
                    try: 
                        n_gram_seq.append(block_timestamp - seq[i - j + 1][2])
                    except:
                        n_gram_seq.append(0.0)
                seq[i] += n_gram_seq
            new_eoa2seq[eoa] = seq
            
        return new_eoa2seq

    @staticmethod
    def gen_adjacency_matrix(eoa2seq, vocab, a0_weight,device):
        rows, cols, data = [], [], []

        for eoa in eoa2seq.keys():
            seq = eoa2seq[eoa]
            for trx in seq:
                in_out = trx[4]
                value = trx[3]
                features = torch.tensor(trx[6:], dtype=torch.float32,device = device)

                weighted_value = value * torch.sum(a0_weight * features)

                if in_out == "OUT":
                    from_id = vocab.convert_tokens_to_ids([eoa])[0] - 3
                    to_id = vocab.convert_tokens_to_ids([trx[0]])[0] - 3
                else:
                    from_id = vocab.convert_tokens_to_ids([trx[0]])[0] - 3
                    to_id = vocab.convert_tokens_to_ids([eoa])[0] - 3

                rows.append(from_id)
                cols.append(to_id)
                data.append(weighted_value)

        return rows, cols, data

    def infer_adj(self,is_training = True):
        rows, cols, data = [], [], []
        num_split = 20
        split_size = len(self.eoa2seq) // num_split

        acc_list = list(self.eoa2seq.keys())
        for i in tqdm(range(num_split)):
            part_keys = acc_list[i*split_size : (i+1)*split_size]
            part_eoa2seq = {k: self.eoa2seq[k] for k in part_keys}
            part_eoa2seq = self.cal_n_gram(part_eoa2seq, n_gram=5)
            r, c, d = self.gen_adjacency_matrix(part_eoa2seq, self.vocab, self.a0_weight, self.device)
            try:
                rows += r
                cols += c
                data += d
                torch.cuda.empty_cache()
            except:
                torch.save(torch.tensor(rows, dtype=torch.long, device=self.device), "/home/phinn/BERT4ETH/ZIPZAP/outputs/rows.pt")
                torch.save(torch.tensor(cols, dtype=torch.long, device=self.device), "/home/phinn/BERT4ETH/ZIPZAP/outputs/cols.pt")
                torch.save(torch.tensor(data, dtype=torch.float32, device=self.device), "/home/phinn/BERT4ETH/ZIPZAP/outputs/data.pt")
                print(f"Error in generating adjacency matrix {i}")
                
        rows = torch.tensor(rows, dtype=torch.long, device=self.device)
        cols = torch.tensor(cols, dtype=torch.long, device=self.device)
        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        num_nodes = 3000000

        indices = torch.stack([rows, cols])
        values = torch.tensor(data, dtype=torch.float32)
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

        torch.save(adj, "/home/phinn/BERT4ETH/ZIPZAP/outputs/adjacency_matrix_training.pt")
        return
        
        

         
def main():
 
    # prepare dataset
    vocab = FreqVocab()
    print("===========Load Sequence===========")
    with open("/home/phinn/BERT4ETH/ZIPZAP/dynamic/eoa2seq_dynamic_exp_adj.pkl","rb") as f:
        eoa2seq = pkl.load(f)

    vocab.update(eoa2seq)
    vocab.generate_vocab()
     
    phisher_account = pd.read_csv("/home/phinn/BERT4ETH/ZIPZAP/data/phisher_account.txt",names = ["account"])
    phisher_account = set(phisher_account.account.values)

    def is_phish(eoa):
        if eoa in phisher_account:
            return 1
        return 0
    
    acc = list(eoa2seq.keys())

    labels = []
    for addr in acc:
        labels.append(is_phish(addr)) 
    
    trainer = BERT4ETH_PR_DATA(args,vocab,eoa2seq)
    trainer.infer_adj()
    print("Adjacency matrix generated")

    return

if __name__ == '__main__':
    main()

    



