from config import args
from gen_finetune_data import BERT4ETHDataloader
from modeling import BERT4ETH
from finetune_cascaded import BERT4ETHTrainer_finetune
import pickle as pkl
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader


def main():
 
    # prepare dataset
    
    print("===========Load Sequence===========")
    with open("/home/phinn/BERT4ETH/ZIPZAP/pretrain/outputs/eoa2seq_zipzap_exp.pkl","rb") as f:
        eoa2seq = pkl.load(f)
    
    with open("/home/phinn/BERT4ETH/ZIPZAP/pretrain/outputs/vocab.zipzap_exp","rb") as f:
        vocab = pkl.load(f)

    vocab.update(eoa2seq)
    vocab.generate_vocab()
     
    phisher_account = pd.read_csv("/home/phinn/BERT4ETH/ZIPZAP/data/phisher_account.txt",names = ["account"])
    phisher_account = set(phisher_account.account.values)

    def is_phish(eoa):
        if eoa in phisher_account:
            return 1
        return 0

    eoa_list = list(eoa2seq.keys())
    
    acc = list(eoa2seq.keys())
    labels = []
    for addr in acc:
        labels.append(is_phish(addr))
    
    print("LABEL", sum(labels))
    
    # split data
    data_train, data_eval = train_test_split(
        acc,
        test_size=0.2,
        stratify=labels,  # bảo toàn phân phối label
        random_state=42
    )

    new_seq_train = {eoa: eoa2seq[eoa] for eoa in data_train}
    new_seq_evaluation = {eoa: eoa2seq[eoa] for eoa in data_eval}
     
    dataloader_train = BERT4ETHDataloader(args, vocab, new_seq_train)
    train_loader = dataloader_train.get_train_loader()
    
    dataloader_eval = BERT4ETHDataloader(args, vocab, new_seq_evaluation)
    eval_loader = dataloader_eval.get_eval_loader()


    model = BERT4ETH(args)

    trainer = BERT4ETHTrainer_finetune(args, vocab , model, train_loader)
    trainer.load("/home/phinn/BERT4ETH/ZIPZAP/pretrain/zipzap_exp/epoch_5.pth")
    trainer.train()

    
    test_data = BERT4ETHTrainer_finetune(args, vocab, model, eval_loader)
    test_data.load("/home/phinn/BERT4ETH/ZIPZAP/finetune_local/epoch_1.pth")
    
    address_id_list,label_list = test_data.infer_embedding()
    y_hat_list = test_data.evaluate()
    
    print("ADDRESS ID LIST:", np.array(address_id_list).shape)
    print("LABEL LIST:", np.array(label_list).shape)
    print("Y HAT LIST:", np.array(y_hat_list).shape)
    
    address_id_list = np.array(address_id_list).reshape([-1])
    y_hat_list = np.array(y_hat_list).reshape([-1])
    label_list = np.array(label_list).reshape([-1])


    address_to_pred_proba = {}
    # address_to_label = {}
    for i in range(len(address_id_list)):
        address = address_id_list[i]
        pred_proba = y_hat_list[i]
        # label = label_list[i]
        try:
            address_to_pred_proba[address].append(pred_proba)
            # address_to_label[address].append(label)
        except:
            address_to_pred_proba[address] = [pred_proba]
            # address_to_label[address] = [label]

    # group to one
    address_list = []
    agg_y_hat_list = []
    agg_label_list = []

    for addr, pred_proba_list in address_to_pred_proba.items():
        address_list.append(addr)
        if len(pred_proba_list) > 1:
            agg_y_hat_list.append(np.mean(pred_proba_list, axis=0))
        else:
            agg_y_hat_list.append(pred_proba_list[0])

        agg_label_list.append(is_phish(addr))

    # print("================ROC Curve====================")
    fpr, tpr, thresholds = roc_curve(agg_label_list, agg_y_hat_list, pos_label=1)
    print("AUC=", auc(fpr, tpr))

    print(np.sum(agg_label_list))
    
    print(np.sum(agg_y_hat_list))

    # for threshold in [0.01, 0.03, 0.05]:
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:

        print("threshold =", threshold)
        y_pred = np.zeros_like(agg_y_hat_list)
        y_pred[np.where(np.array(agg_y_hat_list) >= threshold)[0]] = 1
        print(np.sum(y_pred))
        print(classification_report(agg_label_list, y_pred, digits=4))

    return

if __name__ == '__main__':
    main()

    



