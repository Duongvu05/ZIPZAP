from config import args
from dataloader import BERT4ETHDataloader
from modeling import BERT4ETH
from trainer import BERT4ETHTrainer
import pickle as pkl
from vocab import FreqVocab
import numpy as np

def main():

    # prepare dataset
    vocab = FreqVocab()
    print("===========Load Sequence===========")
    with open("/home/phinn/BERT4ETH/ZIPZAP/outputs/eoa2seq_zipzap_exp.pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))
    vocab.update(eoa2seq)
    # generate mapping
    vocab.generate_vocab()

    n_bucket = 10   
    total_freq = np.sum(vocab.frequency)
    print(len(vocab.frequency))
    unit_freq = total_freq/n_bucket

    offset = 0
    bucket_list = []

    for i in range(n_bucket):
        lower = offset
        count = 0
        for j in range(lower, len(vocab.frequency)):
            count += vocab.frequency[j]
            if count >= unit_freq or j == len(vocab.frequency)-1:
                upper = j
                break

        bucket_list.append([lower, upper])
        offset = upper + 1

    print(bucket_list)

    # save vocab
    print("token_size:{}".format(len(vocab.vocab_words)))
    vocab_file_name = args.data_dir + args.vocab_filename + "." + args.bizdate
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pkl.dump(vocab, output_file, protocol=2)

    # dataloader
    dataloader = BERT4ETHDataloader(args, vocab, eoa2seq)
    train_loader = dataloader.get_train_loader()

    # model
    model = BERT4ETH(args)

    # tranier
    trainer = BERT4ETHTrainer(args, vocab, model, train_loader)
    trainer.train()

if __name__ == '__main__':
    main()

