import argparse
import math

parser = argparse.ArgumentParser(description='ZipZap')
################
# Dataloader
################
parser.add_argument('--dataloader_random_seed', type=float, default=12345)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--ckpt_dir', default="ckpt_local", type=str)
parser.add_argument('--data_dir', type=str, default='../outputs/', help='data dir.')
parser.add_argument('--vocab_filename', type=str, default='vocab', help='vocab filename')
parser.add_argument("--dup_times", type=int, default=1, help= "data duplicate times")
parser.add_argument("--drop" , type = bool , default= False , help="whether to drop out, reducing the RS")
################
# Trainer
################
parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda'])
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--num_train_steps', default=10000000)
parser.add_argument('--num_warmup_steps', default=100)
parser.add_argument('--num_epochs', type=int, help='Number of epochs for training')
################
# Model
################
parser.add_argument('--model_init_seed', type=int, default=54321)
parser.add_argument('--masked_lm_prob', type=float, default=0.8, help='Masked LM probability.')
parser.add_argument('--neg_sample_num', type=int, default=5000, help='The number of negative samples in a batch')
parser.add_argument('--neg_strategy', type=str, default="zip", help='Strategy of negative sampling.')
parser.add_argument('--max_seq_length', type=int, default=56, help='max sequence length.')
parser.add_argument('--bizdate', type=str, default=None, help='the signature of running experiments')
parser.add_argument('--init_checkpoint', type=str, help='the directory name of checkpoint')
################
args = parser.parse_args()

def set_template(args):
    args.enable_lr_schedule = True
    args.decay_step = 25
    args.gamma = 1.0
    args.num_epochs = 50
    args.model_init_seed = 0

    # model configuration
    args.hidden_act = "gelu"
    args.hidden_size = 64
    args.initializer_range = 0.02
    args.num_hidden_layers = 8
    args.num_attention_heads = 2
    args.vocab_size = 3000000
    args.max_seq_length = 56
    args.hidden_dropout_prob = 0.2
    args.attention_probs_dropout_prob = 0.2
    args.buckle_list = [[0, 31], [32, 331], [332, 1587], [1588, 5748], [5749, 16901], [16902, 44753], [44754, 112405], [112406, 271216], [271217, 730116], [730117, 1899292]]
    args.factor_list = [64, 46, 32, 23, 16, 12, 8, 6, 4, 3]
 
    args.max_predictions_per_seq = math.ceil(args.max_seq_length * args.masked_lm_prob)
    args.sliding_step = round(args.max_seq_length * 0.6)

set_template(args)

print("==========Hyper-parameters============")
print("Epoch #:", args.num_epochs)
print("Vocab #:", args.vocab_size)
print("Hidden #:", args.hidden_size)
print("Max Length:", args.max_seq_length)
print("ckpt_dir:", args.ckpt_dir)
print("learning_rate:", args.lr)
print("Max predictions per seq:", args.max_predictions_per_seq)

