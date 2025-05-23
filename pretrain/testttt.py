import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import collections
import functools
import random
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import six
import time
import math
from vocab import FreqVocab

tf.logging.set_verbosity(tf.logging.INFO)

random_seed = 12345
rng = random.Random(random_seed)

short_seq_prob = 0  # Probability of creating sequences which are shorter than the maximum length。
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("pool_size", 10, "multiprocesses pool size.")
flags.DEFINE_integer("max_seq_length", 29, "max sequence length.")
flags.DEFINE_float("masked_lm_prob", 0.8, "Masked LM probability.")
flags.DEFINE_float("mask_prob", 1.0, "mask probabaility")
flags.DEFINE_bool("do_eval", False, "")
flags.DEFINE_bool("do_embed", True, "")
flags.DEFINE_integer("dupe_factor", 10, "Number of times to duplicate the input data (with different masks).")
flags.DEFINE_string("data_dir", './data/', "data dir.")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")
flags.DEFINE_bool("total_drop", True, "whether to drop")
flags.DEFINE_bool("drop", False, "whether to drop")

HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(
    ",")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

MAX_PREDICTIONS_PER_SEQ = math.ceil(FLAGS.max_seq_length * FLAGS.masked_lm_prob)
SLIDING_STEP = round(FLAGS.max_seq_length * 0.6)

print("MAX_SEQUENCE_LENGTH:", FLAGS.max_seq_length)
print("MAX_PREDICTIONS_PER_SEQ:", MAX_PREDICTIONS_PER_SEQ)
print("SLIDING_STEP:", SLIDING_STEP)

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, address, tokens, masked_lm_positions, masked_lm_labels):

        self.address = [address]
        self.tokens = list(map(lambda x: x[0], tokens))
        self.block_timestamps = list(map(lambda x: x[2], tokens))
        self.values = list(map(lambda x: x[3], tokens))

        def map_io_flag(token):
            flag = token[4]
            if flag == "OUT":
                return 1
            elif flag == "IN":
                return 2
            else:
                return 0

        self.io_flags = list(map(map_io_flag, tokens))
        self.cnts = list(map(lambda x: x[5], tokens))
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = "address: %s\n" % (self.address[0])
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def gen_samples(sequences,
                dupe_factor,
                masked_lm_prob,
                max_predictions_per_seq,
                pool_size,
                rng,
                force_head=False):
    instances = []
    # create train
    if force_head:
        for step in range(dupe_factor):
            start = time.time()
            for tokens in sequences:
                (address, tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions_force_head(tokens)
                instance = TrainingInstance(
                    address=address,
                    tokens=tokens,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            end = time.time()
            cost = end - start
            print("step=%d, time=%.2f" % (step, cost))
        print("=======Finish========")

    else:
        for step in range(dupe_factor):
            start = time.time()
            for tokens in sequences:
                (address, tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, rng)
                instance = TrainingInstance(
                    address=address,
                    tokens=tokens,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            end = time.time()
            cost = end - start
            print("step=%d, time=%.2f" % (step, cost))
        print("=======Finish========")
    return instances


def create_masked_lm_predictions_force_head(tokens):
    """Creates the predictions for the masked LM objective."""
    first_index = 0
    address = tokens[0][0]
    output_tokens = [list(i) for i in tokens]  # note that change the value of output_tokens will also change tokens
    output_tokens[first_index] = ["[MASK]", 0, 0, 0, 0, 0]
    masked_lm_positions = [first_index]
    masked_lm_labels = [tokens[first_index][0]]

    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, rng):
    """Creates the predictions for the masked LM objective."""

    address = tokens[0][0]
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)
    output_tokens = [list(i) for i in tokens]  # note that change the value of output_tokens will also change tokens
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(len(tokens) * masked_lm_prob)))
    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = "[MASK]"
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index][0]))
        output_tokens[index][0] = masked_token

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def create_embedding_predictions(tokens):
    """Creates the predictions for the masked LM objective."""
    address = tokens[0][0]
    output_tokens = tokens
    masked_lm_positions = []
    masked_lm_labels = []
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def gen_embedding_samples(sequences):
    instances = []
    # create train
    start = time.time()
    for tokens in sequences:
        (address, tokens, masked_lm_positions,
         masked_lm_labels) = create_embedding_predictions(tokens)
        instance = TrainingInstance(
            address=address,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    end = time.time()
    print("=======Finish========")
    print("cost time:%.2f" % (end - start))
    return instances


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