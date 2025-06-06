import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import *
from utils import fix_random_seed_as

class PositionalEmbedding(nn.Module):
    def __init__(self,max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        # Compute the positional encodings once in log space.

    def forward(self, x , is_sinusoidal=False):
        batch_size = x.size(0)
        seq_len = x.size(1)
        if is_sinusoidal:
            embedding_tensor = torch.zeros(self.max_len, self.d_model)
            for i in range(self.max_len):
                for j in range(self.d_model):
                    if j % 2 == 0:
                        embedding_tensor[i, j] = math.sin(i / (10000 ** (j / self.d_model)))
                    else:
                        embedding_tensor[i, j] = math.cos(i / (10000 ** ((j -1) / self.d_model)))
        else:
           embedding_tensor = nn.Embedding(self.max_len, self.d_model)
        
        return torch.nn.functional.embedding(x, embedding_tensor.weight.to(x.device))

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size)
        self.weight.data.uniform_(-0.02, 0.02) # set initialization range

class Space_Partitioning_Embedding(nn.Module):
    def __init__(self,hidden_size,buckle_list, factor_list):
        super().__init__()
        self.hidden_size = hidden_size
        self.buckle_list = buckle_list
        self.factor_list = factor_list
        self.embedding_table_list, self.factor_table_list = self.partitioning()
     
    def partitioning(self):
        self.embedding_table_list = nn.ModuleList()
        self.factor_table_list = []
        for i in range(len(self.buckle_list)):
            buckle = self.buckle_list[i]
            lower = buckle[0]
            upper = buckle[1]
            
            embedding_size = self.factor_list[i]
            embedding_table = nn.Embedding(upper - lower + 1,embedding_size,padding_idx=0) #maybe phải xem lại 
            nn.init.trunc_normal_(embedding_table.weight, mean=0, std=0.02,a = -1, b = 1)
            self.embedding_table_list.append(embedding_table)

            factor_table = torch.empty(embedding_size, self.hidden_size)
            nn.init.trunc_normal_(factor_table, mean=0, std=0.02,a = -1, b = 1)

            self.factor_table_list.append(factor_table)
        
        return self.embedding_table_list, self.factor_table_list
    
    def forward(self,input_ids):
        embedding_output = []
        for i in range(len(self.embedding_table_list)):
            buckle = self.buckle_list[i]
            lower = buckle[0]
            upper = buckle[1]
            embedding_table = self.embedding_table_list[i]
            factor_table = self.factor_table_list[i]
            self.factor_table_list[i] = factor_table.to(input_ids.device)

            condition_mask1 = input_ids >= lower
            condition_mask2 = input_ids <= upper
            mask1 = condition_mask1.to(torch.int32)
            mask2 = condition_mask2.to(torch.int32)

            mask = mask1*mask2 
            mask_2d = mask.unsqueeze(-1).repeat(1,1,self.hidden_size).to(torch.float32)
            embedding_factor = embedding_table((input_ids - lower) * mask)
            
            if i == 0:
                embedding = embedding_factor * mask_2d
            else:
                embedding = torch.matmul(embedding_factor, self.factor_table_list[i]) * mask_2d
            embedding_output.append(embedding)

        embedding_output = torch.stack(embedding_output, dim=0).sum(dim=0)

        return embedding_output

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        3. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, args):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()

        self.token_embed = Space_Partitioning_Embedding(args.hidden_size,args.buckle_list,args.factor_list)
        self.value_embed = TokenEmbedding(vocab_size=15 , embed_size=args.hidden_size)
        self.count_embed = TokenEmbedding(vocab_size=15, embed_size=args.hidden_size)
        self.position_embed = PositionalEmbedding(max_len=args.max_seq_length , d_model=args.hidden_size)
        self.io_embed = SegmentEmbedding(embed_size=args.hidden_size)

        self.dropout = nn.Dropout(p=args.hidden_dropout_prob)

    def forward(self, input_ids, counts, values, io_flags, positions):
        x = self.token_embed(input_ids) + self.count_embed(counts) + self.position_embed(positions) + self.io_embed(io_flags) + self.value_embed(values)
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.bert_max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

        self.out = nn.Linear(self.hidden, args.num_items + 1)

    def forward(self, x):
        mask = (x > 1).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        x = self.out(x)

        return x

    def init_weights(self):
        pass

class BERT4ETH_cross_sharing(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()
        # embedding for BERT, sum of positional, segment, token embeddings
        self.args= args
        self.embedding = BERTEmbedding(args)

        # multi-layers transformer blocks, deep network
        self.transformer =TransformerBlock(args.hidden_size,
                              args.num_attention_heads,
                              args.hidden_size * 4,
                              args.hidden_dropout_prob)
            
        # self.out = nn.Linear(config["hidden_size"], config["vocab_size"])

    def forward(self, input_ids, counts, values, io_flags, positions):

        mask = (input_ids > 1).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(input_ids, counts, values, io_flags, positions)

        # running over multiple transformer blocks
        for _ in range(self.args.num_hidden_layers):
            x = self.transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass

class BERT4ETH(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()
        # embedding for BERT, sum of positional, segment, token embeddings
        self.args= args
        self.embedding = BERTEmbedding(args)

        # multi-layers transformer blocks, deep network
        self.transformer = nn.ModuleList([TransformerBlock(args.hidden_size,
                                                           args.num_attention_heads,
                                                           args.hidden_size * 4,
                                                           args.hidden_dropout_prob) for _ in range(args.num_hidden_layers)])
            

        # self.out = nn.Linear(config["hidden_size"], config["vocab_size"])

    def forward(self, input_ids, counts, values, io_flags, positions):

        mask = (input_ids > 1).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(input_ids, counts, values, io_flags, positions)

        # running over multiple transformer blocks
        for transformer in self.transformer:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass




