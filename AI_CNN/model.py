import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be divisible by 2")
        self.bilstm = nn.LSTM(input_dim, hidden_dim // 2,
                              num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x):
        return self.bilstm(x)


class CNNLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        cnn_in = x.permute(0, 2, 1)
        cnn_out = self.conv(cnn_in).permute(0, 2, 1)
        return F.relu(cnn_out)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: Optional[float] = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttentionLayer(nn.Module):
    """
    Reference: https://github.com/huggingface/transformers
    """

    def __init__(self, hidden_dim: int, num_heads: Optional[int] = 4, dropout_prob: Optional[float] = 0.2):
        super().__init__()
        self.num_att_heads = num_heads
        self.att_head_size = hidden_dim // num_heads
        self.all_head_size = hidden_dim

        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_att_heads, self.att_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.att_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class CNNBiLSTMAtt(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = CNNLayer(embed_dim, hidden_dim)
        self.bilstm = BiLSTMLayer(hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
        self.att = SelfAttentionLayer(hidden_dim)
        self.norm = LayerNorm(hidden_dim)

    def forward(self, x, exted_att_mask):
        embeds = self.embedding(x)
        cnn_out = self.conv(embeds)
        lstm_out, _ = self.bilstm(cnn_out)
        normout = self.norm(lstm_out + self.att(lstm_out, exted_att_mask))
        return self.hidden2tag(normout)