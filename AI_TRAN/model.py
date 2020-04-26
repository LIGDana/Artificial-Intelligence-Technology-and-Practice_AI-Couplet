import math
from typing import Optional

import torch
import torch.nn as nn

class TraForEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
        self.mapper = nn.Linear(embed_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(layer, 4)

    def forward(self, x, key_padding_mask):
        embeds = self.embedding(x)
        encoderin = self.mapper(embeds).transpose(0, 1)
        out = self.encoder(encoderin, src_key_padding_mask=key_padding_mask).transpose(0, 1)
        return self.hidden2tag(out)
