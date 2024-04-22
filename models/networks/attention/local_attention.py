import torch
import torch.nn as nn


class LocalAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super(LocalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.window_size = window_size # make this configurable

    def forward(self, query, key, value):
        # local attention only within a window around each token
        batch_size, seq_len, _ = query.size()
        new_key = new_value = torch.zeros_like(key)
        for i in range(seq_len):
            left = max(0, i - self.window_size)
            right = min(seq_len, i + self.window_size + 1)
            new_key[:, i, :] = key[:, left:right, :].mean(dim=1)
            new_value[:, i, :] = value[:, left:right, :].mean(dim=1)
        return self.attention(query, new_key, new_value)
