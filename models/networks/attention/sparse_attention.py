import torch
import torch.nn as nn


class SparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, sparsity_factor, dropout=0.1):
        super(SparseAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.sparsity_factor = sparsity_factor  # make this configurable

    def forward(self, query, key, value):
        # randomly drop attention connections
        batch_size, seq_len, _ = query.size()
        mask = torch.rand(batch_size, seq_len, seq_len) < self.sparsity_factor
        attn_output, attn_output_weights = self.attention(query, key, value, attn_mask=mask)
        return attn_output
