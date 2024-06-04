import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, x, pos_embeddings):
        attn_output, _ = self.attention(x, pos_embeddings, pos_embeddings)
        return attn_output
