import torch
import torch.nn as nn
import math

class LorentzInvariantPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LorentzInvariantPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, x_coords):
       # compute the lorentz invariant distances between each pair of positions
        x_coords_diff = x_coords.unsqueeze(1) - x_coords.unsqueeze(2)
        lorentz_distances = torch.sqrt(torch.sum(x_coords_diff[:, :, :, 1:] ** 2, dim=-1) - x_coords_diff[:, :, :, 0] ** 2)

        # now compute the position embeddings based on the lorentz invariant distances
        pos_emb = torch.zeros_like(lorentz_distances)
        pos_emb[:, :, :] = self.pe[:lorentz_distances.size(1), :].unsqueeze(1)

        x = x + pos_emb
        return self.dropout(x)
