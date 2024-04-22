import torch
import torch.nn as nn
import math

class LorentzInvariantPositionalEncodingv2(nn.Module):
    """Lorentz-invariant positional encoding module for Transformer models."""

    def __init__(self, d_model, dropout):
        super(LorentzInvariantPositionalEncodingv2, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        # NOTE: try remove this transformation layer in the debugging
        self.transform = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LeakyReLU(0.01), # we use a leaky ReLU activation function here to preserve negative values
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, x_coords):
        batch_size, seq_len = x.size(0), x.size(1)

        # Compute Lorentz-invariant distances
        x_coords_diff = x_coords.unsqueeze(1) - x_coords.unsqueeze(2)
        epsilon = 1e-6
        # compute the Lorentz distances and clip to avoid NaNs
        lorentz_distances = torch.sum(x_coords_diff[:, :, :, 1:] ** 2, dim=-1) - x_coords_diff[:, :, :, 0] ** 2
        lorentz_distances = torch.sqrt(torch.clamp(lorentz_distances, min=0.0) + epsilon)

        print(lorentz_distances[:10])

        # reshape to (batch_size * seq_len, seq_len, 1)
        lorentz_distances = lorentz_distances.view(batch_size * seq_len, seq_len, 1)

        # apply Lorentz-invariant transformation to positional embeddings and add to input tensor
        pos_emb = self.transform(lorentz_distances)
        pos_emb = pos_emb.view(batch_size, seq_len, seq_len, self.d_model)

        # sum over the second sequence dimension to get the final positional embeddings
        pos_emb = pos_emb.sum(dim=2)

        x = x + pos_emb
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x
