import torch
import torch.nn as nn
import math
import numpy as np

"""
Notes: x_coords represents the coordinate features of the objects in the higher dimensional space,
that preserves Lorentz invariance.

"""


class PositionalEncoding(nn.Module):
    """Positional encoding module for Transformer models.

    Parameters
    ----------
        d_model : int
            The dimension of the input feature.
        dropout : float, optional
            The dropout probability. Default: 0.1.

    Inputs
    ------
        x : torch.Tensor
            The input tensor of shape (batch_size, seq_len, d_model).

    Outputs
    ------
        torch.Tensor
            The output tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, d_model, dropout=0.1):
        """Initialises the PositionalEncoding module."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x):
        """Forward pass of the PositionalEncoding module."""
        batch_size, seq_len = x.size(0), x.size(1)
        position = (
            torch.arange(0, seq_len, dtype=torch.float, device=x.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        position_encoding = torch.zeros(
            batch_size, seq_len, self.d_model, device=x.device
        )

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device)
            * (-math.log(10000.0) / self.d_model)
        )
        position_encoding[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        position_encoding[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)

        x = x + position_encoding
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x


class LorentzInvariantAttention(nn.Module):
    """Lorentz invariant attention module for Transformer models.

    Parameters
    ----------
        d_model : int
            The dimension of the input feature.
        nhead : int
            The number of attention heads.
        dropout : float, optional
            The dropout probability. Default: 0.1.

    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(LorentzInvariantAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.minkowski = torch.from_numpy(
            np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )

    def psi(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    def innerprod(self, x1, x2):
        return torch.sum(
            torch.mul(torch.matmul(x1, self.minkowski), x2), dim=-1, keepdim=True
        )

    def forward(self, x, x_coords):
        # ensure metric is on same device as 4-vecs
        self.minkowski = self.minkowski.to(x_coords.device)

        innerprod_result = self.innerprod(x_coords, x_coords)
        psi_innerprod_result = self.psi(innerprod_result)
        diff_coords = x_coords[:, None, :] - x_coords[:, :, None]
        innerprod_diff_coords = self.innerprod(diff_coords, diff_coords)

        diag_innerprod_diff_coords = torch.diagonal(
            innerprod_diff_coords, dim1=-2, dim2=-1
        )
        diag_innerprod_diff_coords = diag_innerprod_diff_coords.transpose(-1, -2)
        psi_diag_innerprod_diff_coords = self.psi(diag_innerprod_diff_coords)

        x_lorentz = torch.cat(
            [
                innerprod_result,
                innerprod_result,
                psi_innerprod_result,
                psi_diag_innerprod_diff_coords.transpose(-1, -2),
            ],
            dim=-1,
        )
        # repeat x_lorentz to match the shape of x
        x_lorentz = x_lorentz.repeat(1, 1, x.shape[-1] // x_lorentz.shape[-1])
        x = x + x_lorentz
        return self.self_attn(x, x, x)[0]


class TransformerClassifier2(nn.Module):
    """Transformer classifier 2 module."""
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerClassifier2, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.attention = LorentzInvariantAttention(d_model, nhead, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, x_coords):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.attention(x, x_coords)
        x = x.mean(dim=1)  # Global average pooling over the sequence dimension
        output = self.classifier(x)
        return output
