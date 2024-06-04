import torch
import torch.nn as nn
import math


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
