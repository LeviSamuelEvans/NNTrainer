from . import LearnedPositionalEncoding
from . import ResidualBlock

import torch
import torch.nn as nn


class TransformerClassifier4(nn.Module):
    """Transformer-based classifier model 4.

    Parameters
    ----------
        input_dim (int):
            The dimension of the input features.
        d_model (int):
            The dimension of the transformer model.
        nhead (int):
            The number of attention heads in the transformer model.
        num_layers (int):
            The number of transformer layers.
        dropout (float, optional):
            The dropout probability. Default is 0.1.
    """

    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerClassifier4, self).__init__()

        # NEW: update input embedding to use nn.Sequential which also applied a layer normalisation
        self.input_embedding = (
            nn.Sequential(  # NEW - use nn.Sequential to combine layers
                nn.Linear(input_dim, d_model), nn.LayerNorm(d_model)
            )
        )
        # NEW: Use the learned positional encoding
        self.pos_encoder = LearnedPositionalEncoding(d_model, dropout, max_len=1000)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True
        )

        # NEW: Use nn.Sequential to combine the transformer encoder and layer normalisation
        self.transformer_encoder = nn.Sequential(
            nn.TransformerEncoder(encoder_layers, num_layers), nn.LayerNorm(d_model)
        )
        self.attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # NEW:
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),  # nn.ReLU(),
            nn.Dropout(dropout),
            # ResidualBlock(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            # ResidualBlock(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        pooled_output, _ = self.attention_pooling(x, x, x)
        pooled_output = pooled_output.mean(dim=1)
        x = self.classifier(pooled_output)

        return x


""""
NOTES:
------

Updated to:
    - Learned positional encoding -> (from LearnedPositionalEncoding class in ../Embedding/Learned.py)
        -- pass pooled features through the clasifier head.
    - layer normalisation after the transformer encoder layers.
    - use of nn.Sequential to combine layers and apply layer normalisation.
    - residual block in the classifier head -> (from ResidualBlock class in ../layers/residual.py)
"""
