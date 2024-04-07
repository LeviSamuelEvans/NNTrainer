import torch
import torch.nn as nn
import math
import numpy as np
from . import LorentzInvariantAttention
from . import PositionalEncoding

"""
Notes: x_coords represents the coordinate features of the objects in the higher dimensional space,
that preserves Lorentz invariance.

"""
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
            #nn.Sigmoid(),
        )

    def forward(self, x, x_coords):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.attention(x, x_coords)
        x = x.mean(dim=1)  # Global average pooling over the sequence dimension
        output = self.classifier(x)
        return output
