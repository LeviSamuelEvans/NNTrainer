from . import LorentzEmbedding
from . import PositionalEncoding

import torch
import torch.nn as nn


class TransformerClassifier3(nn.Module):
    """Transformer-based classifier model 3.

    We use the LorenzEmbedding class to implement the positional encoding based on the Lorentz-invariant distances.
    """

    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerClassifier3, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        #self.pos_encoder = LorentzEmbedding(d_model, dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # NEW

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x): # x_coords):
        """Forward pass of the TransformerClassifier3 model.

        The pooling here is done using the attention mechanism.
        """
        x = self.input_embedding(x)
        #x = self.pos_encoder(x, x_coords)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        pooled_output, _ = self.attention_pooling(x, x, x)
        pooled_output = pooled_output.mean(dim=1)
        x = self.classifier(pooled_output)
        return x