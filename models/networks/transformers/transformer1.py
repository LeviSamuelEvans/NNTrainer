import torch
import torch.nn as nn
from . import PositionalEncoding


class TransformerClassifier1(nn.Module):
    """Transformer-based classifier model 1.

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

    Attributes:
    ----------
        input_embedding : nn.Linear
            Linear layer for input feature embedding.
        pos_encoder : PositionalEncoding
            Positional encoding layer.
        transformer_encoder : nn.TransformerEncoder
            Transformer encoder layer.
        classifier : nn.Sequential
            Sequential layers for classification.

    """

    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerClassifier1, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # creates a transformer encoder with num_layers layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            # nn.Sigmoid() # REMOVE AS USING BCEwithLogitsLoss when balancing classes
        )

    def forward(self, x):
        """Forward pass of the TransformerClassifier1 model."""

        x = self.input_embedding(x)  # notes: embed input features to d_model dimensions
        x = self.pos_encoder(x)  # notes: apply positional encoding
        x = self.transformer_encoder(x)  # notes: pass through transformer encoder

        # global average pooling over the sequence dimension (i.e. using mean)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x
