import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x):
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


class TransformerClassifier1(nn.Module):
    """
    Transformer-based classifier model 1.

    Args:
        input_dim (int): The dimension of the input features.
        d_model (int): The dimension of the transformer model.
        nhead (int): The number of attention heads in the transformer model.
        num_layers (int): The number of transformer layers.
        dropout (float, optional): The dropout probability. Default is 0.1.

    Attributes:
        input_embedding (nn.Linear): Linear layer for input feature embedding.
        pos_encoder (PositionalEncoding): Positional encoding layer.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder layer.
        classifier (nn.Sequential): Sequential layers for classification.

    """

    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
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
            nn.Linear(d_model, 1),
            # nn.Sigmoid() # REMOVE AS USING BCEwithLogitsLoss when balancing classes
        )

    def forward(self, x):
        """
        Forward pass of the TransformerClassifier1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).

        """

        x = self.input_embedding(x)  # notes: embed input features to d_model dimensions
        x = self.pos_encoder(x)      # notes: apply positional encoding
        x = self.transformer_encoder(x)  # notes: pass through transformer encoder

        # global average pooling over the sequence dimension (i.e. using mean)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x

