import torch
import torch.nn as nn

from . import LearnedPositionalEncoding
from . import ResidualBlock

class TransformerClassifier5(nn.Module):
    """Transformer-based classifier model 5 with cross-attention."""

    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerClassifier5, self).__init__()

        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.pos_encoder = LearnedPositionalEncoding(d_model, dropout, max_len=1000)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True
        )

        self.transformer_encoder = nn.Sequential(
            nn.TransformerEncoder(encoder_layers, num_layers),
            nn.LayerNorm(d_model)
        )

        # NEW: Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # NEW: Key-value projection layer to match latent dimensions
        self.kv_projection = nn.Linear(input_dim, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.GELU(), # NEW: trying out based on https://arxiv.org/pdf/2401.00452.pdf
            nn.Dropout(dropout),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, key_value_pairs):
        """
        NOTES:
        -----
        For now, we pass the same input to the cross-attention layer as the query, key, and value.
        i.e the key_value pairs are also the input four-vectors x.

        However, here we could insert additional features to the key_value_pairs, which could be used
        to inject that additonal information into the model.
            - decay angles
            - btag?
            - met
            - Higher-level features
            - etc.

        The paper above uses this for multi-scale training, using substructure variables
        and jet-level variables, with the cross-attention layer used to combine the two.

        """
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        key_value_pairs_projected = self.kv_projection(key_value_pairs)

        # NEW: Apply cross-attention
        cross_attention_output, _ = self.cross_attention(x, key_value_pairs_projected, key_value_pairs_projected)
        x = x + cross_attention_output

        pooled_output, _ = self.attention_pooling(x, x, x)
        pooled_output, _ = torch.max(pooled_output, dim=1)
        x = self.classifier(pooled_output)

        return x