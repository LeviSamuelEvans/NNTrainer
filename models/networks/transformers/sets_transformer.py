from . import LorentzInvariantAttention
import torch.nn as nn
import torch
from . import ResidualBlock


class SetsTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        """Initialise  SetsTransformerEncoder.

        Parameters
        ----------
        d_model : int
            Dimension of the model.
        nhead : int
            Number of attention heads.
        num_layers : int
            Number of encoder layers.
        dropout : float
            Dropout probability.

        Refs
        -------
        https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html

        """
        super(SetsTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            SetsTransformerEncoderLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, x_coords):
        for layer in self.layers:
            x = layer(x, x_coords)
        return x

class SetsTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(SetsTransformerEncoderLayer, self).__init__()
        #self.attention = LorentzInvariantAttention(d_model, nhead, dropout)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, x_coords):
        # Applying self-attention, thus query, key, and value are all the same
        # make sure x is shaped (L, N, E) (Seq. length, Batch size, Embedding dim)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output
         # revert to original shape (N, L, E)
        x = x.permute(1, 0, 2)
        x = x + self.feed_forward(self.norm2(x))
        return x

class SetsTransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(SetsTransformerClassifier, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.set_transformer = SetsTransformerEncoder(d_model, nhead, num_layers, dropout)
        self.attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # NEW
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, x_coords):
        x = self.input_embedding(x)
        x = self.set_transformer(x, x_coords)

        # attention pooling
        pooled_output, _ = self.attention_pooling(x, x, x)
        pooled_output, _ = torch.max(pooled_output, dim=1)

        output = self.classifier(pooled_output)
        return output