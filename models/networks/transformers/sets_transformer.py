from . import LorentzAtt
import torch.nn as nn

"""
Investigating:
- allowing the attention to derive the relationships between the sets, without
ordering the inputs.
- apply attention to pooling mechanism

"""

LorentzInvariantAttention = LorentzAtt

class SetsTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
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
        self.attention = LorentzInvariantAttention(d_model, nhead, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, x_coords):
        x = x + self.attention(self.norm1(x), x_coords)
        x = x + self.feed_forward(self.norm2(x))
        return x

class SetsTransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(SetsTransformerClassifier, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.set_transformer = SetsTransformerEncoder(d_model, nhead, num_layers, dropout)
        self.attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # NEW
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, x_coords):
        x = self.input_embedding(x)
        x = self.set_transformer(x, x_coords)

        # attention pooling
        pooled_output, _ = self.attention_pooling(x, x, x)
        pooled_output = pooled_output.mean(dim=1)

        output = self.classifier(pooled_output)
        return output