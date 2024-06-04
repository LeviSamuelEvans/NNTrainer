import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from . import LearnedPositionalEncodingv2 as LearnedPositionalEncoding
from . import GCNClassifier


class GCNtransformer(nn.Module):
    """Transformer-based classifier model with cross-attention using edge indices."""

    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(GCNtransformer, self).__init__()

        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model)
        )

        self.pos_encoder = LearnedPositionalEncoding(d_model, dropout, max_len=1000)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        # after concatenating edge features, apply linear transformation to match d_model dimensions!
        self.edge_feature_transform = nn.Linear(2 * d_model, d_model)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.classifier = GCNClassifier(d_model, 128, 64, dropout, num_classes=d_model)

        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, x, edge_index, batch=None):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # use edge_index to gather node features for cross-attention
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_features = self.edge_feature_transform(edge_features)

        cross_attention_output, _ = self.cross_attention(
            x, edge_features, edge_features
        )
        x = x + cross_attention_output

        # pass transformer output to GCNClassifier
        x = self.classifier(x, edge_index)

        pooled_output, _ = self.attention_pooling(x, x, x)
        pooled_output = pooled_output.mean(
            dim=1
        )  # Aggregate over the sequence dimension

        if batch is not None:
            pooled_output = pyg_nn.global_mean_pool(pooled_output, batch)

        return pooled_output
