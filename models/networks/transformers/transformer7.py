import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from . import LearnedPositionalEncodingv2 as LearnedPositionalEncoding
from . import GCNClassifier

class HierarchicalGCNTransformer(nn.Module):
    """Transformer-based classifier model with enhanced cross-attention and hierarchical design."""

    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(HierarchicalGCNTransformer, self).__init__()

       # embedding layer
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.pos_encoder = LearnedPositionalEncoding(d_model, dropout, max_len=1000)

        # transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # local graph structure processing
        self.local_gcn = pyg_nn.GCNConv(d_model, d_model)

        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.classifier = GCNClassifier(d_model, 128, 64, dropout, num_classes=d_model)

        self.attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, x, edge_index, batch=None):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x_transformer = self.transformer_encoder(x)

        # process the local graph structure
        x_local_gcn = self.local_gcn(x, edge_index)

        # Int local and global features using cross-attention
        x_combined = torch.cat([x_transformer.unsqueeze(0), x_local_gcn.unsqueeze(0)], dim=0)
        x_combined, _ = self.cross_attention(x_combined, x_combined, x_combined)
        x_combined = x_combined.mean(dim=0)  # combine the features using mean
        x_classified = self.classifier(x_combined, edge_index)

        # Pooling...
        pooled_output, _ = self.attention_pooling(x_classified, x_classified, x_classified)
        pooled_output = pooled_output.mean(dim=1)

        if batch is not None:
            pooled_output = pyg_nn.global_mean_pool(pooled_output, batch)

        return pooled_output
