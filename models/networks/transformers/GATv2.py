import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from . import LearnedPositionalEncodingv2 as LearnedPositionalEncoding
from . import GATv2Classifier
from . import ResidualBlockv2
from . import GlobalAttentionPooling

class HierarchicalGATTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, edge_attr_dim=1):
        super().__init__()
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model)
        )

        self.pos_encoder = LearnedPositionalEncoding(d_model, dropout, max_len=1000)

        self.gat_layers = nn.ModuleList()
        self.gat_transforms = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                pyg_nn.GATv2Conv(d_model, d_model, heads=nhead, dropout=dropout, add_self_loops=False, edge_dim=edge_attr_dim)
            )
            self.gat_transforms.append(nn.Linear(nhead * d_model, d_model))

        self.pre_pooling_layer_norm = nn.LayerNorm(d_model)
        self.pooling = GlobalAttentionPooling(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            ResidualBlockv2(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):

        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = x.view(-1, 256)

        for i, (gat_layer, gat_transform) in enumerate(zip(self.gat_layers, self.gat_transforms)):
            x = gat_layer(x, edge_index, edge_attr=edge_attr)
            x = gat_transform(x)

        x = self.pre_pooling_layer_norm(x)

        x_pooled = self.pooling(x, batch)

        x = self.classifier(x_pooled)

        return x