import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from . import LearnedPositionalEncodingv2 as LearnedPositionalEncoding
from . import GATv2Classifier
from . import ResidualBlockv2
from . import GlobalAttentionPooling


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )

    def forward(self, query, key, value):
        attn_output, _ = self.cross_attn(query, key, value)
        return attn_output


class GATtransformerv3(nn.Module):
    """
    References
    ---------
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html
    """

    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dropout,
        edge_attr_dim=6,
        use_cross_attention=True,
    ):
        super().__init__()
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model)
        )
        self.pos_encoder = LearnedPositionalEncoding(d_model, dropout, max_len=1000)

        self.gat_layers = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(
                pyg_nn.GATv2Conv(
                    d_model,
                    d_model,
                    heads=nhead,
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=edge_attr_dim,
                    concat=False,
                )
            )
            self.transformer_layers.append(
                pyg_nn.TransformerConv(
                    d_model,
                    d_model,
                    heads=nhead,
                    dropout=dropout,
                    edge_dim=edge_attr_dim,
                    concat=False,
                    beta=True,
                )
            )
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.layer_norms.append(nn.LayerNorm(d_model))

        self.use_cross_attention = use_cross_attention
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

        for gat_layer, transformer_layer, cross_attn_layer, layer_norm in zip(
            self.gat_layers,
            self.transformer_layers,
            self.cross_attn_layers,
            self.layer_norms,
        ):
            residual = x
            x_gat = gat_layer(x, edge_index, edge_attr=edge_attr)
            x_gat = layer_norm(x_gat)
            x_transformer = transformer_layer(x_gat, edge_index, edge_attr=edge_attr)
            x_transformer = layer_norm(x_transformer)
            if self.use_cross_attention:
                x = cross_attn_layer(x_transformer, x_gat, x) + residual

        x = self.pre_pooling_layer_norm(x)
        x_pooled = self.pooling(x, batch)
        x = self.classifier(x_pooled)
        return x
