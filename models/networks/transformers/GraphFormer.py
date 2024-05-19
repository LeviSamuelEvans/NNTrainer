import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import networkx as nx
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

class GraphTransformer(nn.Module):
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

        # Centrality encoding
        self.centrality_encoder = nn.Embedding(1000, d_model)

        self.transformer_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
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

    def forward(self, x, edge_index, edge_attr, batch=None, degree=None):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)

        # Add centrality encoding
        centrality = self.centrality_encoder(degree)
        x = x + centrality

        for transformer_layer, cross_attn_layer, layer_norm in zip(
            self.transformer_layers,
            self.cross_attn_layers,
            self.layer_norms,
        ):
            residual = x

            # (shortest path distances)
            spatial_encoding = self.compute_spatial_encoding(edge_index, num_nodes=x.size(0))

            # edge encoding in attention
            edge_encoding = self.compute_edge_encoding(edge_attr, edge_index)

            x_transformer = transformer_layer(x, edge_index, edge_attr=edge_attr,
                                              spatial_encoding=spatial_encoding,
                                              edge_encoding=edge_encoding)
            x_transformer = layer_norm(x_transformer)

            if self.use_cross_attention:
                x = cross_attn_layer(x_transformer, x_transformer, x) + residual

        x = self.pre_pooling_layer_norm(x)
        x_pooled = self.pooling(x, batch)
        x = self.classifier(x_pooled)

        return x

    def compute_spatial_encoding(self, edge_index, num_nodes):
        # edge_index to a networkx graph
        G = pyg_utils.to_networkx(edge_index, num_nodes=num_nodes)

        # shortest path distances for each pair of nodes
        shortest_path_dict = dict(nx.all_pairs_shortest_path_length(G))
        spatial_encoding = torch.full((num_nodes, num_nodes), float('inf'))

        for i, sp_dict in shortest_path_dict.items():
            for j, d in sp_dict.items():
                spatial_encoding[i, j] = d

        #replace inf with a large number, e.g., 1000
        spatial_encoding[spatial_encoding == float('inf')] = 1000
        return spatial_encoding.to(edge_index.device)

    def compute_edge_encoding(self, edge_attr, edge_index):
        num_edges = edge_index.size(1)
        edge_features_dim = edge_attr.size(1)

        # edge encoding matrix
        edge_encoding = torch.zeros((num_edges, edge_features_dim), device=edge_attr.device)

        # aggregate edge features along the shortest paths
        edge_index_list = edge_index.t().tolist()
        edge_dict = {(src, dst): edge_attr[i] for i, (src, dst) in enumerate(edge_index_list)}

        aggregated_edge_attr = torch.zeros_like(edge_attr)
        for (src, dst), attr in edge_dict.items():
            path_edges = nx.shortest_path(G, source=src, target=dst)
            path_edge_attrs = torch.stack([edge_dict[(u, v)] for u, v in zip(path_edges[:-1], path_edges[1:])])
            aggregated_edge_attr[i] = path_edge_attrs.mean(dim=0)

        return aggregated_edge_attr
