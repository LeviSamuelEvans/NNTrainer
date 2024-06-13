import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

from . import LearnedPositionalEncodingv2 as LearnedPositionalEncoding
from . import ResidualBlockv2
import logging


class Pooling(nn.Module):
    """Pooling layer utilising attention scores to pool the graph node embeddings.

    - if the input tensor is 3D (seq_len, batch_size, embedding_dim),
      we shape it to 2D (seq_len * batch_size, embedding_dim)
    - if the input tensor is 2D (batch_size, embedding_dim), we repeat the batch tensor
      to match the number of nodes in the input tensor
    - The attention scores are calculated using a linear layer followed by a softmax activation
    - The attention scores are used to weight the embeddings by element-wise multiplication
    - The weighted embeddings are summed for each batch
    - The index_add_ method adds the weighted node embeddings to the out tensor based on
      indices provided by the batch tensor, effectively summing the embeddings for each batch
      according to the attention scores.

    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.attention = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x, batch=None):
        # (seq_len, batch_size, in_channels)
        if x.dim() == 3:
            seq_len, batch_size, _ = x.size()
            x = x.view(seq_len * batch_size, self.in_channels)
            batch = batch.repeat(seq_len)

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.attention(x)
        x = score * x
        x = (
            torch.zeros(batch.max().item() + 1, x.size(1))
            .to(x.device)
            .scatter_add_(0, batch.unsqueeze(-1).repeat(1, x.size(1)), x)
        )

        # alternative implementation using broadcasting, more mem efficient but not yet working...
        # -> use broadcasting to sum the embeddings for each batch instead of scatter_add_ and repeat
        # out = torch.zeros(batch.max().item() + 1, x.size(1)).to(x.device)
        # out.index_add_(0, batch, x)

        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )

    def forward(self, query, key, value, dynamic_scores=None):
        # [sequence_length, batch_size, embedding_dim]
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key.dim() == 2:
            key = key.unsqueeze(1)
        if value.dim() == 2:
            value = value.unsqueeze(1)

        if dynamic_scores is not None:
            # shape dyn scores for MH attention
            num_heads, seq_len, _ = dynamic_scores.size()

            # need to match the shape of the attention mask to the query shape
            # (seq_len, num_heads, seq_len)
            dynamic_scores = dynamic_scores.permute(1, 0, 2)
            # (num_heads, seq_len, seq_len)
            dynamic_scores = dynamic_scores.reshape(num_heads, seq_len, seq_len)

            attn_output, _ = self.cross_attn(
                query, key, value, attn_mask=dynamic_scores
            )
        else:
            attn_output, _ = self.cross_attn(query, key, value)

        return attn_output


class dynGATtransformer(nn.Module):
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
                # Improved version of the Graph Attention Network (GAT) layer with edge features
                # the improvement from GATv2 lies in the dynamic attention scores (more adaptive coefficients)
                pyg_nn.GATv2Conv(
                    d_model,
                    d_model,
                    heads=nhead,
                    dropout=dropout,
                    add_self_loops=False,  # -> :add self loops to the adjacency matrix set to false to avoid duplicate edges
                    edge_dim=edge_attr_dim,
                    concat=False,
                )
            )
            self.transformer_layers.append(
                # TransformerConv is a generalization of GATConv, replacing the GAT's attention mechanism
                # with a transformer encoder-style attention. We apply it here to the output of the GAT layer
                # to capture global dependencies in the graph without needing fixed global graph features.
                pyg_nn.TransformerConv(
                    d_model,
                    d_model,
                    heads=nhead,
                    dropout=dropout,
                    edge_dim=edge_attr_dim,
                    concat=False,  # -> :concat = false means we use the average of the attention heads
                    beta=True,  # -> :use beta parameter for edge updates
                )
            )
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.layer_norms.append(nn.LayerNorm(d_model))

        self.use_cross_attention = use_cross_attention
        self.pre_pooling_layer_norm = nn.LayerNorm(d_model)
        self.pooling = Pooling(d_model)
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
                dynamic_scores = self.dynamic_attention_scores(edge_attr)

                # sequence length and number of heads
                seq_len = x.size(0)
                num_edges = dynamic_scores.size(0)
                num_heads = self.cross_attn_layers[0].cross_attn.num_heads

                # reshpaing the dynamic scores to match the shape of the attention mask
                dynamic_scores = dynamic_scores.unsqueeze(1).repeat(1, seq_len)
                dynamic_scores = dynamic_scores.unsqueeze(1).repeat(1, num_heads, 1)
                dynamic_scores = dynamic_scores.view(num_edges, num_heads, seq_len)
                dynamic_scores = dynamic_scores.permute(1, 0, 2)

                dynamic_scores = dynamic_scores[:, :seq_len, :seq_len]

                x = cross_attn_layer(x_transformer, x_gat, x, dynamic_scores) + residual

            # need the batch tensor has the same number of nodes as the input tensor
            if batch is not None and batch.size(0) != x.size(0):
                num_nodes = x.size(0)
                batch = batch.repeat(num_nodes // batch.size(0))

            x_pooled = self.pooling(x, batch)
            x = self.classifier(x_pooled)
            return x

    def dynamic_attention_scores(self, edge_attr):

        # invariant mass is the 5th column in the edge_attr tensor
        invariant_mass = edge_attr[:, 5]

        # Higgs boson mass (add reference)
        higgs_mass = 125.09
        # ~ rough resolution of detector based on jet mass?
        sigma = 15.0

        # Gaussian-based attention scores
        dynamic_scores = torch.exp(-0.5 * ((invariant_mass - higgs_mass) / sigma) ** 2)
        # now normalise the scores to be between 0 and 1
        dynamic_scores = (dynamic_scores - dynamic_scores.min()) / (
            dynamic_scores.max() - dynamic_scores.min()
        )

        # (inverted scores for attention mask compatibility)
        dynamic_scores = 1 - dynamic_scores

        # print("Dynamic scores after all processing")
        # print(dynamic_scores[:10])
        return dynamic_scores
