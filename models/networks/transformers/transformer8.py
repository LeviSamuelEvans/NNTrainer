import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from . import LearnedPositionalEncodingv2 as LearnedPositionalEncoding
from . import GATv2Classifier
from . import ResidualBlockv2
from . import GlobalAttentionPooling


class HierarchicalGATTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model)
        )
        self.pos_encoder = LearnedPositionalEncoding(d_model, dropout, max_len=1000)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            #activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.local_gat = GATv2Classifier(d_model, 128, d_model, dropout)
        self.combine_with_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
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
        x_input = x
        x = self.input_embedding(x)
        x_input_embedded = self.input_embedding(x_input) # Embed the input for the local GAT so dimensions match when combined with the transformer output
        #x = self.pos_encoder(x) # REMOVE FOR THIS RUN, the positional information could actually be detrimental...:/
        x = self.transformer_encoder(x)
        # NOTE: the graph attention network is applied to the output of the transformer encoder, and the architecure is sparse in nature via max_distance parameter
        x_local_gat = self.local_gat(
            x_input_embedded, edge_index, edge_attr
        )  # (NODES: 4-vectors and other representations of [J,L,Et], EDGES: [dR, (dPhi,dEta)]  -> TODO: GLOBALS)

        # let's combine the global and local features using attention!
        x_combined, _ = self.combine_with_attention(
            x.unsqueeze(1), x_local_gat.unsqueeze(1), x_local_gat.unsqueeze(1)
        )
        x_combined = x_combined.squeeze(1)

        # Let's apply layer norm before pooling
        x_combined = self.pre_pooling_layer_norm(x_combined)

        # Now, let's apply global pooling via the GlobalAttentionPooling module
        x_pooled = self.pooling(x_combined, batch)

        x = self.classifier(x_pooled)
        return x
