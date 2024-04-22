import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from . import LearnedPositionalEncodingv2 as LearnedPositionalEncoding
from . import GATv2Classifier
from . import ResidualBlock
from . import ResidualBlockv2
from . import MultiHeadAdditiveAttention
from . import GlobalAttentionPooling

class HierarchicalGATTransformer(nn.Module):
    """Transformer-based classifier model with enhanced cross-attention and hierarchical design."""

    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(HierarchicalGATTransformer, self).__init__()

        # embedding and positional encoding via linear layer and learned positional encoding from ../embedding/Learned
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

        self.residual_transformer = nn.Linear(d_model, d_model)

        # GAT for local graph structure with deltaR distances
        self.local_gat = GATv2Classifier(d_model, 128, d_model, dropout)

        # Cross-attention for integrating the local graph structure
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.cross_attention = MultiHeadAdditiveAttention(d_model, nhead, dropout=dropout)

        # Attention pooling
        self.attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.attention_pooling = MultiHeadAdditiveAttention(d_model, nhead, dropout=dropout)

        self.attention_pool = GlobalAttentionPooling(d_model)

        #edge pooling
        # self.edge_pool = pyg_nn.EdgePooling(d_model, dropout=dropout)
        
        # La Fusionnnnn layers
        self.fusion_layer1 = nn.Linear(2*d_model, d_model)
        self.fusion_norm1 = nn.LayerNorm(d_model)
        self.fusion_layer2 = nn.Linear(d_model, d_model)
        self.fusion_norm2 = nn.LayerNorm(d_model)

        # Classifier
        # TODO: Add layer norms and skip connections between the local an global features
        # TODO: Add residual connections in the classifier via ../layers/residual
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlockv2(256),
            nn.Linear(256, 128),
            nn.LayerNorm(128), # test Layer norms here
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlockv2(128),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, batch=None):

        x = self.input_embedding(x)
        x = self.pos_encoder(x)

        x_transformer = self.transformer_encoder(x)
        x_transformer = x + self.residual_transformer(x_transformer)

        # process the local grpah structure using GAT
        x_local_gat = self.local_gat(x, edge_index)

        # Apply EdgePooling here
        # x_pooled, edge_index, _, batch, _ = self.edge_pool(x_local_gat, edge_index, batch=batch)

        # now, integrate the local and global features using cross-attention
        x_combined = torch.cat([x_transformer.unsqueeze(0), x_local_gat.unsqueeze(0)], dim=0)
        x_combined, _ = self.cross_attention(x_combined, x_combined, x_combined)
        x_combined = x_combined.mean(dim=0)

        # use fusion layers to combine the features together
        x_fused = torch.cat([x_transformer, x_combined], dim=-1)
        x_fused = self.fusion_layer1(x_fused)
        x_fused = self.fusion_norm1(x_fused)
        x_fused = nn.functional.gelu(x_fused)
        x_fused = self.fusion_layer2(x_fused)
        x_fused = self.fusion_norm2(x_fused)

        # Attention pooling layer of the fused features
        pooled_output, _ = self.attention_pooling(x_fused, x_fused, x_fused)

        # Global mean pooling if batch data is provided
        if batch is not None:
            # Ensure the batch tensor has the correct shape
            batch = batch.view(-1)
            if batch.max() >= pooled_output.size(0):
                print("Error: max index in batch exceeds the number of rows in pooled_output.")
                print("Max index in batch:", batch.max())
                print("Number of rows in pooled_output:", pooled_output.size(0))
                raise ValueError("Batch indices exceed the number of rows in pooled_output.")

            # apply global mean pooling on the pooled output using the batch indices
            #pooled_output = pyg_nn.global_mean_pool(pooled_output, batch)
            pooled_output = self.attention_pool(pooled_output, batch)
            #pooled_output = pyg_nn.global_max_pool(pooled_output, batch)

        # re-shape the pooled output
        pooled_output = pooled_output.view(-1, self.classifier[0].in_features)

        # now, pass the pooled outputs through the sequential classifier
        output = self.classifier(pooled_output)

        return output
