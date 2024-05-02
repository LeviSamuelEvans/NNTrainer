import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from models.networks.layers.residual_v2 import ResidualBlockv2


class GATv2Classifier(nn.Module):
    """
    Graph Attention Network v2 (GATv2) Classifier for use in transformer models.

    The GATv2Classifier consists of two GATv2 layers followed by a
    multi-layer perceptron (MLP) with GELU activations.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Dimensionality of the hidden layer.
    output_dim : int
        Dimensionality of the output layer.
    dropout : float
        Dropout rate for regularization.
    num_heads : int
        Number of attention heads in the GATv2 layers.

    Attributes
    ----------
    gatv2_1 (GATv2Conv):
        First GATv2 layer.
    gatv2_2 (GATv2Conv):
        Second GATv2 layer.
    classifier (nn.Sequential):
        Sequential neural network for classification.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_heads=8):
        super(GATv2Classifier, self).__init__()

        edge_attr_dim = 1

        self.gatv2_1 = GATv2Conv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=False,
            concat=True,
            edge_dim=edge_attr_dim,
        )
        self.gatv2_2 = GATv2Conv(
            hidden_dim * num_heads,
            output_dim,
            heads=2,
            dropout=dropout,
            add_self_loops=False,
            concat=False,
            edge_dim=edge_attr_dim,
        )

        self.classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlockv2(output_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.gatv2_1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.gatv2_2(x, edge_index, edge_attr=edge_attr))
        x = self.classifier(x)
        return x
