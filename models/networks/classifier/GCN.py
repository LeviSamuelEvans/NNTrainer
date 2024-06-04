import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from . import ResidualBlockGCN


class GCNClassifier(nn.Module):
    """
    Graph Convolutional Network (GCN) Classifier for use in
    transformer models.

    The GCNClassifier consists of two graph convolutional layers
    followed by a multi-layer perceptron (MLP) with GELU activations.
    We make use of GAT (Graph Attention Network) layers for the
    graph convolutions.

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

    Attributes
    ----------

        gcn1 (GCNConv):
            First graph convolutional layer.
        gcn2 (GCNConv):
            Second graph convolutional layer.
        classifier (nn.Sequential):
            Sequential neural network.

    References
    ----------
    https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html

    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_classes):
        super(GCNClassifier, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass of the GCNClassifier."""

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.classifier(x)

        return x  # .squeeze(-1)
