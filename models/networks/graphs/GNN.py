import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleGCN(nn.Module):
    """A simple graph convolutional network with 2 graph convolutional layers.

    Parameters
    ----------
        input_dim : int
            The dimension of the input tensor.
        hidden_dim : int
            The number of hidden units.
        output_dim : int
            The dimension of the output tensor.
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        """Initialises the simple graph convolutional network."""
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        """The forward pass of the convolutional network."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# Instantiate the model
model = SimpleGCN(input_dim=4)
