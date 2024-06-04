import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


global_features_dim = 1
hidden_dim = 64
output_dim = 1


class ResidualGNN1(nn.Module):
    """A graph neural network with residual connections.

    The convolutional layers are implemented using the Graph
    Convolutional Network (GCN) layer.

    Parameters:
    -----------
        input_dim : int
            The dimensionality of the input features.
        dropout_prob : float
            The probability of an element to be zeroed in dropout layers.

    """

    def __init__(self, input_dim, dropout_prob=0.5):
        """Initialises the graph neural network."""
        super(ResidualGNN1, self).__init__()

        self.conv1 = GCNConv(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.conv2 = GCNConv(512, 512)  # Same dimension for residual connection
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.conv3 = GCNConv(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.conv4 = GCNConv(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_prob)

        self.conv5 = GCNConv(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(dropout_prob)

        self.conv6 = GCNConv(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.dropout6 = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self.global_fc1 = nn.Linear(global_features_dim, hidden_dim)
        self.global_fc2 = nn.Linear(hidden_dim, output_dim)

        # Update the input dimension of the final linear layer
        combined_dim = (
            2 + output_dim
        )  # 32 is the output dimension of the last GCN layer, adjust if needed
        self.fc = nn.Linear(combined_dim, 1)

    def forward(self, x, edge_index, global_features):
        """Performs a forward pass through the neural network."""
        max_edge_index = edge_index.max().item()
        num_nodes = x.size(0)
        if max_edge_index >= num_nodes:
            print(
                f"Adjusted max edge index from {max_edge_index} to {num_nodes-1} for debugging."
            )
            edge_index = torch.clamp(edge_index, max=num_nodes - 1)
            # print(f"Out-of-bounds edge indices detected: Max edge index {max_edge_index}, Number of nodes {num_nodes}")
            # raise ValueError("Out-of-bounds edge index detected. Check your graphs!")

        print("Input x shape:", x.shape)
        print(
            f"Max index in edge_index: {edge_index.max().item()}, Size of x: {x.size(0)}"
        )
        identity = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        print("After conv1 shape:", identity.shape)  # DEBUG
        out = self.dropout1(identity)

        out = F.leaky_relu(
            self.bn2(self.conv2(out, edge_index)) + identity
        )  # Residual connection
        out = self.dropout2(out)

        out = F.leaky_relu(self.bn3(self.conv3(out, edge_index)))
        out = self.dropout3(out)

        out = F.leaky_relu(self.bn4(self.conv4(out, edge_index)))
        out = self.dropout4(out)

        out = F.leaky_relu(self.bn5(self.conv5(out, edge_index)))
        out = self.dropout5(out)

        out = F.leaky_relu(self.bn6(self.conv6(out, edge_index)))
        out = self.dropout6(out)

        # Incorporate global features
        global_out = F.leaky_relu(self.global_fc1(global_features))
        global_out = self.global_fc2(global_out)
        print(out.shape, global_out.shape)  # DEBUG

        # Combine node and global outputs
        combined_out = torch.cat((out, global_out), dim=1)

        final_out = self.sigmoid(self.fc(combined_out))
        return final_out
