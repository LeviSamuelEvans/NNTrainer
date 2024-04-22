import torch.nn as nn

# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition pp. 5934-5938
# arXiv:1810.04805.

class ResidualBlock(nn.Module):
    """
    A residual block module that performs residual connection in a neural network.

    Args:
        channels (int): The number of input and output channels.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the residual block.
        """
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x += residual
        x = self.relu(x)
        return x

class ResidualBlockGCN(nn.Module):
    """
    Residual Block for Graph Convolutional Networks (GCN).

    Args:
        channels (int): Number of input and output channels.
    """

    def __init__(self, channels):
        super(ResidualBlockGCN, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        """
        Forward pass of the Residual Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out