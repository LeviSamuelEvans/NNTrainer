import torch.nn as nn


class ResidualBlockv2(nn.Module):
    """
    A residual block module that performs residual connection in a neural network.

    Args:
        dim (int): The number of input and output channels (dimensions) of the residual block.
    """

    def __init__(self, dim):
        super(ResidualBlockv2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the residual block.
        """
        return x + self.layers(x)
