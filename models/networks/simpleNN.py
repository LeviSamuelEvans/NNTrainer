import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    """
    A simple neural network with 4 fully connected layers and sigmoid activation function.

    Args:
    input_dim (int): The number of input features.

    Returns:
    torch.Tensor: The output tensor of the neural network.
    """
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x
