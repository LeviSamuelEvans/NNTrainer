import torch.nn as nn
import torch.nn.functional as F

class ModifiedNN(nn.Module):
    """
    A modified neural network with 4 fully connected layers and dropout.
    This serves as an example of an upgrade from the simpleNN

    Args:
        input_dim (int): The dimension of the input tensor.
        dropout_prob (float, optional): The probability of dropout. Default is 0.5.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        dropout1 (nn.Dropout): The first dropout layer.
        fc2 (nn.Linear): The second fully connected layer.
        dropout2 (nn.Dropout): The second dropout layer.
        fc3 (nn.Linear): The third fully connected layer.
        dropout3 (nn.Dropout): The third dropout layer.
        fc4 (nn.Linear): The fourth fully connected layer.
        sigmoid (nn.Sigmoid): The sigmoid activation function.

    Methods:
        forward(x): Defines the forward pass of the neural network.

    """
    def __init__(self, input_dim, dropout_prob=0.5):
        super(ModifiedNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.leaky_relu(self.fc3(x))
        x = self.dropout3(x)

        x = self.sigmoid(self.fc4(x))
        return x