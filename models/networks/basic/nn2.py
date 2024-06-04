import torch.nn as nn
import torch.nn.functional as F


class ModifiedNN(nn.Module):
    """A neural network with 4 fully connected layers and dropout.
    This serves as an example of an upgrade from the simpleNN.

    Parameters
    ----------
        input_dim (int): The dimension of the input tensor.
        dropout_prob (float, optional): The probability of dropout. Default is 0.5.

    Methods:
    ----------
        forward : x -> torch.Tensor
            Defines the forward pass of the neural network.

    """

    def __init__(self, input_dim, dropout=0.5):
        "Initialises the neural network."
        super(ModifiedNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(32, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Defines the forward pass of the neural network.

        Parameters
        ----------
            x : torch.Tensor
                The input tensor.

        Returns
        -------
            torch.Tensor
                The output tensor.

        """
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.leaky_relu(self.fc3(x))
        x = self.dropout3(x)

        x = self.fc4(x)
        return x
