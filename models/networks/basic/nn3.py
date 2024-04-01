import torch.nn as nn
import torch.nn.functional as F


class ComplexNN(nn.Module):
    """A neural network with 6 fully connected layers and batch normalisation.

    Parameters
    ----------
        input_dim : int
            The dimension of the input tensor.
        dropout : float
            The probability of an element to be zeroed in dropout layer.

    """

    def __init__(self, input_dim, dropout=0.5):
        super(ComplexNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout)

        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout5 = nn.Dropout(dropout)

        self.fc6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Performs a forward pass through the neural network.

        Parameters
        ----------
            x : torch.Tensor
                The input tensor

        Returns
        -------
            torch.Tensor
                The output tensor.

        """
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = F.leaky_relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)

        x = self.sigmoid(self.fc6(x))
        return x
