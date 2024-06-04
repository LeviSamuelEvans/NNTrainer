import torch.nn as nn
import torch.nn.functional as F


class ResidualComplexNN(nn.Module):
    """A deep neural network with residual connections.

    Employs the use of skip connections to improve training performance.
    Using He initialization for linear layers and sets bias to 0.

    Parameters
    ----------
        input_dim : int
            The dimension of the input tensor.
        dropout : float
            The probability of dropout.

    """

    def __init__(self, input_dim, dropout=0.5):
        super(ResidualComplexNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, 512)  # Same dimension for residual connection
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(dropout)

        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.dropout6 = nn.Dropout(dropout)

        self.fc7 = nn.Linear(32, 1)
        # self.sigmoid = nn.Sigmoid()

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
        identity = F.leaky_relu(self.bn1(self.fc1(x)))
        out = self.dropout1(identity)

        out = F.leaky_relu(self.bn2(self.fc2(out)) + identity)  # Residual connection
        out = self.dropout2(out)

        out = F.leaky_relu(self.bn3(self.fc3(out)))
        out = self.dropout3(out)

        out = F.leaky_relu(self.bn4(self.fc4(out)))
        out = self.dropout4(out)

        out = F.leaky_relu(self.bn5(self.fc5(out)))
        out = self.dropout5(out)

        out = F.leaky_relu(self.bn6(self.fc6(out)))
        out = self.dropout6(out)

        # out = self.sigmoid(self.fc7(out))
        out = self.fc7(out)  # if using BCEWithLogitsLoss, don't use sigmoid here
        return out
