import torch.nn as nn
import torch.nn.functional as F


class ResidualComplexNN(nn.Module):
    """
    A deep neural network with residual connections.
    Employs the use of skip connections to improve training performance.

    Args:
        input_dim (int): The dimension of the input tensor.
        dropout_prob (float): The probability of dropout.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        bn1 (nn.BatchNorm1d): The first batch normalization layer.
        dropout1 (nn.Dropout): The first dropout layer.
        fc2 (nn.Linear): The second fully connected layer.
        bn2 (nn.BatchNorm1d): The second batch normalization layer.
        dropout2 (nn.Dropout): The second dropout layer.
        fc3 (nn.Linear): The third fully connected layer.
        bn3 (nn.BatchNorm1d): The third batch normalization layer.
        dropout3 (nn.Dropout): The third dropout layer.
        fc4 (nn.Linear): The fourth fully connected layer.
        bn4 (nn.BatchNorm1d): The fourth batch normalization layer.
        dropout4 (nn.Dropout): The fourth dropout layer.
        fc5 (nn.Linear): The fifth fully connected layer.
        bn5 (nn.BatchNorm1d): The fifth batch normalization layer.
        dropout5 (nn.Dropout): The fifth dropout layer.
        fc6 (nn.Linear): The sixth fully connected layer.
        bn6 (nn.BatchNorm1d): The sixth batch normalization layer.
        dropout6 (nn.Dropout): The sixth dropout layer.
        fc7 (nn.Linear): The seventh fully connected layer.
        sigmoid (nn.Sigmoid): The sigmoid activation function.

    Methods:
        forward(x): Performs a forward pass through the neural network.
        initialize_weights(model): Initializes the weights of the given model
        using He initialization for linear layers and sets bias to 0.
    """

    def __init__(self, input_dim, dropout_prob=0.5):
        super(ResidualComplexNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(512, 512)  # Same dimension for residual connection
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_prob)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(dropout_prob)

        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.dropout6 = nn.Dropout(dropout_prob)

        self.fc7 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

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

        out = self.sigmoid(self.fc7(out))
        # out = self.fc7(out) # if using BCEWithLogitsLoss, don't use sigmoid here
        return out
