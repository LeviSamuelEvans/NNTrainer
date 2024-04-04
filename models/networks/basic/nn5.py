import torch.nn as nn
import torch.nn.functional as F
from models.networks.attention.self_attention import SelfAttention


class ResidualComplexNNwithattention(nn.Module):
    """A deep neural network with residual connections.

    Employs the use of skip connections to improve training performance.
    Uses a self-attention mechanism in the middle of the network.

    Parameters
    ----------
        input_dim : int
            The dimension of the input tensor.
        dropout : float
            The probability of dropout.

    """

    def __init__(self, input_dim, dropout_prob=0.5):
        "Initialises the neural network."
        super(ResidualComplexNNwithattention, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(512, 512)  # Same dimension for residual connection
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.attention = SelfAttention(512)  # first attention layer

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
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Performs a forward pass through the neural network."""
        identity = F.leaky_relu(self.bn1(self.fc1(x)))
        out = self.dropout1(identity)

        out = F.leaky_relu(self.bn2(self.fc2(out)) + identity)  # a residual connection
        out = self.dropout2(out)

        # Apply self-attention
        attention_out = self.attention(out)
        out = (
            out + attention_out
        )  # could also concatenate instead of sum here! (worth a try)

        out = F.leaky_relu(self.bn3(self.fc3(out)))
        out = self.dropout3(out)

        out = F.leaky_relu(self.bn4(self.fc4(out)))
        out = self.dropout4(out)

        out = F.leaky_relu(self.bn5(self.fc5(out)))
        out = self.dropout5(out)

        out = F.leaky_relu(self.bn6(self.fc6(out)))
        out = self.dropout6(out)

        #out = self.sigmoid(
        # use softmax for multi-class classification when appropriate
        out = self.fc7(out) # if using BCEWithLogitsLoss, don't use sigmoid here
        return out
