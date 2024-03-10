import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.attention.multiheaded_attention import MultiHeadSelfAttention


def iqr_pooling(tensor, dim):
    """
    Playing around with alt. aggregation functions :D
    """
    q75 = torch.quantile(tensor, 0.75, dim=dim, keepdim=False)
    q25 = torch.quantile(tensor, 0.25, dim=dim, keepdim=False)
    iqr = q75 - q25
    return iqr


class ResidualComplexNNwith_MH_attention(nn.Module):
    """
    Residual Complex Neural Network with Multi-Head Attention.

    Args:
        input_dim (int): The dimensionality of the input.
        dropout_prob (float, optional): The dropout probability. Defaults to 0.5.
        num_heads (int, optional): The number of attention heads. Defaults to 8.
    """

    def __init__(self, input_dim, dropout_prob=0.5, num_heads=8):
        super(ResidualComplexNNwith_MH_attention, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.attention = MultiHeadSelfAttention(input_dim=512, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(512)

        # Feed-Forward Network within Attention Block
        self.ffn = nn.Sequential(
            nn.Linear(512, 2048),  # expand dimensions
            nn.ReLU(),
            nn.Linear(2048, 512),  # contracts dimenions back
        )
        self.ln2 = nn.LayerNorm(512)

        self.fc3 = nn.Linear(512 * 3, 256)
        self.ln3 = nn.LayerNorm(256)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)
        self.dropout4 = nn.Dropout(dropout_prob)

        self.fc5 = nn.Linear(128, 64)
        self.ln5 = nn.LayerNorm(64)
        self.dropout5 = nn.Dropout(dropout_prob)

        self.fc6 = nn.Linear(64, 32)
        self.ln6 = nn.LayerNorm(32)
        self.dropout6 = nn.Dropout(dropout_prob)

        self.fc7 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = F.leaky_relu(self.bn1(self.fc1(x)))
        out = self.dropout1(identity)

        out = F.leaky_relu(self.bn2(self.fc2(out)) + identity)  # a residual connection
        out = self.dropout2(out)
        # print(out.shape)

        # Apply self-attention
        attention_out = self.attention(out)
        out = self.ln1(
            attention_out + out
        )  # Apply residual connection and layer norm together
        # print(out.shape)
        # Feed-forward network
        out = self.ffn(out)
        out = self.ln2(out)
        # print(out.shape)

        # Global average pooling over the sequence dimension using mean (could use max, or others)
        # out = out.mean(dim=1)

        # try to apply multiple pooling strategies together
        mean_pooled = torch.mean(out, dim=1)
        max_pooled, _ = torch.max(out, dim=1)
        iqr_pooled = iqr_pooling(out, dim=1)

        # Combine pooled features
        combined_pooled = torch.cat([mean_pooled, max_pooled, iqr_pooled], dim=1)

        out = F.leaky_relu(self.ln3(self.fc3(combined_pooled)))
        out = self.dropout3(out)

        out = F.leaky_relu(self.ln4(self.fc4(out)))
        out = self.dropout4(out)

        out = F.leaky_relu(self.ln5(self.fc5(out)))
        out = self.dropout5(out)

        out = F.leaky_relu(self.ln6(self.fc6(out)))
        out = self.dropout6(out)

        out = self.sigmoid(self.fc7(out))
        return out


"""
Notes:

- concatenating layers here will increases the model's capacity by allowing it to process the original and attended features separately in the subsequent layers.
  more computationally expensive though
- some layer norms after each significant operation (post-attention, post-FFN, and after each residual connection) to stabilise the training
- feed-forward network (FFN) within the attention block to allow for more complex interactions between the original and attended features
- global average pooling over the sequence dimension using mean (and playing around with other pooling strategies)

"""
