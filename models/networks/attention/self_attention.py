import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention module that computes attention weights and applies them to the input tensor.

    Parameters
    ----------
        input_dim : int
            The dimension of the input tensor.
    """

    def __init__(self, input_dim):
        """Initialises the self-attention module."""
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """Forward pass of the self-attention module.

        The input tensor is multiplied by the query, key, and value matrices
        to compute the attention scores.

        Parameters
        ----------
            x : torch.Tensor
                The input tensor.

        Returns:
        ----------
            torch.Tensor
                The output tensor after applying self-attention.
        """
        Q = self.query(x)  # les requêtes
        K = self.key(x)    # les clés
        V = self.value(x)  # les valeurs

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output
