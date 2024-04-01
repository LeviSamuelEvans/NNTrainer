import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """A multi-headed self-attention module that computes attention weights and applies them to the input tensor.

    The input tensor is split into `num_heads` heads for Q, K, and V, and the attention scores are calculated.

    Parameters
    ----------
        input_dim : int
            The dimension of the input tensor.
        num_heads : int
            The number of attention heads.
    """
    def __init__(self, input_dim, num_heads):
        """Initialises the multi-headed self-attention module."""
        super(MultiHeadSelfAttention, self).__init__()
        assert (
            input_dim % num_heads == 0
        ), "Input dim. must be divisible by the number of attention heads."

        self.num_heads = num_heads
        self.dim_per_head = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # layer to combine concatenated heads
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """The forward pass of the multi-headed self-attention module.

        The input tensor is multiplied by the query, key, and value matrices
        """
        batch_size = x.size(0)

        # Split the embedding into `num_heads` heads for Q, K, and V
        Q = (
            self.query(x)
            .view(batch_size, -1, self.num_heads, self.dim_per_head)
            .transpose(1, 2)
        )  # (batch_size, num_heads, seq_length, dim_per_head)
        K = (
            self.key(x)
            .view(batch_size, -1, self.num_heads, self.dim_per_head)
            .transpose(1, 2)
        )  # (batch_size, num_heads, seq_length, dim_per_head)
        V = (
            self.value(x)
            .view(batch_size, -1, self.num_heads, self.dim_per_head)
            .transpose(1, 2)
        )  # (batch_size, num_heads, seq_length, dim_per_head)

        # Calculate the attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.dim_per_head**0.5
        )
        attention_weights = F.softmax(attention_scores, dim=-1)

        # apply attention weights to values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.dim_per_head)
        )

        # Pass through final linear layer
        output = self.fc_out(attention_output)

        return output
