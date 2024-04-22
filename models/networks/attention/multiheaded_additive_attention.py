import torch
import torch.nn as nn

class MultiHeadAdditiveAttention(nn.Module):
    """
    Multi-Head Additive Attention module.

    Parameters
    ----------
        d_model : int
            The input and output dimension of the attention module.
        num_heads : int
            The number of attention heads.
        dropout : float, optional
            The dropout probability.

    Attributes
    ----------
        d_model : int
            The input and output dimension of the attention module.
        num_heads : int
            The number of attention heads.
        head_dim : int
            The dimension of each attention head.

        query_proj (nn.Linear): Linear projection layer for queries.
        key_proj (nn.Linear): Linear projection layer for keys.
        value_proj (nn.Linear): Linear projection layer for values.
        score_proj (nn.Linear): Linear projection layer for attention scores.
        output_proj (nn.Linear): Linear projection layer for the final output.
         dropout (nn.Dropout): Dropout layer.

    Methods
    -------
        forward [query, key, value, mask=None]:
            Performs forward pass of the attention module.

    """

    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAdditiveAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.score_proj = nn.Linear(d_model, num_heads)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Performs forward pass of the attention module.

        Returns
        -------
            output : torch.Tensor
                The output tensor of shape (batch_size, seq_len, d_model).
            attention_weights : torch.Tensor
                The attention weights tensor of shape (batch_size, num_heads, seq_len).

        """
        batch_size, seq_len, _ = query.size()

        # project the query, key, and value matrices into the multi-head space
        projected_query = self.query_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        projected_key = self.key_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        projected_value = self.value_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # compute the additive attention scores
        scores = self.score_proj(torch.tanh(projected_query.unsqueeze(-2) + projected_key.unsqueeze(-3))).squeeze(-1)

        # apply a mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # apply softmax to compute the attention weights
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights.unsqueeze(-2), projected_value).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # projection of the attended values using the output projection
        output = self.output_proj(attended_values)

        # apply dropout
        output = self.dropout(output)

        return output, attention_weights