import torch
import torch.nn as nn

class AdditiveAttention(nn.Module):
    def __init__(self, d_model):
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, query, key, value, mask=None):
        # project the query, key, and value matrices
        projected_query = self.query_proj(query)
        projected_key = self.key_proj(key)
        projected_value = self.value_proj(value)

        # compute the additive attention scores
        scores = self.score_proj(torch.tanh(projected_query.unsqueeze(-2) + projected_key.unsqueeze(-3))).squeeze(-1)

        # qpply the mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # apply softmax to compute the attention weights and compute values
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights.unsqueeze(-2), projected_value).squeeze(-2)

        return attended_values, attention_weights


"""
NOTES:
------

In the additive attention implementation, the query, key, and value matrices are first projected using separate linear layers.
Then, the additive attention scores are computed by adding the projected query and key matrices, applying the hyperbolic tangent (tanh)
activation function, and projecting the result using another linear layer.

The attention weights are obtained by applying the softmax function to the scores, and the attended values are
computed by multiplying the attention weights with the projected value matrix.

Whereas, usual the scaled dot-product attention is computed by taking the dot product between the query and key matrices,
scaling it by a factor of 1/sqrt(d_model), and then applying the softmax function to obtain the attention weights.
The attended values are then computed by multiplying the attention weights with the value matrix.
"""