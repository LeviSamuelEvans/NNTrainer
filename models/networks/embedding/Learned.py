import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding.

    Information
    -----------

        This module learns the positional encoding as part of the model training.

        The nn.Embedding layer is used to learn the positional encoding, where max_len
        is the maximum length of the input sequence and d_model is the dimensionality of
        the model. During the forward pass, the positions are generated using torch.arange
        as a sequence of integers from 0 to seq_len, and passes through the embedding layer.

        This is a variation from the original positional encoding, where the positional
        encoding is fixed and not learned during training.


    References
    ----------
        https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

    """

    def __init__(self, d_model, dropout, max_len):
        """Initialise the learned positional encoding."""
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embeddings = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(x.size(0), -1)
        position_embeddings = self.pos_embeddings(positions)
        x = x + position_embeddings
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x


class LearnedPositionalEncodingv2(nn.Module):
    """Learned positional encoding v2.

    Information
    -----------

        This module learns the positional encoding as part of the model training.

        The nn.Embedding layer is used to learn the positional encoding, where max_len
        is the maximum length of the input sequence and d_model is the dimensionality of
        the model. During the forward pass, the positions are generated using torch.arange
        as a sequence of integers from 0 to seq_len, and passes through the embedding layer.

        This is a variation from the original positional encoding, where the positional
        encoding is fixed and not learned during training.

        Additional reshaping and error checking to ensure compatibility
        between the input and positional embeddings from v1.

    References
    ----------
        https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

    """

    def __init__(self, d_model, dropout, max_len):
        """Initialise the learned positional encoding."""
        super(LearnedPositionalEncodingv2, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embeddings = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(x.size(0), -1)
        position_embeddings = self.pos_embeddings(positions)
        position_embeddings = position_embeddings.squeeze()

        # ensure the position_embeddings is correctly reshaped to match x!!
        position_embeddings = position_embeddings.view(x.shape[0], x.shape[1], -1)

        if position_embeddings.size(2) != x.size(1):
            raise ValueError(
                f"Dimension mismatch: position_embeddings have {position_embeddings.size(2)} features but x expects {x.size(1)}."
            )

        # reduce to the correct feature dimension if necessary...
        position_embeddings = position_embeddings.mean(dim=2)

        x = x + position_embeddings
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x
