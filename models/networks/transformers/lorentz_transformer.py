import torch
import torch.nn as nn
import math
import numpy as np

"""
Notes: x_coords represents the coordinate features of the objects in the higher dimensional space,
that preserves Lorentz invariance.

"""


class PositionalEncoding(nn.Module):
    """Positional encoding module for Transformer models.

    Parameters
    ----------
        d_model : int
            The dimension of the input feature.
        dropout : float, optional
            The dropout probability. Default: 0.1.

    Inputs
    ------
        x : torch.Tensor
            The input tensor of shape (batch_size, seq_len, d_model).

    Outputs
    ------
        torch.Tensor
            The output tensor of shape (batch_size, seq_len, d_model).
    """

    def __init__(self, d_model, dropout=0.1):
        """Initialises the PositionalEncoding module."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x):
        """Forward pass of the PositionalEncoding module."""
        batch_size, seq_len = x.size(0), x.size(1)
        position = (
            torch.arange(0, seq_len, dtype=torch.float, device=x.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        position_encoding = torch.zeros(
            batch_size, seq_len, self.d_model, device=x.device
        )

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device)
            * (-math.log(10000.0) / self.d_model)
        )
        position_encoding[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        position_encoding[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)

        x = x + position_encoding
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x


class LorentzInvariantAttention(nn.Module):
    """Lorentz invariant attention module for Transformer models.

    Parameters
    ----------
        d_model : int
            The dimension of the input feature.
        nhead : int
            The number of attention heads.
        dropout : float, optional
            The dropout probability. Default: 0.1.

    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super(LorentzInvariantAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.minkowski = torch.from_numpy(
            np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )

    def psi(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    def innerprod(self, x1, x2):
        # Split the input tensors into four-vectors and extra features
        x1_four_vec = x1[..., :4]
        x2_four_vec = x2[..., :4]
        x1_extra_feat = x1[..., 4:]
        x2_extra_feat = x2[..., 4:]

        # Compute the inner product using only the four-vector components
        inner_prod_four_vec = torch.sum(
            torch.mul(torch.matmul(x1_four_vec, self.minkowski), x2_four_vec),
            dim=-1,
            keepdim=True,
        )

        # Concatenate the inner product result with the extra features
        inner_prod = torch.cat(
            [inner_prod_four_vec, x1_extra_feat, x2_extra_feat], dim=-1
        )

        return inner_prod

    def forward(self, x, x_coords_list):
        """Forward pass of the LorentzInvariantAttention module.

        Methodology
        -----------

        The purpose of this method is to compute Lorentz-invariant attention weights using the provided 4-vectors and
        combine them with the input features x. Here's a step-by-step explanation:

        - It iterates over each set of 4-vectors (x_coords) in x_coords_list. For each x_coords:
            - It computes the inner product of x_coords with itself using the innerprod function. This inner product
              is Lorentz-invariant, meaning it remains the same under Lorentz transformations.
            - It applies a non-linear function psi to the inner product result. This function is used to introduce
              non-linearity and can be thought of as a feature transformation.
            - It computes the difference between x_coords and its transpose, which represents the relative positions
              between the 4-vectors.
            - It computes the inner product of the difference with itself, again using the Lorentz-invariant innerprod
              function.
            - It extracts the diagonal elements of the inner product result and applies the psi function to them.
              These diagonal elements represent the self-interaction terms.
            - It concatenates the inner product results and the transformed diagonal elements along the last dimension
              to form a tensor called x_lorentz. This tensor represents the Lorentz-invariant features computed from
              the 4-vectors.

        After processing all the 4-vectors in x_coords_list, it concatenates all the x_lorentz tensors along the last
        dimension. This step combines the Lorentz-invariant features from all the 4-vectors.

        Next, it reshapes the concatenated x_lorentz tensor to match the shape of x in the first two dimensions
        (batch size and sequence length). This is done to ensure compatibility for the subsequent element-wise addition.

        - Since the reshaped x_lorentz tensor may have a different size than x in the third dimension (feature dimension),
          it pads or truncates x_lorentz to match the size of x at dimension 2. If x_lorentz is smaller than x, it pads
          x_lorentz with zeros. If x_lorentz is larger than x, it truncates x_lorentz to match the size of x.

        - Finally, it performs an element-wise addition of the reshaped and padded/truncated x_lorentz tensor with the
          input features x. This step combines the Lorentz-invariant features with the original input features.

        The updated x tensor is then passed through a self-attention mechanism (self.self_attn) to compute the
        attention weights and generate the output of the Lorentz-invariant attention module.

        """
        # ensure the Minkowski metric tensor is on the same device as the 4-vectors
        self.minkowski = self.minkowski.to(x_coords_list[0].device)

        x_lorentz_list = []
        for x_coords in x_coords_list:
            # compute the inner product of x_coords with itself
            innerprod_result = self.innerprod(x_coords, x_coords)
            # apply the psi function to the inner product result elementwise
            psi_innerprod_result = self.psi(innerprod_result)

            # compute the difference between x_coords and its transpose
            diff_coords = x_coords[:, None, :] - x_coords[:, :, None]
            # compute the inner product of the difference with itself
            innerprod_diff_coords = self.innerprod(diff_coords, diff_coords)

            # Now, extract the diagonal elements of the inner product result
            diag_innerprod_diff_coords = torch.diagonal(
                innerprod_diff_coords, dim1=-2, dim2=-1
            )
            diag_innerprod_diff_coords = diag_innerprod_diff_coords.transpose(-1, -2)
            # apply the psi function to the diagonal elements
            psi_diag_innerprod_diff_coords = self.psi(diag_innerprod_diff_coords)

            # concatenate the inner product results and psi results along the last dimension
            x_lorentz = torch.cat(
                [
                    innerprod_result,
                    innerprod_result,
                    psi_innerprod_result,
                    psi_diag_innerprod_diff_coords.transpose(-1, -2),
                ],
                dim=-1,
            )
            x_lorentz_list.append(x_lorentz)

        # concatenate all x_lorentz tensors along the last dimension
        x_lorentz = torch.cat(x_lorentz_list, dim=-1)

        # reshape x_lorentz to match the shape of x in the first two dimensions
        x_lorentz = x_lorentz.view(x.shape[0], x.shape[1], -1)

        # pad or truncate x_lorentz to match the size of x at dimension 2
        pad_size = x.shape[2] - x_lorentz.shape[2]
        if pad_size > 0:
            # now, iff x_lorentz is smaller than x, we will pad it with zeros
            pad_tensor = torch.zeros(
                x_lorentz.shape[0],
                x_lorentz.shape[1],
                pad_size,
                device=x_lorentz.device,
            )
            x_lorentz = torch.cat([x_lorentz, pad_tensor], dim=-1)
        elif pad_size < 0:
            # but, if f x_lorentz is larger than x, we truncate it to match the size of x
            x_lorentz = x_lorentz[:, :, : x.shape[2]]

        # finally, add x_lorentz to x elementwise...
        x = x + x_lorentz

        # apply self-attention to the updated x and return the result
        return self.self_attn(x, x, x)[0]


class TransformerClassifier2(nn.Module):
    """Transformer classifier 2 module."""

    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerClassifier2, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.attention = LorentzInvariantAttention(d_model, nhead, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            #nn.Sigmoid(),
        )

    def forward(self, x, x_coords):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.attention(x, x_coords)
        x = x.mean(dim=1)  # Global average pooling over the sequence dimension
        output = self.classifier(x)
        return output
