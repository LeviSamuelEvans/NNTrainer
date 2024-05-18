import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GlobalAttentionPooling(nn.Module):
    """
    References
    ---------
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.AttentionalAggregation.html?highlight=AttentionalAggregation
    """
    def __init__(self, in_features):
        super(GlobalAttentionPooling, self).__init__()
        self.attention = pyg_nn.AttentionalAggregation(gate_nn=nn.Linear(in_features, 1))

    def forward(self, x, batch):
        return self.attention(x, batch)