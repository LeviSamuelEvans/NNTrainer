import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GlobalAttentionPooling(nn.Module):
    def __init__(self, in_features):
        super(GlobalAttentionPooling, self).__init__()
        self.attention = pyg_nn.GlobalAttention(gate_nn=nn.Linear(in_features, 1))

    def forward(self, x, batch):
        return self.attention(x, batch)