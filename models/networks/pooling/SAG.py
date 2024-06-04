import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

# from torch_geometric.utils import topk, filter_adj


class SAGPooling(nn.Module):
    """
    Self-Attention Graph Pooling (SAGPool)

    References
    ----------
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.SAGPooling.html
    """

    def __init__(self, in_channels, ratio=0.5):
        super(SAGPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = nn.Linear(in_channels, 1)
        self.non_linearity = nn.ReLU()

    def topk(self, x, ratio, batch):
        num_nodes = x.size(0)
        num_keep = max(1, int(ratio * num_nodes))
        score = x
        _, perm = score.sort(dim=0, descending=True)
        perm = perm[:num_keep]
        return perm

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score = self.score_layer(x).squeeze()

        perm = self.topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = pyg_nn.pool.filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0)
        )

        return x, edge_index, edge_attr, batch
