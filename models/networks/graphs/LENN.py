import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
#from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
import numpy as np

hidden = 14
outputs = 2

class LorentzEdgeBlock(torch.nn.Module):
    """Edge block for the Lorentz Interaction Network."""
    def __init__(self):
        super(LorentzEdgeBlock, self).__init__()
        self.edge_mlp = Seq(Lin(4, hidden), ReLU(), Lin(hidden, hidden))
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
        return torch.sum(
            torch.mul(torch.matmul(x1, self.minkowski), x2), 1, keepdim=True
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat(
            [
                self.innerprod(src, src),
                self.innerprod(src, dest),
                self.psi(self.innerprod(dest, dest)),
                self.psi(self.innerprod(src - dest, src - dest)),
            ],
            dim=1,
        )
        return self.edge_mlp(out)


class LorentzNodeBlock(torch.nn.Module):
    """Node block for the Lorentz Interaction Network."""
    def __init__(self):
        super(LorentzNodeBlock, self).__init__()
        self.node_mlp_1 = Seq(Lin(1 + hidden, hidden), ReLU(), Lin(hidden, hidden))
        self.node_mlp_2 = Seq(Lin(1 + hidden, hidden), ReLU(), Lin(hidden, hidden))
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

    def innerprod(self, x1, x2):
        return torch.sum(
            torch.mul(torch.matmul(x1, self.minkowski), x2), 1, keepdim=True
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([self.innerprod(x[row], x[row]), edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([self.innerprod(x, x), out], dim=1)
        return self.node_mlp_2(out)


class GlobalBlock(torch.nn.Module):
    """Global block for the Lorentz Interaction Network."""
    def __init__(self):
        super(GlobalBlock, self).__init__()
        self.global_mlp = Seq(Lin(hidden, hidden), ReLU(), Lin(hidden, outputs))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter_mean(x, batch, dim=0)
        return self.global_mlp(out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LorentzInteractionNetwork(torch.nn.Module):
    """Lorentz Interaction Network model.

    Builds the Lorentz Interaction Network model, using
    the EdgeBlock, NodeBlock, and GlobalBlocks.

    Parameters
    ----------
        hidden : int
            The number of hidden units.
        outputs : int
            The number of output units.

    Methods
    -------
        forward(x, edge_index, batch)
            The forward pass of the Lorentz Interaction Network.

    """
    def __init__(self):
        super(LorentzInteractionNetwork, self).__init__()
        self.lorentzinteractionnetwork = MetaLayer(
            LorentzEdgeBlock(), LorentzNodeBlock(), GlobalBlock()
        )

    def forward(self, x, edge_index, batch):
        print(f"x shape: {x.shape}")  # DEBUG
        print(f"edge_index shape: {edge_index.shape}")  # DEBUG
        print(f"batch shape: {batch.shape}")  # DEBUG
        print(f"max edge_index: {edge_index.max()}")  # DEBUG

        x, edge_attr, u = self.lorentzinteractionnetwork(
            x, edge_index, None, None, batch
        )
        return u

    # ensure all modules are moved to the device :/
    def to(self, device):
        super(LorentzInteractionNetwork, self).to(device)

        for module in self.modules():
            if hasattr(module, "minkowski"):
                module.minkowski = module.minkowski.to(device)
        return self
