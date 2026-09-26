import torch.nn as nn
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import GINEConv, global_mean_pool


class GINEOnly(nn.Module):
    """
    Internal GINE baseline.

    It uses the same atom/bond encoders, hidden dimension, number of
    GINE layers, normalization, activation, and graph pooling as the
    sequence comparison models, while removing the sequence encoder
    and cross-attention branch.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
    ):
        super().__init__()

        self.atom_encoder = AtomEncoder(
            emb_dim=hidden_dim
        )

        self.bond_encoder = BondEncoder(
            emb_dim=hidden_dim
        )

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.convs.append(
                GINEConv(nn=mlp)
            )

            self.batch_norms.append(
                nn.BatchNorm1d(hidden_dim)
            )

        # Keep the same two-layer prediction-head depth.
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(
            data.edge_attr
        )

        for conv, batch_norm in zip(
            self.convs,
            self.batch_norms,
        ):
            x = conv(
                x,
                data.edge_index,
                edge_attr,
            )

            x = batch_norm(x)
            x = F.relu(x)

        graph_repr = global_mean_pool(
            x,
            data.batch,
        )

        return self.pred_head(graph_repr)
