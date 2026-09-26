import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from mamba_ssm import Mamba


class DualBranchCrossAttnBaseMasked(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
    ):
        super().__init__()

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.convs.append(GINEConv(nn=mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def serialize_graph(
        self,
        x,
        edge_index,
        batch,
    ):
        raise NotImplementedError

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

        graph_repr = global_mean_pool(x, batch)

        x_seq, padding_mask = self.serialize_graph(
            x,
            edge_index,
            batch,
        )

        x_seq = self.mamba(x_seq)

        # Mamba may generate non-zero outputs at padded positions.
        # Explicitly remove them before cross-attention.
        x_seq = x_seq.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        graph_token = graph_repr.unsqueeze(1)

        attn_out, _ = self.cross_attn(
            query=graph_token,
            key=x_seq,
            value=x_seq,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        fused = torch.cat(
            [
                graph_repr,
                attn_out.squeeze(1),
            ],
            dim=-1,
        )

        return self.pred_head(fused)
