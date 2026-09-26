import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import GINEConv, global_mean_pool


class PositionWiseAdapter(nn.Module):
    """
    Independent node-wise capacity adapter.

    Every node is transformed independently using the same feed-forward
    function. No information is exchanged between different nodes.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        adapter_dim: int = 453,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            hidden_dim,
            adapter_dim,
            bias=True,
        )

        self.fc2 = nn.Linear(
            adapter_dim,
            hidden_dim,
            bias=True,
        )

    def forward(self, x):
        return self.fc2(
            F.relu(
                self.fc1(x)
            )
        )


class GINEPointwiseAttnNoSeq(nn.Module):
    """
    Parameter-matched no-context control.

    Pipeline:
        GINE node encoding
        -> graph mean pooling
        -> padded node-set packing
        -> independent position-wise adapter
        -> graph-guided cross-attention
        -> prediction head

    The position-wise adapter processes every node independently.
    No positional encoding or node-to-node contextual operator is used.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        adapter_dim: int = 453,
        output_dim: int = 1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim

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
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
            )

            self.convs.append(
                GINEConv(nn=mlp)
            )

            self.batch_norms.append(
                nn.BatchNorm1d(
                    hidden_dim
                )
            )

        self.pointwise_adapter = PositionWiseAdapter(
            hidden_dim=hidden_dim,
            adapter_dim=adapter_dim,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.pred_head = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

    @staticmethod
    def pack_nodes(
        x,
        batch,
    ):
        """
        Pack sparse node representations into a padded dense tensor.

        This operation only prepares batched node sets for masked
        cross-attention. It does not apply positional information or
        contextual propagation.
        """

        if batch.numel() == 0:
            raise ValueError(
                "Cannot pack an empty node batch."
            )

        batch_size = int(
            batch.max().item()
        ) + 1

        sequences = []
        lengths = []

        for graph_id in range(batch_size):
            node_indices = torch.where(
                batch == graph_id
            )[0]

            graph_nodes = x[node_indices]

            sequences.append(
                graph_nodes
            )

            lengths.append(
                graph_nodes.size(0)
            )

        max_nodes = max(lengths)
        hidden_dim = x.size(-1)

        padded_nodes = x.new_zeros(
            batch_size,
            max_nodes,
            hidden_dim,
        )

        # True means padded / invalid.
        padding_mask = torch.ones(
            batch_size,
            max_nodes,
            dtype=torch.bool,
            device=x.device,
        )

        for graph_id, graph_nodes in enumerate(
            sequences
        ):
            length = graph_nodes.size(0)

            padded_nodes[
                graph_id,
                :length,
            ] = graph_nodes

            padding_mask[
                graph_id,
                :length,
            ] = False

        return padded_nodes, padding_mask

    def forward(self, data):
        x = self.atom_encoder(
            data.x
        )

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

        node_set, padding_mask = self.pack_nodes(
            x,
            data.batch,
        )

        node_set = self.pointwise_adapter(
            node_set
        )

        # The adapter has biases, so zero-padding may become non-zero.
        # Reset all invalid positions before attention.
        node_set = node_set.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        graph_token = graph_repr.unsqueeze(1)

        attn_out, _ = self.cross_attn(
            query=graph_token,
            key=node_set,
            value=node_set,
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

        return self.pred_head(
            fused
        )
