import torch.nn as nn

from models.dual_branch.SequenceComparisons.gine_pointwise_attn_noseq import (
    GINEPointwiseAttnNoSeq,
)


class QM9GINEPointwiseAttnNoSeqTask(
    GINEPointwiseAttnNoSeq
):
    """
    Two-dimensional QM9 no-context control.

    Atomic coordinates in data.pos are intentionally ignored.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        adapter_dim: int = 453,
        node_feature_dim: int = 11,
        edge_feature_dim: int = 4,
        output_dim: int = 1,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            adapter_dim=adapter_dim,
            output_dim=output_dim,
        )

        self.atom_encoder = nn.Linear(
            node_feature_dim,
            hidden_dim,
        )

        self.bond_encoder = nn.Linear(
            edge_feature_dim,
            hidden_dim,
        )
