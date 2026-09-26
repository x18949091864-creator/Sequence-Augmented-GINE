import torch.nn as nn

from models.dual_branch.SequenceComparisons.gine_only import (
    GINEOnly,
)
from models.dual_branch.CrossAttn.dualbranch_crossattn_index_masked import (
    DualBranchCrossAttnRandomMasked,
)
from models.dual_branch.SequenceComparisons.dualbranch_crossattn_index_transformer_masked import (
    DualBranchCrossAttnRandomTransformerMasked,
)
from models.dual_branch.SequenceComparisons.dualbranch_crossattn_index_gru_masked import (
    DualBranchCrossAttnRandomGRUMasked,
)


def build_graph_prediction_head(
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, output_dim),
    )


def build_dual_branch_prediction_head(
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, output_dim),
    )


class QM9GINEOnlyTask(GINEOnly):
    """
    Two-dimensional GINE baseline for QM9.

    Uses only:
        data.x
        data.edge_index
        data.edge_attr

    The three-dimensional coordinates in data.pos are intentionally ignored.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        node_feature_dim: int = 11,
        edge_feature_dim: int = 4,
        output_dim: int = 1,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Replace the OGB categorical encoders with continuous-feature
        # projections suitable for PyG QM9 features.
        self.atom_encoder = nn.Linear(
            node_feature_dim,
            hidden_dim,
        )

        self.bond_encoder = nn.Linear(
            edge_feature_dim,
            hidden_dim,
        )

        self.pred_head = build_graph_prediction_head(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )


class QM9GINEMambaTask(
    DualBranchCrossAttnRandomMasked
):
    """
    Fixed-index GINE-Mamba model for QM9.

    Uses only two-dimensional molecular graph features and does not use
    atomic coordinates from data.pos.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        node_feature_dim: int = 11,
        edge_feature_dim: int = 4,
        output_dim: int = 1,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Replace the OGB categorical encoders with continuous-feature
        # projections suitable for PyG QM9 features.
        self.atom_encoder = nn.Linear(
            node_feature_dim,
            hidden_dim,
        )

        self.bond_encoder = nn.Linear(
            edge_feature_dim,
            hidden_dim,
        )

        self.pred_head = build_dual_branch_prediction_head(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

class QM9GINETransformerTask(
    DualBranchCrossAttnRandomTransformerMasked
):
    """
    Fixed-index GINE-Transformer model for QM9.

    The graph encoder, node serialization, cross-attention fusion,
    and prediction head follow the GINE-Mamba setting. Only the
    sequence encoder is replaced by a one-layer Transformer.
    Atomic coordinates in data.pos are intentionally ignored.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        transformer_heads: int = 4,
        dim_feedforward: int = 256,
        node_feature_dim: int = 11,
        edge_feature_dim: int = 4,
        output_dim: int = 1,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            transformer_heads=transformer_heads,
            dim_feedforward=dim_feedforward,
        )

        self.atom_encoder = nn.Linear(
            node_feature_dim,
            hidden_dim,
        )

        self.bond_encoder = nn.Linear(
            edge_feature_dim,
            hidden_dim,
        )

        self.pred_head = build_dual_branch_prediction_head(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

class QM9GINEGRUTask(
    DualBranchCrossAttnRandomGRUMasked
):
    """
    Fixed-index GINE-GRU model for QM9.

    The graph encoder, serialization, cross-attention fusion,
    and prediction head follow the same protocol as GINE-Mamba.
    Only the sequence encoder is replaced by a one-layer GRU.
    Atomic coordinates in data.pos are intentionally ignored.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        gru_num_layers: int = 1,
        node_feature_dim: int = 11,
        edge_feature_dim: int = 4,
        output_dim: int = 1,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            gru_num_layers=gru_num_layers,
        )

        self.atom_encoder = nn.Linear(
            node_feature_dim,
            hidden_dim,
        )

        self.bond_encoder = nn.Linear(
            edge_feature_dim,
            hidden_dim,
        )

        self.pred_head = build_dual_branch_prediction_head(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

