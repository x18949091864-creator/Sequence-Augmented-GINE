import torch.nn as nn

from models.dual_branch.CrossAttn.dualbranch_crossattn_random_masked import (
    DualBranchCrossAttnRandomMasked,
)


class GRUSequenceEncoder(nn.Module):
    """
    Single-layer GRU adapter.

    The adapter returns only the sequence output so that its interface
    matches the Mamba module used by DualBranchCrossAttnRandomMasked.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def forward(self, x):
        output, _ = self.gru(x)
        return output


class DualBranchCrossAttnRandomGRUMasked(
    DualBranchCrossAttnRandomMasked
):
    """
    Parameter-matched GINE-GRU comparison model.

    All graph encoding, serialization, padding masking, cross-attention,
    and prediction components are inherited from the masked GINE-Mamba
    implementation. Only the Mamba sequence encoder is replaced by GRU.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        gru_num_layers=1,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # The inherited forward method calls self.mamba(x_seq).
        # Replacing this attribute preserves the same forward pipeline.
        self.mamba = GRUSequenceEncoder(
            hidden_dim=hidden_dim,
            num_layers=gru_num_layers,
        )
