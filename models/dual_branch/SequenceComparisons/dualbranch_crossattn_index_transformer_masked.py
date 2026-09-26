import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool

from models.dual_branch.CrossAttn.dualbranch_crossattn_index_masked import (
    DualBranchCrossAttnRandomMasked,
)


class TransformerSequenceEncoder(nn.Module):
    """
    Single-layer Transformer encoder with explicit padding masking.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.0,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

    def forward(
        self,
        x,
        padding_mask=None,
    ):
        return self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )


class DualBranchCrossAttnRandomTransformerMasked(
    DualBranchCrossAttnRandomMasked
):
    """
    Padding-aware GINE-Transformer comparison model.

    The graph encoder, serialization, cross-attention, and prediction head
    match the masked GINE-Mamba implementation. The sequence encoder is
    replaced by a one-layer Transformer encoder.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        transformer_heads=4,
        dim_feedforward=256,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        self.mamba = TransformerSequenceEncoder(
            hidden_dim=hidden_dim,
            num_heads=transformer_heads,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        for conv, bn in zip(
            self.convs,
            self.batch_norms,
        ):
            x = conv(
                x,
                edge_index,
                edge_attr,
            )
            x = bn(x)
            x = F.relu(x)

        graph_repr = global_mean_pool(
            x,
            batch,
        )

        x_seq, padding_mask = self.serialize_graph(
            x,
            edge_index,
            batch,
        )

        x_seq = self.mamba(
            x_seq,
            padding_mask=padding_mask,
        )

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
