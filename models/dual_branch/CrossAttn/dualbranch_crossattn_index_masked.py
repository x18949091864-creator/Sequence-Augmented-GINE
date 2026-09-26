import torch

from models.dual_branch.CrossAttn.dualbranch_crossattn_base_masked import (
    DualBranchCrossAttnBaseMasked
)


class DualBranchCrossAttnRandomMasked(
    DualBranchCrossAttnBaseMasked
):
    def serialize_graph(
        self,
        x,
        edge_index,
        batch,
    ):
        batch_size = int(batch.max().item()) + 1

        sequences = []
        lengths = []

        for graph_id in range(batch_size):
            node_indices = torch.where(batch == graph_id)[0]
            sub_x = x[node_indices]

            # Fixed Index protocol:
            # original node-index order is used during both training
            # and validation/test.
            order = torch.arange(
                sub_x.size(0),
                device=x.device,
            )

            seq = sub_x[order]

            sequences.append(seq)
            lengths.append(seq.size(0))

        max_nodes = max(lengths)
        hidden_dim = x.size(-1)

        padded_sequences = x.new_zeros(
            batch_size,
            max_nodes,
            hidden_dim,
        )

        # True means padded/invalid position for MultiheadAttention.
        padding_mask = torch.ones(
            batch_size,
            max_nodes,
            dtype=torch.bool,
            device=x.device,
        )

        for graph_id, seq in enumerate(sequences):
            length = seq.size(0)

            padded_sequences[
                graph_id,
                :length,
            ] = seq

            padding_mask[
                graph_id,
                :length,
            ] = False

        return padded_sequences, padding_mask
