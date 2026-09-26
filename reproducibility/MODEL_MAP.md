# Model map

The authoritative source/class mapping, base classes, task wrappers, and counts are in RELEASE_AUDIT.md section 1. The same source classes serve both QM9 targets; QM9 wrappers replace OGB categorical encoders with 11-to-hidden node and 4-to-hidden edge projections. GINE-Wide is the same GINEOnly class with hidden dimension 192 (PCQM) or 196 (QM9).

Fixed-Index Mamba: CrossAttn/dualbranch_crossattn_index_masked.py -> DualBranchCrossAttnRandomMasked -> DualBranchCrossAttnBaseMasked. Fixed-Index GRU and Transformer inherit that same class and replace the sequence encoder; the Transformer overrides forward to pass its padding mask to the encoder. NoSeq uses an independent pointwise adapter, retains masked cross-attention, and has no contextual sequence encoder.

Historical RandomMasked names must not be interpreted as random serialization for the formal Fixed-Index configs. The actual serializer uses torch.arange within each graph.
