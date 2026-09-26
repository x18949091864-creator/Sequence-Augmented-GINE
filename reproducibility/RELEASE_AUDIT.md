# Release audit (frozen snapshot, 2026-09-26)

## Scope and verification gate

Release worktree: this repository. Branch release/chemical-papers, base HEAD 5651c6c46adf4ee37256cd7445c0dcf6de0f6eef, initially clean. The frozen snapshot's SHA256SUMS.txt passed sha256sum -c for every listed file. All paths below are relative to the frozen snapshot. This audit precedes release cleanup. No training, dataset download, or GPU use occurred.

## 1. Model map

All 12 trainable-parameter counts below were measured by CPU construction in the frozen snapshot with the equimamba environment, CUDA_VISIBLE_DEVICES empty. PCQM is the PCQM4Mv2-Prefix50k family. QM9 uses node_feature_dim=11 and edge_feature_dim=4; no coordinates enter the wrappers.

| Paper model | PCQM source/class and base | QM9 wrapper source/class | Configuration | PCQM count | QM9 count |
| --- | --- | --- | --- | ---: | ---: |
| GINE-128 | models/dual_branch/SequenceComparisons/gine_only.py: GINEOnly (nn.Module) | models/dual_branch/TaskModels/qm9_task_models_index.py: QM9GINEOnlyTask (GINEOnly) | hidden 128, four GINE layers, mean pooling, no attention | 173697 | 151937 |
| GINE-Wide | same GINEOnly class | same QM9GINEOnlyTask class | PCQM hidden 192; QM9 hidden 196; four GINE layers | 371137 | 352605 |
| GINE-Attn (NoSeq) | models/dual_branch/SequenceComparisons/gine_pointwise_attn_noseq.py: GINEPointwiseAttnNoSeq (nn.Module) | models/dual_branch/TaskModels/qm9_noseq_task_model.py: QM9GINEPointwiseAttnNoSeqTask (GINEPointwiseAttnNoSeq) | hidden 128, adapter width 453, four attention heads | 372678 | 350918 |
| GINE-GRU | models/dual_branch/SequenceComparisons/dualbranch_crossattn_index_gru_masked.py: DualBranchCrossAttnRandomGRUMasked (DualBranchCrossAttnRandomMasked) | models/dual_branch/TaskModels/qm9_task_models_index.py: QM9GINEGRUTask (DualBranchCrossAttnRandomGRUMasked) | hidden 128, one unidirectional GRU layer, four attention heads | 355201 | 333441 |
| GINE-Transformer | models/dual_branch/SequenceComparisons/dualbranch_crossattn_index_transformer_masked.py: DualBranchCrossAttnRandomTransformerMasked (DualBranchCrossAttnRandomMasked) | models/dual_branch/TaskModels/qm9_task_models_index.py: QM9GINETransformerTask (DualBranchCrossAttnRandomTransformerMasked) | hidden 128, one Transformer layer, FFN 256, four encoder heads, four cross-attention heads | 388609 | 366849 |
| GINE-Mamba | models/dual_branch/CrossAttn/dualbranch_crossattn_index_masked.py: DualBranchCrossAttnRandomMasked (DualBranchCrossAttnBaseMasked in models/dual_branch/CrossAttn/dualbranch_crossattn_base_masked.py) | models/dual_branch/TaskModels/qm9_task_models_index.py: QM9GINEMambaTask (DualBranchCrossAttnRandomMasked) | hidden 128, d_state 16, d_conv 4, expand 2, four attention heads | 372609 | 350849 |

These are current final Fixed-Index classes selected by the formal configs. Their retained RandomMasked class names are historical identifiers; dualbranch_crossattn_index_masked.py serializes original per-graph node index order, not a random permutation. GINE-Wide is a config of GINEOnly, not a separate class.

## 2. Experiment map

| Task | Training entrypoints and config families | Dataset, target, split | Training and selection |
| --- | --- | --- | --- |
| PCQM4Mv2-Prefix50k gap | experiments/pcqm4mv2/common/train_sequence_model_index.py for GINE-128, GINE-Wide, Mamba; train_sequence_model_gru_index.py for GRU; train_sequence_model_transformer_index.py for Transformer; train_noseq_attn.py for NoSeq. Config families: baseline, baseline_wide, masked_mamba (config_50k_index_seed*.json), matched_gru, matched_transformer (config_50k_index_seed*.json), matched_noseq_attn. | pcqm4m_dataset.py: PCQM4MPyGDataset wraps official OGB PCQM4Mv2Dataset.get_idx_split(); Subset(range(50000)) of official train, Subset(range(10000)) of official valid. Original dataset indices are preserved by wrapper __getitem__. Target is dataset y, HOMO-LUMO gap. No paper test subset is constructed in these entrypoints. | Seeds 7, 42, 123, 2024, 3407 across formal config families. 50 epochs maximum, AdamW lr 0.0001, batch 8, ReduceLROnPlateau factor 0.5/patience 3, early stop patience 20, best checkpoint on minimum validation MAE. |
| QM9 dipole mu | experiments/qm9/common/train_index.py for GINE-128, GINE-Wide, Mamba, GRU, Transformer; train_noseq_attn.py for NoSeq. Config families in experiments/qm9/{baseline,baseline_wide,masked_mamba,matched_gru,matched_transformer,noseq_attn}. | torch_geometric.datasets.QM9 with experiments/qm9/split_indices.pt. Saved target index 0, name mu, unit D. Saved split seed 42; 110000 train, 10000 valid, 10831 test; 130831 distinct indices. Coordinates in data.pos are ignored by model forward. | Model seeds 7, 42, 123, 2024, 3407. 150 epochs maximum, AdamW lr 0.0003, batch 128, ReduceLROnPlateau factor 0.5/patience 3, early stop patience 20, best checkpoint on validation MAE in D, then final test from best checkpoint. |
| QM9 HOMO-LUMO gap | experiments/qm9/common/train_gap.py for GINE-128, GINE-Wide, Mamba, GRU, Transformer; train_gap_noseq_attn.py for NoSeq. Config families in experiments/qm9/gap/{baseline,baseline_wide,masked_mamba,matched_gru,matched_transformer,noseq_attn}. | QM9 with experiments/qm9/split_indices_gap_verified.pt. Saved target index 4, name gap, unit eV. Saved split seed 42; same 110000/10000/10831 index partition as mu, all distinct. | Same five model seeds; 150 epochs maximum, AdamW lr 0.0003, batch 128, scheduler factor 0.5/patience 3, early stop patience 20, best checkpoint on validation MAE in eV, then final test from best checkpoint. |

The unverified gap split file and verified gap split file have identical train, valid, and test indices; both match the mu split indices. The verified file is the one named by formal gap configs. Formal config log_root values are private absolute paths and require release-only JSON edits before reproduction. Seed 42 is the split-generation seed, separate from the five model seeds. The PCQM test labels are not used by the formal Prefix50k training scripts.

## 3. Shared component map

- OGB AtomEncoder/BondEncoder and four GINEConv + BatchNorm1d + ReLU layers: models/dual_branch/CrossAttn/dualbranch_crossattn_base_masked.py for Mamba/GRU/Transformer; independent matched equivalents in SequenceComparisons/gine_only.py and gine_pointwise_attn_noseq.py. GINEConv is called without eps/train_eps overrides; local PyG 2.7.0 signature defaults are eps=0.0, train_eps=False.
- QM9 node and edge projection: models/dual_branch/TaskModels/qm9_task_models_index.py and qm9_noseq_task_model.py replace OGB encoders with Linear(11, hidden_dim) and Linear(4, hidden_dim).
- Fixed-Index serialization/padding: models/dual_branch/CrossAttn/dualbranch_crossattn_index_masked.py: DualBranchCrossAttnRandomMasked.serialize_graph. It uses torch.arange over each graph's original node positions and builds zero-padded sequences and boolean padding_mask (True means invalid).
- NoSeq node packing and pointwise adapter: SequenceComparisons/gine_pointwise_attn_noseq.py. It masks adapter outputs at padded positions.
- Post-Mamba output zeroing, graph mean pooling, cross-attention, and 256-to-128-to-1 prediction head: CrossAttn/dualbranch_crossattn_base_masked.py. GRU inherits this forward path after replacing self.mamba with a one-layer GRU adapter. Transformer has its own masking-aware forward in SequenceComparisons/dualbranch_crossattn_index_transformer_masked.py. Both zero padded sequence outputs.
- QM9 task heads are rebuilt by build_graph_prediction_head and build_dual_branch_prediction_head in TaskModels/qm9_task_models_index.py; final output_dim is one.

## 4. Attention audit

DualBranchCrossAttnBaseMasked uses torch.nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True); NoSeq has the same operator. Head dimension d_h=32. The graph mean representation is query, node representations are key and value, and key_padding_mask=padding_mask excludes padded positions. The index Transformer override and NoSeq forward retain this mask. No final paper source overrides MultiheadAttention score scaling.

The implementation uses PyTorch standard scaled dot-product attention with 1/sqrt(d_h). The d_h denominator in the submitted manuscript equation is a typographical error. The implementation and all reported results are unchanged.

## 5. Legacy / exclude map

Exclude from the primary reproduction path: old unmasked CrossAttn base.py, index.py, random.py, BFS/DFS classes, Concat exploratory variants, *.before_mask_fix and *.bak_before_* files, random-train/fixed-eval exploratory variants, BBBP/EGNN, MolHIV/MolPCBA experiments, root historical training scripts, visualization prototypes, raw prediction exporters, temporary files, data, logs, checkpoints, and IDE/cache material. Do not modify the frozen snapshot.

Import caveat: PCQM primary training modules import random GRU/Transformer or random Mamba modules at module import time even when their selected config is Fixed-Index. These files are import dependencies and must be retained if those entrypoints remain byte-identical. Their presence does not make the random serialization a primary paper run. Excluding them without a proven harmless import refactor would break imports.

## 6. Hard-coded path audit

MUST FIX FOR RELEASE: Every formal PCQM and QM9 JSON config in the snapshot has a private absolute home-directory log_root. Change only the output path in copied release configs to a relative logs/... path; this does not alter model, split, target, optimizer, scheduler, early stop, or checkpoint criterion. The formal training scripts' default dataset roots are relative data/pcqm4m and data/QM9, and their split paths are relative.

SAFE HISTORICAL TEXT: snapshot metadata naming the original path is provenance only; do not copy it into primary executable paths.

NOT IN PRIMARY RELEASE PATH: other historical checkpoint relabeling scripts and raw export scripts contain absolute private run/checkpoint roots. Two portable auxiliary audit scripts (PCQM batch and QM9 mu ordering) are retained byte-identical; they require external locked checkpoints and are not primary training entrypoints.

## 7. Import risk map

- The package root must be the working directory or on PYTHONPATH: imports start with models.dual_branch..., experiments..., and pcqm4m_dataset.
- Several class names are misleading or duplicate: Fixed-Index class DualBranchCrossAttnRandomMasked; fixed GRU/Transformer classes also say Random; random and index files define identically named GRUSequenceEncoder/TransformerSequenceEncoder.
- PCQM main training entrypoints eagerly import random comparison modules; retain their transitive base imports to preserve importability.
- QM9 wrappers inherit PCQM bases and replace atom/bond encoders; flattening or renaming would risk behavior and checkpoint state names.
- PyG, OGB, RDKit, mamba_ssm and other third-party dependencies are public packages in the owner-provided equimamba environment. No private local Python package was found in the traced primary model imports.
- No circular import was observed in the traced model dependency graph. Training modules are script entrypoints with root-relative data/config/log assumptions.

## Environment and remaining verification

Owner-provided environment: Python 3.10.20, PyTorch 2.5.1+cu121, PyG 2.7.0, OGB 1.3.6, RDKit 2022.09.5, mamba_ssm 2.2.4, measured from installed packages. These values characterize the available validation environment; the frozen source snapshot has no environment lock file proving the exact original experiment versions. CPU parameter counts pass; import, architecture, masking, forward, provenance, large-file, and secret/path release audits remain for Phase 2. The frozen snapshot contains source/config/split files but no formal manuscript results table, so no result number will be fabricated.
