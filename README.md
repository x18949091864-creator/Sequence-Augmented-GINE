# Sequence-Augmented GINE for Molecular Quantum-Property Prediction

A Controlled Study of Capacity, Selective Readout, and Node Relabeling

This release supports a controlled comparison of model capacity, selective graph-guided readout, contextual node interaction, and sensitivity to node relabeling. It preserves the final manuscript implementation rather than replacing it with a new model.

## Models and tasks

The six models are GINE-128, GINE-Wide, GINE-Attn (NoSeq), GINE-GRU, GINE-Transformer, and GINE-Mamba. The three primary tasks are PCQM4Mv2-Prefix50k HOMO-LUMO gap, QM9 dipole moment (mu), and QM9 HOMO-LUMO gap. All use molecular graph features; **no 3D coordinates are model inputs**.

| Model | PCQM trainable parameters | QM9 trainable parameters |
| --- | ---: | ---: |
| GINE-128 | 173697 | 151937 |
| GINE-Wide | 371137 | 352605 |
| GINE-Attn (NoSeq) | 372678 | 350918 |
| GINE-GRU | 355201 | 333441 |
| GINE-Transformer | 388609 | 366849 |
| GINE-Mamba | 372609 | 350849 |

These counts were confirmed by CPU instantiation of the frozen paper source. GINE-Wide has hidden dimension 192 for PCQM and 196 for QM9; augmented models use 128. Model names in source retain historical RandomMasked identifiers, but the formal primary configs use Fixed-Index serialization.

## Data and fixed splits

PCQM4Mv2-Prefix50k uses the first 50,000 molecules of the **official OGB training split** and the first 10,000 molecules of the **official OGB validation split**, in the original dataset-index order. These are neither random samples nor scaffold samples. The wrapper in pcqm4m_dataset.py maps split positions to official indices. **PCQM4Mv2 data are not redistributed.** Obtain the dataset through the official OGB distribution.

QM9 uses the included fixed split files: 110000 training, 10000 validation, and 10831 test molecules. The saved split seed is 42. Dipole moment is target index 0 (D); HOMO-LUMO gap is target index 4 (eV). The gap configs use split_indices_gap_verified.pt. The three saved index partitions were checked to be identical across mu and gap, and the 130831 indices are distinct. Obtain QM9 through PyG; the raw and processed dataset is not redistributed here.

The five formal model seeds are **7, 42, 123, 2024, 3407**. The selected target is normalized using training-split statistics for QM9; validation MAE in the original unit selects the best checkpoint.

## PCQM4Mv2-Prefix50k reproducibility

The exact ordered training and validation indices are in [data_splits/pcqm4mv2_prefix50k/](data_splits/pcqm4mv2_prefix50k/). The directory contains 50,000 official OGB training indices and 10,000 official OGB validation indices, a builder that regenerates them from the official split, and a verifier for their counts, hashes, and order. The underlying PCQM4Mv2 molecular records are obtained from OGB and are not redistributed here.

## Repository layout

- models/dual_branch/: final model classes and their required import dependencies.
- experiments/pcqm4mv2/: formal Prefix50k training entrypoints and JSON configs.
- experiments/qm9/: formal mu and gap training entrypoints, configs, and fixed split indices.
- pcqm4m_dataset.py: official-index PCQM wrapper.
- reproducibility/: audit, model map, environment, commands, provenance, and validation.
- results/reported_results.csv: source-backed reported values only; currently header-only because no formal manuscript results table was in the frozen source snapshot.

## Environment and reproduction

Use Python 3.10 with PyTorch, PyG, OGB, RDKit, and mamba_ssm. The validation workstation environment and measured package versions are in reproducibility/ENVIRONMENT.md. The snapshot does not contain a complete original environment lock file. Work from the repository root with dependencies and separately obtained datasets available.

Examples for seed 42:

    python experiments/pcqm4mv2/common/train_sequence_model_index.py --config experiments/pcqm4mv2/masked_mamba/config_50k_index_seed42.json
    python experiments/qm9/common/train_index.py --config experiments/qm9/masked_mamba/config_seed42.json
    python experiments/qm9/common/train_gap.py --config experiments/qm9/gap/masked_mamba/config_seed42.json

These commands start training and are provided for users who explicitly choose to reproduce the experiments; they were not run during release preparation. All model/task commands and data prerequisites are listed in reproducibility/REPRODUCE.md. JSON log_root paths are relative to this repository and point under logs/, which is ignored by Git.

## Node-relabeling audit

The formal setting serializes nodes by their original per-graph indices. A node permutation therefore tests whether the learned predictor changes when the same graph is relabeled. The retained auxiliary scripts pcqm4mv2_node_relabeling_batch_audit.py and qm9_mu_ordering_audit.py are byte-identical to the frozen research snapshot. They require separately obtained datasets and nonredistributed locked checkpoints; the QM9 mu script also requires a GPU when a user runs its formal inference audit. Other historical audit variants with private absolute paths are excluded. This release documents the Fixed-Index serializer and audit provenance in reproducibility/RELEASE_AUDIT.md. No relabeling result is claimed without traceable formal records.

## Release status

Parameter counts and static architecture/masking checks pass. Synthetic CPU forwards pass for all ten CPU-compatible task-family instances. The installed mamba_ssm 2.2.4 backend requires CUDA tensors, so CPU forward is not applicable to the two GINE-Mamba instances. One authorized no-gradient CUDA synthetic forward per exact retained GINE-Mamba implementation passed on physical GPU 1, giving 12/12 executable validations in their supported runtime. No training or model-source modification was performed. See reproducibility/FINAL_RELEASE_AUDIT.md for the final release decision and remaining provenance limits.
