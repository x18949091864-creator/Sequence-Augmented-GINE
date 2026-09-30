# Sequence-Augmented GINE for Molecular Quantum-Property Prediction

A Controlled Study of Capacity, Selective Readout, and Node Relabeling

This release supports a controlled comparison of model capacity, selective graph-guided readout, contextual node interaction, and sensitivity to node relabeling. It preserves the final manuscript implementation rather than replacing it with a new model.

## Models and tasks

The six models are GINE-128, GINE-Wide, GINE-Attn (NoSeq), GINE-GRU, GINE-Transformer, and GINE-Mamba.

The three primary tasks are:

- PCQM4Mv2-Prefix50k HOMO-LUMO gap
- QM9 dipole moment (mu)
- QM9 HOMO-LUMO gap

All models use molecular graph features. **No 3D coordinates are used as model inputs.**

| Model | PCQM trainable parameters | QM9 trainable parameters |
| --- | ---: | ---: |
| GINE-128 | 173697 | 151937 |
| GINE-Wide | 371137 | 352605 |
| GINE-Attn (NoSeq) | 372678 | 350918 |
| GINE-GRU | 355201 | 333441 |
| GINE-Transformer | 388609 | 366849 |
| GINE-Mamba | 372609 | 350849 |

These counts were confirmed by CPU instantiation of the frozen paper source. GINE-Wide has hidden dimension 192 for PCQM and 196 for QM9; augmented models use hidden dimension 128.

Model names in the source retain historical RandomMasked identifiers, but the formal primary configurations use Fixed-Index serialization.

## Data and fixed splits

### Dataset sources

The underlying molecular records are obtained from the original public dataset sources and are not redistributed in this repository.

- **PCQM4Mv2 (Open Graph Benchmark, OGB-LSC):**  
  https://ogb.stanford.edu/docs/lsc/pcqm4mv2/

- **QM9 original dataset (Figshare):**  
  https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904

- **QM9 preprocessing and dataset interface used in this study (PyTorch Geometric):**  
  https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html

### PCQM4Mv2-Prefix50k

PCQM4Mv2-Prefix50k is constructed deterministically from the official OGB PCQM4Mv2 split indices.

It uses:

- the first 50,000 entries of the **official OGB training split-index array**, and
- the first 10,000 entries of the **official OGB validation split-index array**,

while preserving the original split-index order.

No random subsampling or scaffold-based subsampling is performed.

The wrapper in `pcqm4m_dataset.py` maps the selected split positions to the corresponding official PCQM4Mv2 molecule indices.

**PCQM4Mv2 molecular records are not redistributed in this repository.** They should be obtained through the official OGB distribution linked above.

### QM9

QM9 uses the fixed split files included in this repository:

- 110,000 training molecules
- 10,000 validation molecules
- 10,831 test molecules

The saved split seed is 42.

The prediction targets are:

- dipole moment: target index 0, unit D
- HOMO-LUMO gap: target index 4, unit eV

The gap configurations use `split_indices_gap_verified.pt`.

The three saved index partitions were checked to be identical across the dipole-moment and gap tasks, and all 130,831 indices are distinct.

QM9 molecular records are not redistributed in this repository. They can be obtained from the original Figshare collection or through the PyTorch Geometric QM9 dataset interface linked above.

The five formal model-training seeds are **7, 42, 123, 2024, and 3407**.

For QM9, the selected prediction target is normalized using statistics calculated from the training split. Validation MAE in the original physical unit is used to select the best checkpoint.

## PCQM4Mv2-Prefix50k reproducibility

The exact ordered training and validation indices used in this study are provided in:

[`data_splits/pcqm4mv2_prefix50k/`](data_splits/pcqm4mv2_prefix50k/)

This directory contains:

- the exact 50,000 official OGB training indices used in the study;
- the exact 10,000 official OGB validation indices used in the study;
- a builder script that regenerates the subset indices from the official OGB split;
- a verification script for checking index counts, order, and integrity;
- SHA-256 hashes for integrity checking.

The underlying PCQM4Mv2 molecular records are obtained from OGB and are not redistributed here.

Using the official OGB PCQM4Mv2 distribution together with the published indices and reconstruction scripts allows the exact Prefix50k subset used in the manuscript to be reconstructed.

## Repository layout

- `models/dual_branch/`: final model classes and required import dependencies.
- `experiments/pcqm4mv2/`: formal PCQM4Mv2-Prefix50k training entry points and JSON configurations.
- `experiments/qm9/`: formal QM9 dipole-moment and HOMO-LUMO-gap training entry points, configurations, and fixed split indices.
- `data_splits/pcqm4mv2_prefix50k/`: exact ordered Prefix50k training and validation indices and reconstruction/verification utilities.
- `pcqm4m_dataset.py`: official-index PCQM4Mv2 dataset wrapper.
- `reproducibility/`: audit records, model map, environment information, reproduction commands, provenance, and validation documentation.
- `results/reported_results.csv`: source-backed reported-result records included in the release snapshot.

## Environment and reproduction

The experiments were developed using Python 3.10 with PyTorch, PyTorch Geometric, OGB, RDKit, and `mamba_ssm`.

The validation workstation environment and measured package versions are documented in:

[`reproducibility/ENVIRONMENT.md`](reproducibility/ENVIRONMENT.md)

The release snapshot does not contain a complete original environment lock file. Users should work from the repository root with the required dependencies and separately obtained datasets available.

Detailed commands and data prerequisites are provided in:

[`reproducibility/REPRODUCE.md`](reproducibility/REPRODUCE.md)

Example commands for seed 42 are shown below.

### PCQM4Mv2-Prefix50k

```bash
python experiments/pcqm4mv2/common/train_sequence_model_index.py \
  --config experiments/pcqm4mv2/masked_mamba/config_50k_index_seed42.json
