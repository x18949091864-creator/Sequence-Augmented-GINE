# Reproduction guide

Work from the repository root in a compatible Python 3.10 environment. Obtain PCQM4Mv2 from OGB and QM9 from PyG separately. Place them at data/pcqm4m and data/QM9 respectively, or set dataset_root in a local config copy. PCQM4Mv2 is not redistributed. Keep the included QM9 split index files unchanged.

## PCQM4Mv2-Prefix50k gap

Every listed JSON family has five formal seed configs: 7, 42, 123, 2024, 3407. For baseline and GRU, the seed-42 file is config_50k.json. For other families it is config_50k_seed42.json or config_50k_index_seed42.json as shown.

| Model | Entry point | Seed-42 config |
| --- | --- | --- |
| GINE-128 | experiments/pcqm4mv2/common/train_sequence_model_index.py | experiments/pcqm4mv2/baseline/config_50k.json |
| GINE-Wide | experiments/pcqm4mv2/common/train_sequence_model_index.py | experiments/pcqm4mv2/baseline_wide/config_50k_seed42.json |
| GINE-Attn (NoSeq) | experiments/pcqm4mv2/common/train_noseq_attn.py | experiments/pcqm4mv2/matched_noseq_attn/config_50k_seed42.json |
| GINE-GRU | experiments/pcqm4mv2/common/train_sequence_model_gru_index.py | experiments/pcqm4mv2/matched_gru/config_50k.json |
| GINE-Transformer | experiments/pcqm4mv2/common/train_sequence_model_transformer_index.py | experiments/pcqm4mv2/matched_transformer/config_50k_index_seed42.json |
| GINE-Mamba | experiments/pcqm4mv2/common/train_sequence_model_index.py | experiments/pcqm4mv2/masked_mamba/config_50k_index_seed42.json |

The dataset wrapper uses official OGB train and valid split indices, followed by the first 50000 and 10000 split positions respectively. It does not reshuffle or scaffold sample the subset. Training loader shuffling affects epoch order, not subset membership.

## QM9 mu and gap

For mu, use experiments/qm9/common/train_index.py for every model except NoSeq, which uses train_noseq_attn.py. Config paths are experiments/qm9/<family>/config_seed<seed>.json, where families are baseline, baseline_wide, noseq_attn, matched_gru, matched_transformer, and masked_mamba. For gap, use experiments/qm9/common/train_gap.py or train_gap_noseq_attn.py, with analogous configs under experiments/qm9/gap/<family>/. The included mu split has target 0; gap configs use split_indices_gap_verified.pt with target 4.

Examples:

    python experiments/pcqm4mv2/common/train_sequence_model_gru_index.py --config experiments/pcqm4mv2/matched_gru/config_50k.json
    python experiments/qm9/common/train_noseq_attn.py --config experiments/qm9/noseq_attn/config_seed42.json
    python experiments/qm9/common/train_gap.py --config experiments/qm9/gap/matched_transformer/config_seed42.json

These are training commands for independent reproduction only. Release validation runs compile, import, parameter, architecture, masking, and tiny synthetic CPU forward checks for ten CPU-compatible model instances. The installed mamba_ssm backend requires CUDA tensors, so the two GINE-Mamba instances were checked by one separately authorized no-gradient synthetic CUDA forward each; no training was run. The retained auxiliary locked-checkpoint node-relabeling audit scripts are pcqm4mv2_node_relabeling_batch_audit.py and qm9_mu_ordering_audit.py. They require external formal checkpoints and datasets; the QM9 mu audit uses GPU inference. Release validation does not run either audit.
