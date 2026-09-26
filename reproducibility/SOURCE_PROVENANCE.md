# Source provenance

Frozen paper snapshot verified by sha256sum -c SHA256SUMS.txt before and after release changes. Every A file below was compared byte-for-byte to its same relative path in the snapshot.

## A. Byte-identical paper source

- experiments/pcqm4mv2/common/train_noseq_attn.py (SHA-256 d78e30ae7cfae4efeabdc1d274db3e219e5358b7494260df3fd4c68a0c32be54)
- experiments/pcqm4mv2/common/train_sequence_model_gru_index.py (SHA-256 dc6be88059de0da8adcaa1880d8afe88fda997003b7ee186b236999d5f5b690b)
- experiments/pcqm4mv2/common/train_sequence_model_index.py (SHA-256 578fb630d0a8e1ba55d53d06893e63af14fb5cb9d7c5ec7578ae49b81bf86178)
- experiments/pcqm4mv2/common/train_sequence_model_transformer_index.py (SHA-256 172e797956555fc2d0441907c4b20664da170ece9eea019814c6c0311ccda40d)
- experiments/qm9/common/create_split.py (SHA-256 f32a53c935c56354c6742257d639a856528df5131dbd0eaf6f9bd43301dd2325)
- experiments/qm9/common/train_gap.py (SHA-256 b433265945319b0919ffd2156f981d00ab7f9024b141035b380816d8c0648e58)
- experiments/qm9/common/train_gap_noseq_attn.py (SHA-256 e3db4edd22210932d9a7c5ff563aaaa9b42e2cd4f2e07e45cfeaba0ca959b702)
- experiments/qm9/common/train_index.py (SHA-256 410c44bc621095f25c65986dc1ce20e75be11606fa14949b4317cd6b40d76f3b)
- experiments/qm9/common/train_noseq_attn.py (SHA-256 bbbf6a4f66e2a348aa8d690bc38d75e9671f57cf21e8c20e628a1d72fa2ca53e)
- models/dual_branch/CrossAttn/dualbranch_crossattn_base_masked.py (SHA-256 e196f214a008b79e911ec73ca33cb038bccbb46e6355a5be1bfec9725c4f91e4)
- models/dual_branch/CrossAttn/dualbranch_crossattn_index_masked.py (SHA-256 360b7361c8e7557dae9cc5010b4c3612bd61d10f7d84d40ddefea325f16d3ea9)
- models/dual_branch/CrossAttn/dualbranch_crossattn_random_masked.py (SHA-256 89fe893c7dbce7aae8a07f5aea60cd62f5a6b24992e2a0e201939eb30cfdd437)
- models/dual_branch/SequenceComparisons/__init__.py (SHA-256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855)
- models/dual_branch/SequenceComparisons/dualbranch_crossattn_index_gru_masked.py (SHA-256 7e663ca984e012f74e85f598f1fbcafadaca93aaea17e50ec5e7bfc5cfb86fb8)
- models/dual_branch/SequenceComparisons/dualbranch_crossattn_index_transformer_masked.py (SHA-256 c0f4944c25e24b4a74788c5abab8cce15fc5a1b7c18e7ea5f65def6c19e41828)
- models/dual_branch/SequenceComparisons/dualbranch_crossattn_random_gru_masked.py (SHA-256 a16da17af0497d4c6fa9f76b321cff372094f67284ce3dcf3ead6d7558ae3c27)
- models/dual_branch/SequenceComparisons/dualbranch_crossattn_random_transformer_masked.py (SHA-256 c4eaef26e7bcf228445594da4ec184550a71a441e79f5ab1ebb2979e2bb1fc4c)
- models/dual_branch/SequenceComparisons/gine_only.py (SHA-256 51f27b2c392693ce4a0f5f4cc253baf9ff69160cbbc806331d000c6fd8b2d0e6)
- models/dual_branch/SequenceComparisons/gine_pointwise_attn_noseq.py (SHA-256 981022d6be5761456405dbaa0e14abacaea119117e812ee3aa193e49eb4b7eff)
- models/dual_branch/TaskModels/__init__.py (SHA-256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855)
- models/dual_branch/TaskModels/qm9_noseq_task_model.py (SHA-256 88a84b2fecb56a91d3ab27d5d55e87c8d7d3ccf348b06680a68280194f5fadc3)
- models/dual_branch/TaskModels/qm9_task_models_index.py (SHA-256 f6516fff6e20ee7334d81938afd2c40f5356fc406f2c303407ca8a2740544627)
- pcqm4m_dataset.py (SHA-256 9c7c6c561465cb61d67a53ded458d4bd64feaa2244735d4d34da5a1c945dba48)
- pcqm4mv2_node_relabeling_batch_audit.py (SHA-256 550312ca853a06523c9c19888dde41b14e8cef51d5e67ee69843a3e10909b614)
- qm9_mu_ordering_audit.py (SHA-256 83adaf9ded0a94ad4f67cdbaaf842276a424273c8f66033a85d229fbb1b9d7ce)
- experiments/qm9/split_indices.pt (SHA-256 2519cc8d0a3d9ee081e918d3c596abd1aca80de45281eaedc30148ae106e6f68)
- experiments/qm9/split_indices_gap_verified.pt (SHA-256 9b3b64e4b42a0d4b10ada1d685a4bd2249c62fee6e9a1b11cae2d683e745e0c0)

## B. Release-only non-mathematical edits

The 90 formal JSON configuration files listed below differ from the frozen snapshot only in log_root: the private absolute log output root was replaced with a repository-relative logs/... output root. All remaining bytes and all other JSON values are identical. This changes output-directory handling only; no constructor, forward pass, dimension, attention, masking, split, target, seed, optimizer, scheduler, early-stopping, or checkpoint-selection field was changed.

- experiments/pcqm4mv2/baseline/config_50k.json: log_root only
- experiments/pcqm4mv2/baseline/config_50k_seed123.json: log_root only
- experiments/pcqm4mv2/baseline/config_50k_seed2024.json: log_root only
- experiments/pcqm4mv2/baseline/config_50k_seed3407.json: log_root only
- experiments/pcqm4mv2/baseline/config_50k_seed7.json: log_root only
- experiments/pcqm4mv2/baseline_wide/config_50k_seed123.json: log_root only
- experiments/pcqm4mv2/baseline_wide/config_50k_seed2024.json: log_root only
- experiments/pcqm4mv2/baseline_wide/config_50k_seed3407.json: log_root only
- experiments/pcqm4mv2/baseline_wide/config_50k_seed42.json: log_root only
- experiments/pcqm4mv2/baseline_wide/config_50k_seed7.json: log_root only
- experiments/pcqm4mv2/masked_mamba/config_50k_index_seed123.json: log_root only
- experiments/pcqm4mv2/masked_mamba/config_50k_index_seed2024.json: log_root only
- experiments/pcqm4mv2/masked_mamba/config_50k_index_seed3407.json: log_root only
- experiments/pcqm4mv2/masked_mamba/config_50k_index_seed42.json: log_root only
- experiments/pcqm4mv2/masked_mamba/config_50k_index_seed7.json: log_root only
- experiments/pcqm4mv2/matched_gru/config_50k.json: log_root only
- experiments/pcqm4mv2/matched_gru/config_50k_seed123.json: log_root only
- experiments/pcqm4mv2/matched_gru/config_50k_seed2024.json: log_root only
- experiments/pcqm4mv2/matched_gru/config_50k_seed3407.json: log_root only
- experiments/pcqm4mv2/matched_gru/config_50k_seed7.json: log_root only
- experiments/pcqm4mv2/matched_noseq_attn/config_50k_seed123.json: log_root only
- experiments/pcqm4mv2/matched_noseq_attn/config_50k_seed2024.json: log_root only
- experiments/pcqm4mv2/matched_noseq_attn/config_50k_seed3407.json: log_root only
- experiments/pcqm4mv2/matched_noseq_attn/config_50k_seed42.json: log_root only
- experiments/pcqm4mv2/matched_noseq_attn/config_50k_seed7.json: log_root only
- experiments/pcqm4mv2/matched_transformer/config_50k_index_seed123.json: log_root only
- experiments/pcqm4mv2/matched_transformer/config_50k_index_seed2024.json: log_root only
- experiments/pcqm4mv2/matched_transformer/config_50k_index_seed3407.json: log_root only
- experiments/pcqm4mv2/matched_transformer/config_50k_index_seed42.json: log_root only
- experiments/pcqm4mv2/matched_transformer/config_50k_index_seed7.json: log_root only
- experiments/qm9/baseline/config_seed123.json: log_root only
- experiments/qm9/baseline/config_seed2024.json: log_root only
- experiments/qm9/baseline/config_seed3407.json: log_root only
- experiments/qm9/baseline/config_seed42.json: log_root only
- experiments/qm9/baseline/config_seed7.json: log_root only
- experiments/qm9/baseline_wide/config_seed123.json: log_root only
- experiments/qm9/baseline_wide/config_seed2024.json: log_root only
- experiments/qm9/baseline_wide/config_seed3407.json: log_root only
- experiments/qm9/baseline_wide/config_seed42.json: log_root only
- experiments/qm9/baseline_wide/config_seed7.json: log_root only
- experiments/qm9/gap/baseline/config_seed123.json: log_root only
- experiments/qm9/gap/baseline/config_seed2024.json: log_root only
- experiments/qm9/gap/baseline/config_seed3407.json: log_root only
- experiments/qm9/gap/baseline/config_seed42.json: log_root only
- experiments/qm9/gap/baseline/config_seed7.json: log_root only
- experiments/qm9/gap/baseline_wide/config_seed123.json: log_root only
- experiments/qm9/gap/baseline_wide/config_seed2024.json: log_root only
- experiments/qm9/gap/baseline_wide/config_seed3407.json: log_root only
- experiments/qm9/gap/baseline_wide/config_seed42.json: log_root only
- experiments/qm9/gap/baseline_wide/config_seed7.json: log_root only
- experiments/qm9/gap/masked_mamba/config_seed123.json: log_root only
- experiments/qm9/gap/masked_mamba/config_seed2024.json: log_root only
- experiments/qm9/gap/masked_mamba/config_seed3407.json: log_root only
- experiments/qm9/gap/masked_mamba/config_seed42.json: log_root only
- experiments/qm9/gap/masked_mamba/config_seed7.json: log_root only
- experiments/qm9/gap/matched_gru/config_seed123.json: log_root only
- experiments/qm9/gap/matched_gru/config_seed2024.json: log_root only
- experiments/qm9/gap/matched_gru/config_seed3407.json: log_root only
- experiments/qm9/gap/matched_gru/config_seed42.json: log_root only
- experiments/qm9/gap/matched_gru/config_seed7.json: log_root only
- experiments/qm9/gap/matched_transformer/config_seed123.json: log_root only
- experiments/qm9/gap/matched_transformer/config_seed2024.json: log_root only
- experiments/qm9/gap/matched_transformer/config_seed3407.json: log_root only
- experiments/qm9/gap/matched_transformer/config_seed42.json: log_root only
- experiments/qm9/gap/matched_transformer/config_seed7.json: log_root only
- experiments/qm9/gap/noseq_attn/config_seed123.json: log_root only
- experiments/qm9/gap/noseq_attn/config_seed2024.json: log_root only
- experiments/qm9/gap/noseq_attn/config_seed3407.json: log_root only
- experiments/qm9/gap/noseq_attn/config_seed42.json: log_root only
- experiments/qm9/gap/noseq_attn/config_seed7.json: log_root only
- experiments/qm9/masked_mamba/config_seed123.json: log_root only
- experiments/qm9/masked_mamba/config_seed2024.json: log_root only
- experiments/qm9/masked_mamba/config_seed3407.json: log_root only
- experiments/qm9/masked_mamba/config_seed42.json: log_root only
- experiments/qm9/masked_mamba/config_seed7.json: log_root only
- experiments/qm9/matched_gru/config_seed123.json: log_root only
- experiments/qm9/matched_gru/config_seed2024.json: log_root only
- experiments/qm9/matched_gru/config_seed3407.json: log_root only
- experiments/qm9/matched_gru/config_seed42.json: log_root only
- experiments/qm9/matched_gru/config_seed7.json: log_root only
- experiments/qm9/matched_transformer/config_seed123.json: log_root only
- experiments/qm9/matched_transformer/config_seed2024.json: log_root only
- experiments/qm9/matched_transformer/config_seed3407.json: log_root only
- experiments/qm9/matched_transformer/config_seed42.json: log_root only
- experiments/qm9/matched_transformer/config_seed7.json: log_root only
- experiments/qm9/noseq_attn/config_seed123.json: log_root only
- experiments/qm9/noseq_attn/config_seed2024.json: log_root only
- experiments/qm9/noseq_attn/config_seed3407.json: log_root only
- experiments/qm9/noseq_attn/config_seed42.json: log_root only
- experiments/qm9/noseq_attn/config_seed7.json: log_root only

## Release-created material

README.md, .gitignore, reproducibility documentation and validation scripts/reports, and results/reported_results.csv are release-created, not copies of mathematical model source. The results CSV has a header only because the frozen source did not contain traceable formal manuscript result values.
