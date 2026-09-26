# Final release audit

## Decision

**SAFE TO COMMIT: YES.** Snapshot integrity, source provenance, imports, parameter counts, architecture, masking, splits, and synthetic forwards pass under the supported runtimes. The installed mamba_ssm backend requires CUDA tensors. Therefore, CPU synthetic forward validation was not applicable to the two GINE-Mamba instances. The exact retained PCQM and QM9 GINE-Mamba implementations were subsequently validated by one no-gradient CUDA synthetic forward pass each. Both produced finite outputs with the expected batch shape. No training and no model-source modification were performed. The documented formal-data and historical-results limits below remain; they do not indicate a failed source or model check. No commit or push was performed.

## Repository and snapshot

- Authorized release worktree on branch release/chemical-papers at base HEAD 5651c6c46adf4ee37256cd7445c0dcf6de0f6eef.
- Snapshot SHA256SUMS.txt: 260/260 entries pass, checked before and after release work and after CUDA validation.
- Original archive and frozen snapshot were read only.
- Exactly 181 previously tracked release-only files were removed by explicit git rm paths/directories after REMOVAL_MANIFEST.txt was written. Categories: DATA 12, CHECKPOINT 21, LOG 105, IDE 7, CACHE 12, LEGACY 16, UNRELATED 8. Static import-chain check found no removal collision. Import, count, architecture, masking, and smoke reports were rerun after each deletion group and remained byte-identical to the pre-removal reports.
- No commit or push occurred.

## Provenance and implementation

- 27 retained source/split files remain byte-identical to the verified frozen snapshot after CUDA validation, including two auxiliary node-relabeling audit scripts. See SOURCE_PROVENANCE.md for individual SHA-256 values.
- 90 formal JSON configs differ only in log_root, changed from a private absolute location to relative logs/... . No scientific parameter, data split, target, seed, optimizer, scheduler, stopping, or checkpoint-selection field changed.
- Fixed-Index serializer preserves per-graph original node order. PCQM Prefix50k uses the first 50000 official OGB training split entries and first 10000 official validation split entries, by original indices.
- QM9 mu target 0 and gap target 4 use verified 110000/10000/10831 split files, split seed 42, all indices unique.
- Cross-attention is torch.nn.MultiheadAttention with embed_dim 128, four heads, graph query, node key/value, and key_padding_mask. Standard score scaling is 1/sqrt(d_h), where d_h=32. The submitted manuscript's d_h denominator is a typographical equation error; implementation and reported results are unchanged.
- Primary model/experiment paths have no unconditional GPU selection. The auxiliary QM9 formal ordering audit intentionally requires GPU inference; that full checkpoint/data audit was not run.

## Validation

| Check | Result |
| --- | --- |
| Python compileall -q . after final updates | PASS |
| Primary import without dataset download | PASS, nine entrypoint/loader modules; all model classes imported |
| Auxiliary node-audit imports | PASS with its documented output-directory environment variable |
| Trainable parameter counts | PASS 12/12 |
| Architecture invariants | PASS 12/12 |
| Serialization/padding/masking | PASS for all augmented variants |
| CPU synthetic forward, two unequal-size legal molecular graphs | PASS 10/10 CPU-compatible; two GINE-Mamba instances NOT APPLICABLE on CPU with installed backend |
| Authorized CUDA synthetic inference, same 3-node and 5-node graphs | PASS 2/2 GINE-Mamba on physical GPU 1 (NVIDIA RTX 4090; torch cuda:0), output (2,1) and finite |
| Overall executable model validation | PASS 12/12 in supported runtime |
| Prefix50k and QM9 target/split audit | PASS |
| Files larger than 10 MB | PASS, none |
| Secret/private-path scan of retained text | PASS, no possible credentials, private-key blocks, or private absolute paths |
| Source byte comparison | PASS 27/27; config-only diff verification PASS 90/90 |

## Limits and open items

- The installed Mamba backend requires CUDA tensors for inference; GINE-Mamba CPU forward is not supported by this validation environment. The one-pass CUDA evidence is in cuda_smoke_check.json and smoke_test.txt. No CPU support is claimed.
- Locked-checkpoint node-relabeling scripts are retained but formal audit inference was not run: datasets and formal checkpoints are not redistributed, and the QM9 mu audit requires GPU. The README documents this limitation.
- results/reported_results.csv contains only a header. The frozen source snapshot does not contain a traceable formal manuscript results table; values were not guessed.
- The frozen snapshot does not contain an exact original environment lock file. ENVIRONMENT.md reports versions measured from the owner-provided validation environment.

The release is compact, source-traceable, and ready for a human-reviewed commit. This audit did not create a commit or push.
