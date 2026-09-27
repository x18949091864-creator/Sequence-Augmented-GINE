# PCQM4Mv2-Prefix50k

PCQM4Mv2-Prefix50k is a deterministic subset definition of the official OGB PCQM4Mv2 dataset, **not** a separately redistributed molecular dataset. Obtain the molecular records from OGB.

The paper's training subset is the first 50,000 entries of the official OGB training split-index array; validation is the first 10,000 entries of the official validation array. Their order is preserved. The original experiments used no random subsampling, scaffold resplitting, target filtering, or molecule filtering to construct these prefixes, and used no test split.

The two CSV files contain one actual zero-based PCQM4Mv2 dataset index per row under the header `index`. They contain no molecular records, SMILES, graphs, or target values. `manifest.json` records their counts, construction, OGB package version, and SHA-256 hashes of the exact CSV bytes.

From the repository root, regenerate the files using a locally obtained OGB copy:

```bash
python data_splits/pcqm4mv2_prefix50k/build_prefix50k.py --root /path/to/ogb/data/root
```

The root must contain `pcqm4m-v2/split_dict.pt`. The builder calls OGB's `PCQM4Mv2Dataset.get_idx_split()` and reads only the official split file; it does not download or preprocess molecular data. It uses only the `train` and `valid` split entries.

Verify the committed files locally, then compare every index in order with the official split:

```bash
python data_splits/pcqm4mv2_prefix50k/verify_prefix50k.py
python data_splits/pcqm4mv2_prefix50k/verify_prefix50k.py --root /path/to/ogb/data/root
```

The first command checks counts, integer format, uniqueness, disjointness, and file hashes. The second also checks exact element-by-element equality with OGB's ordered train/valid prefixes. No test-dev or test-challenge split is used.
