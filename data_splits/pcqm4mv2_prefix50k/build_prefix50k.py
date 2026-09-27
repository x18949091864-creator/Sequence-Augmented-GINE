#!/usr/bin/env python3
"""Build the paper's ordered prefixes of the official PCQM4Mv2 train/valid splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

TRAIN_COUNT = 50_000
VALID_COUNT = 10_000


def load_official_prefixes(root: Path) -> tuple[list[int], list[int]]:
    """Use OGB's split API without loading molecular records or downloading data."""
    from ogb.lsc import PCQM4Mv2Dataset

    folder = root.expanduser().resolve() / "pcqm4m-v2"
    split_file = folder / "split_dict.pt"
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Official split file not found: {split_file}. "
            "Supply --root as the parent of pcqm4m-v2; no download is attempted."
        )

    # OGB's normal constructor reads the complete molecular CSV even with
    # only_smiles=True. get_idx_split() only needs self.folder, so construct a
    # split-only instance and never read raw/processed molecules.
    dataset = PCQM4Mv2Dataset.__new__(PCQM4Mv2Dataset)
    dataset.folder = str(folder)
    split = dataset.get_idx_split()
    result = []
    for name, count in (("train", TRAIN_COUNT), ("valid", VALID_COUNT)):
        values = np.asarray(split[name])
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Official {name} split must be a 1-D integer array")
        if len(values) < count:
            raise ValueError(f"Official {name} split has only {len(values)} entries")
        prefix = [int(value) for value in values[:count]]
        if any(value < 0 for value in prefix):
            raise ValueError(f"Official {name} prefix contains a negative index")
        result.append(prefix)
    return result[0], result[1]


def check_prefixes(train: list[int], valid: list[int]) -> None:
    if len(train) != TRAIN_COUNT or len(valid) != VALID_COUNT:
        raise ValueError("Prefix counts must be 50000 train and 10000 valid")
    if len(set(train)) != TRAIN_COUNT or len(set(valid)) != VALID_COUNT:
        raise ValueError("Duplicate index within a prefix")
    if set(train).intersection(valid):
        raise ValueError("Train and validation prefixes overlap")


def write_csv(path: Path, indices: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["index"])
        writer.writerows((index,) for index in indices)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Local OGB data root containing pcqm4m-v2/split_dict.pt"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path(__file__).resolve().parent,
        help="Output directory (default: this script's directory)"
    )
    args = parser.parse_args()
    train, valid = load_official_prefixes(args.root)
    check_prefixes(train, valid)

    args.outdir.mkdir(parents=True, exist_ok=True)
    train_file = args.outdir / "train_indices.csv"
    valid_file = args.outdir / "valid_indices.csv"
    write_csv(train_file, train)
    write_csv(valid_file, valid)

    import ogb
    manifest = {
        "dataset": "PCQM4Mv2",
        "subset_name": "PCQM4Mv2-Prefix50k",
        "train_count": TRAIN_COUNT,
        "valid_count": VALID_COUNT,
        "construction": (
            "First 50000 entries of official OGB train split and first 10000 "
            "entries of official OGB valid split, in the returned order"
        ),
        "source": "official OGB PCQM4Mv2 split indices via PCQM4Mv2Dataset.get_idx_split()",
        "order_preserved": True,
        "random_subsampling": False,
        "scaffold_resplitting": False,
        "test_split_accessed": False,
        "csv_header": "index",
        "ogb_version": ogb.__version__,
        "train_indices_sha256": sha256_file(train_file),
        "valid_indices_sha256": sha256_file(valid_file),
    }
    manifest_file = args.outdir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: generated {TRAIN_COUNT} train and {VALID_COUNT} valid indices")
    print(f"train_indices.csv SHA256: {manifest['train_indices_sha256']}")
    print(f"valid_indices.csv SHA256: {manifest['valid_indices_sha256']}")


if __name__ == "__main__":
    main()
