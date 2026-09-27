#!/usr/bin/env python3
"""Verify Prefix50k CSVs, manifest, and optionally the official OGB split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from build_prefix50k import (
    TRAIN_COUNT,
    VALID_COUNT,
    load_official_prefixes,
)


def read_indices(path: Path) -> list[int]:
    values = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != ["index"]:
            raise ValueError(f"{path}: expected a single 'index' header")
        for line_number, row in enumerate(reader, 2):
            if len(row) != 1 or re.fullmatch(r"(0|[1-9][0-9]*)", row[0]) is None:
                raise ValueError(f"{path}:{line_number}: expected one nonnegative integer")
            values.append(int(row[0]))
    return values


def check(split_dir: Path, root: Path | None) -> None:
    train_file = split_dir / "train_indices.csv"
    valid_file = split_dir / "valid_indices.csv"
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    train = read_indices(train_file)
    valid = read_indices(valid_file)

    if len(train) != TRAIN_COUNT or len(valid) != VALID_COUNT:
        raise ValueError(f"wrong counts: train={len(train)}, valid={len(valid)}")
    if len(set(train)) != TRAIN_COUNT or len(set(valid)) != VALID_COUNT:
        raise ValueError("duplicate indices within train or validation")
    if set(train).intersection(valid):
        raise ValueError("train and validation indices overlap")

    expected_metadata = {
        "dataset": "PCQM4Mv2",
        "subset_name": "PCQM4Mv2-Prefix50k",
        "train_count": TRAIN_COUNT,
        "valid_count": VALID_COUNT,
        "order_preserved": True,
        "random_subsampling": False,
        "scaffold_resplitting": False,
        "test_split_accessed": False,
        "csv_header": "index",
    }
    for key, expected in expected_metadata.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} does not match {expected!r}")
    for path, key in (
        (train_file, "train_indices_sha256"),
        (valid_file, "valid_indices_sha256"),
    ):
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != manifest.get(key):
            raise ValueError(f"{path.name} SHA256 does not match manifest")

    if root is not None:
        official_train, official_valid = load_official_prefixes(root)
        if train != official_train:
            mismatch = next(i for i, (a, b) in enumerate(zip(train, official_train)) if a != b)
            raise ValueError(f"train order/value differs from OGB at position {mismatch}")
        if valid != official_valid:
            mismatch = next(i for i, (a, b) in enumerate(zip(valid, official_valid)) if a != b)
            raise ValueError(f"valid order/value differs from OGB at position {mismatch}")
        print("PASS: exact ordered equality with official OGB train/valid prefixes")
    else:
        print("PASS: local CSV counts, integers, uniqueness, disjointness, and hashes")
        print("Official OGB equality: SKIPPED (supply --root to check)")
    print(f"PASS: train={len(train)}, valid={len(valid)}; no test split key used")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        help="Optional local OGB data root containing pcqm4m-v2/split_dict.pt"
    )
    parser.add_argument(
        "--split-dir", type=Path, default=Path(__file__).resolve().parent,
        help="Directory containing CSVs and manifest (default: this script's directory)"
    )
    args = parser.parse_args()
    try:
        check(args.split_dir, args.root)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
