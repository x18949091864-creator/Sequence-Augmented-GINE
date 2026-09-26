from pathlib import Path

import torch
from torch_geometric.datasets import QM9


DATA_ROOT = "data/QM9"
OUTPUT_PATH = Path("experiments/qm9/split_indices.pt")

TARGET_INDEX = 0
SPLIT_SEED = 42

NUM_TRAIN = 110000
NUM_VALID = 10000


def main() -> None:
    dataset = QM9(root=DATA_ROOT)
    num_samples = len(dataset)

    expected_total = 130831
    if num_samples != expected_total:
        raise RuntimeError(
            f"Unexpected QM9 size: {num_samples}, "
            f"expected {expected_total}"
        )

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(
        num_samples,
        generator=generator,
    )

    train_indices = permutation[:NUM_TRAIN]
    valid_indices = permutation[
        NUM_TRAIN:NUM_TRAIN + NUM_VALID
    ]
    test_indices = permutation[
        NUM_TRAIN + NUM_VALID:
    ]

    # Compute normalization statistics from training targets only.
    train_targets = torch.cat(
        [
            dataset[int(index)].y[:, TARGET_INDEX]
            for index in train_indices
        ],
        dim=0,
    ).float()

    target_mean = train_targets.mean()
    target_std = train_targets.std(unbiased=True)

    if not torch.isfinite(target_mean):
        raise RuntimeError("Target mean is not finite.")

    if not torch.isfinite(target_std) or target_std <= 0:
        raise RuntimeError("Target std is invalid.")

    split = {
        "dataset": "QM9",
        "target_index": TARGET_INDEX,
        "target_name": "mu",
        "target_unit": "D",
        "split_seed": SPLIT_SEED,
        "num_samples": num_samples,
        "train": train_indices,
        "valid": valid_indices,
        "test": test_indices,
        "target_mean": target_mean,
        "target_std": target_std,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(split, OUTPUT_PATH)

    print("Dataset:", split["dataset"])
    print("Target:", split["target_name"])
    print("Target index:", split["target_index"])
    print("Target unit:", split["target_unit"])
    print("Total:", num_samples)
    print("Train:", len(train_indices))
    print("Validation:", len(valid_indices))
    print("Test:", len(test_indices))
    print("Target mean:", float(target_mean))
    print("Target std:", float(target_std))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
