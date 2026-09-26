#!/usr/bin/env python3
"""Inference-only node-reindexing audit for the formal QM9-mu runs.

Run this script from the EquiMamba repository root.  It loads the 12 audited
best checkpoints (four models x three training seeds), reproduces Fixed-Index
test predictions, and evaluates the same 10 deterministic node reindexings
used by the QM9-gap audit.  It never trains or modifies an experiment run.

Required environment variable:
    MU_ORDERING_AUDIT_DIR   New output directory for this audit.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from experiments.qm9.common.train_index import build_model


REPO_ROOT = Path.cwd().resolve()
SCRIPT_PATH = Path(__file__).resolve()
SPLIT_PATH = (REPO_ROOT / "experiments/qm9/split_indices.pt").resolve()
DATASET_ROOT = (REPO_ROOT / "data/QM9").resolve()
OUTPUT_DIR = Path(os.environ["MU_ORDERING_AUDIT_DIR"]).expanduser().resolve()

TRAINING_SEEDS = (42, 123, 3407)
PERMUTATION_IDS = tuple(range(1, 11))
PERMUTATION_BASE_SEED = 20_260_811
DATASET_INDEX_MULTIPLIER = 1_000_003
PERMUTATION_ID_MULTIPLIER = 1_000_000_007
TEST_BATCH_SIZE = 256
NUM_WORKERS = 0
FIXED_MAE_TOLERANCE_D = 5.0e-6

EXPECTED_SPLIT_SHA256 = (
    "2519cc8d0a3d9ee081e918d3c596abd1aca80de45281eaedc30148ae106e6f68"
)
EXPECTED_SIZES = {
    "num_samples": 130_831,
    "train": 110_000,
    "valid": 10_000,
    "test": 10_831,
}

MODEL_SPECS = {
    "GINE-only": {
        "sequence_model": "gine",
        "config_dir": REPO_ROOT / "experiments/qm9/baseline",
        "experiment_stem": "qm9_mu_gine_seed",
        "output_folder": "gine_only",
    },
    "GINE-GRU": {
        "sequence_model": "gru",
        "config_dir": REPO_ROOT / "experiments/qm9/matched_gru",
        "experiment_stem": "qm9_mu_gru_index_seed",
        "output_folder": "gine_gru",
    },
    "GINE-Transformer": {
        "sequence_model": "transformer",
        "config_dir": REPO_ROOT / "experiments/qm9/matched_transformer",
        "experiment_stem": "qm9_mu_transformer_index_seed",
        "output_folder": "gine_transformer",
    },
    "GINE-Mamba": {
        "sequence_model": "mamba",
        "config_dir": REPO_ROOT / "experiments/qm9/masked_mamba",
        "experiment_stem": "qm9_mu_mamba_index_seed",
        "output_folder": "gine_mamba",
    },
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_json_safe(payload),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sample_sd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.std(array, ddof=1)) if array.size > 1 else float("nan")


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target), dtype=np.float64))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    require(
        all(list(row.keys()) == fieldnames for row in rows),
        f"Inconsistent CSV columns: {path}",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"Refusing to write empty gzip CSV: {path}")
    fieldnames = list(rows[0].keys())
    require(
        all(list(row.keys()) == fieldnames for row in rows),
        f"Inconsistent gzip CSV columns: {path}",
    )
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tensor_indices(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long).view(-1).cpu()
    require(tensor.numel() > 0, f"Empty split: {name}")
    require(
        torch.unique(tensor).numel() == tensor.numel(),
        f"Duplicate indices in {name}",
    )
    return tensor


def discover_run(config: dict[str, Any]) -> Path:
    log_root = Path(str(config["log_root"])).expanduser().resolve()
    require(log_root.is_dir(), f"Missing log root: {log_root}")
    expected_name = str(config["experiment_name"])
    candidates: list[Path] = []
    for config_path in sorted(log_root.rglob("config.json")):
        run_dir = config_path.parent
        if not (run_dir / "result.json").is_file():
            continue
        if not (run_dir / "checkpoints/best_model.pt").is_file():
            continue
        try:
            saved_config = load_json(config_path)
        except Exception:
            continue
        if saved_config.get("experiment_name") == expected_name:
            candidates.append(run_dir)
    if len(candidates) != 1:
        rendered = "\n".join(f"  - {path}" for path in candidates) or "  (none)"
        fail(
            f"Expected exactly one complete run for {expected_name}; "
            f"found {len(candidates)}:\n{rendered}"
        )
    return candidates[0]


def check_formal_config(
    config: dict[str, Any], model_label: str, training_seed: int
) -> None:
    spec = MODEL_SPECS[model_label]
    require(
        config.get("experiment_name")
        == f"{spec['experiment_stem']}{training_seed}",
        f"{model_label}/seed{training_seed}: wrong experiment_name",
    )
    require(
        str(config.get("sequence_model", "")).lower() == spec["sequence_model"],
        f"{model_label}/seed{training_seed}: wrong sequence_model",
    )
    require(
        int(config.get("seed")) == training_seed,
        f"{model_label}/seed{training_seed}: wrong training seed",
    )
    require(
        int(config.get("target_index")) == 0,
        f"{model_label}/seed{training_seed}: target_index is not 0",
    )
    configured_split = (REPO_ROOT / str(config.get("split_path"))).resolve()
    require(
        configured_split == SPLIT_PATH,
        f"{model_label}/seed{training_seed}: unexpected split path",
    )
    configured_dataset = (REPO_ROOT / str(config.get("dataset_root"))).resolve()
    require(
        configured_dataset == DATASET_ROOT,
        f"{model_label}/seed{training_seed}: unexpected dataset root",
    )


def load_formal_run(
    model_label: str,
    training_seed: int,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], Path, int]:
    spec = MODEL_SPECS[model_label]
    config_path = Path(spec["config_dir"]) / f"config_seed{training_seed}.json"
    config = load_json(config_path)
    check_formal_config(config, model_label, training_seed)

    run_dir = discover_run(config)
    saved_config = load_json(run_dir / "config.json")
    check_formal_config(saved_config, model_label, training_seed)
    require(
        saved_config == config,
        f"{model_label}/seed{training_seed}: source and run configs differ",
    )
    result = load_json(run_dir / "result.json")
    require(
        result.get("target_name") == "mu"
        and result.get("target_unit") == "D"
        and int(result.get("target_index")) == 0,
        f"{model_label}/seed{training_seed}: invalid result target metadata",
    )
    require(
        int(result.get("seed")) == training_seed,
        f"{model_label}/seed{training_seed}: result seed mismatch",
    )

    checkpoint_path = run_dir / "checkpoints/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), "Checkpoint must be a dictionary")
    require("model_state_dict" in checkpoint, "Checkpoint lacks model_state_dict")
    require("config" in checkpoint, "Checkpoint lacks config")
    require(
        math.isclose(
            float(torch.as_tensor(checkpoint["target_mean"]).item()),
            target_mean,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        f"{model_label}/seed{training_seed}: checkpoint target mean mismatch",
    )
    require(
        math.isclose(
            float(torch.as_tensor(checkpoint["target_std"]).item()),
            target_std,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        f"{model_label}/seed{training_seed}: checkpoint target SD mismatch",
    )

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    del checkpoint
    return model, config, result, run_dir, parameter_count


def deterministic_permutation_seed(
    global_dataset_index: int, permutation_id: int
) -> int:
    return int(
        PERMUTATION_BASE_SEED
        + DATASET_INDEX_MULTIPLIER * int(global_dataset_index)
        + PERMUTATION_ID_MULTIPLIER * int(permutation_id)
    )


def reindex_graph(data: Any, global_dataset_index: int, permutation_id: int) -> Any:
    data = data.clone()
    number_of_nodes = int(data.num_nodes)
    require(number_of_nodes > 0, "Encountered an empty molecular graph")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        deterministic_permutation_seed(global_dataset_index, permutation_id)
    )
    permutation = torch.randperm(number_of_nodes, generator=generator)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(number_of_nodes, dtype=torch.long)

    original_edge_index = data.edge_index
    for key in list(data.keys()):
        if key in {"edge_index", "edge_attr", "y"}:
            continue
        value = data[key]
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and int(value.size(0)) == number_of_nodes
            and data.is_node_attr(key)
        ):
            data[key] = value.index_select(0, permutation)
    data.edge_index = inverse[original_edge_index]

    require(int(data.num_nodes) == number_of_nodes, "Node count changed during reindexing")
    require(
        data.edge_index.shape == original_edge_index.shape,
        "Edge-index shape changed during reindexing",
    )
    return data


class QM9AuditSubset(Dataset):
    def __init__(
        self,
        dataset: QM9,
        global_indices: np.ndarray,
        permutation_id: int | None,
    ) -> None:
        self.dataset = dataset
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.permutation_id = permutation_id

    def __len__(self) -> int:
        return int(self.global_indices.size)

    def __getitem__(self, position: int) -> Any:
        global_index = int(self.global_indices[position])
        data = self.dataset.get(global_index)
        if self.permutation_id is None:
            return data
        return reindex_graph(data, global_index, self.permutation_id)


def make_loader(
    dataset: QM9,
    test_indices: np.ndarray,
    permutation_id: int | None,
) -> DataLoader:
    subset = QM9AuditSubset(dataset, test_indices, permutation_id)
    return DataLoader(
        subset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        prediction_normalized = model(batch).view(-1)
        prediction = prediction_normalized * target_std + target_mean
        target = batch.y[:, 0].view(-1)
        prediction_parts.append(
            prediction.detach().to(dtype=torch.float64, device="cpu").numpy()
        )
        target_parts.append(target.detach().to(dtype=torch.float64, device="cpu").numpy())
    prediction_array = np.concatenate(prediction_parts)
    target_array = np.concatenate(target_parts)
    require(
        prediction_array.shape == target_array.shape,
        "Prediction/target shape mismatch",
    )
    require(
        np.isfinite(prediction_array).all() and np.isfinite(target_array).all(),
        "Non-finite prediction or target detected",
    )
    return prediction_array, target_array


def configure_inference() -> torch.device:
    require(torch.cuda.is_available(), "CUDA is unavailable; expose physical GPU 1")
    require(
        torch.cuda.device_count() == 1,
        "Expected exactly one visible GPU; run with CUDA_VISIBLE_DEVICES=1",
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda:0")


def main() -> None:
    print("===== QM9-MU FORMAL ORDERING AUDIT =====")
    print(f"Repository: {REPO_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Mode: inference only; no training")

    require(
        (REPO_ROOT / "experiments/qm9/common/train_index.py").is_file(),
        "Run this script from the EquiMamba repository root",
    )
    require(SPLIT_PATH.is_file(), f"Missing split file: {SPLIT_PATH}")
    require(
        DATASET_ROOT.is_dir() and (DATASET_ROOT / "processed").is_dir(),
        f"Existing processed QM9 dataset not found at {DATASET_ROOT}",
    )
    require(
        not OUTPUT_DIR.exists(),
        f"Output directory already exists: {OUTPUT_DIR}",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    device = configure_inference()
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Test batch size: {TEST_BATCH_SIZE}")
    print(f"Permutation IDs: {list(PERMUTATION_IDS)}")

    require(
        sha256_file(SPLIT_PATH) == EXPECTED_SPLIT_SHA256,
        "Fixed split SHA-256 mismatch",
    )
    split = torch.load(SPLIT_PATH, map_location="cpu", weights_only=False)
    require(isinstance(split, dict), "Split file must contain a dictionary")
    require(
        split.get("dataset") == "QM9"
        and split.get("target_name") == "mu"
        and split.get("target_unit") == "D"
        and int(split.get("target_index")) == 0,
        "Split is not the formal QM9-mu split",
    )
    require(int(split.get("split_seed")) == 42, "Split seed is not 42")
    require(
        int(split.get("num_samples")) == EXPECTED_SIZES["num_samples"],
        "Unexpected full dataset size in split metadata",
    )
    split_indices = {
        name: tensor_indices(split[name], name) for name in ("train", "valid", "test")
    }
    for name in ("train", "valid", "test"):
        require(
            int(split_indices[name].numel()) == EXPECTED_SIZES[name],
            f"Unexpected {name} size",
        )
    all_indices = torch.cat(list(split_indices.values()))
    require(
        torch.unique(all_indices).numel() == EXPECTED_SIZES["num_samples"],
        "Split overlap or incomplete coverage detected",
    )

    target_mean = float(torch.as_tensor(split["target_mean"]).item())
    target_std = float(torch.as_tensor(split["target_std"]).item())
    test_indices = split_indices["test"].numpy().astype(np.int64, copy=True)
    dataset = QM9(root=str(DATASET_ROOT))
    require(len(dataset) == EXPECTED_SIZES["num_samples"], "Unexpected QM9 length")

    protocol = {
        "audit_version": "qm9_mu_ordering_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "inference_only_no_training",
        "repository": str(REPO_ROOT),
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "dataset": "QM9",
        "dataset_root": str(DATASET_ROOT),
        "target_index": 0,
        "target_name": "mu",
        "target_unit": "D",
        "split_seed": 42,
        "split_path": str(SPLIT_PATH),
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "test_size": int(test_indices.size),
        "test_indices_sha256": sha256_array(test_indices),
        "training_seeds": list(TRAINING_SEEDS),
        "number_of_permutations": len(PERMUTATION_IDS),
        "permutation_ids": list(PERMUTATION_IDS),
        "permutation_seed_formula": (
            "20260811 + 1000003 * global_dataset_index "
            "+ 1000000007 * permutation_id"
        ),
        "permutation_semantics": (
            "new_node_position_j receives old_node_position_permutation[j]; "
            "edge endpoints are remapped by the inverse permutation; node-wise "
            "attributes move with their nodes; graph targets and edge attributes "
            "remain unchanged"
        ),
        "shared_permutations_across_models_and_training_seeds": True,
        "fixed_mae_reproduction_tolerance_D": FIXED_MAE_TOLERANCE_D,
        "test_batch_size": TEST_BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "device_policy": "physical GPU 1 exposed as cuda:0",
    }
    write_json(OUTPUT_DIR / "audit_protocol.json", protocol)

    permutation_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    canonical_target: np.ndarray | None = None

    print("\n===== FIXED AND PERMUTED INFERENCE =====")
    for model_label, spec in MODEL_SPECS.items():
        for training_seed in TRAINING_SEEDS:
            model, _, result, formal_run_dir, parameter_count = load_formal_run(
                model_label,
                training_seed,
                device,
                target_mean,
                target_std,
            )
            output_run_dir = (
                OUTPUT_DIR / str(spec["output_folder"]) / f"seed_{training_seed}"
            )
            output_run_dir.mkdir(parents=True, exist_ok=False)

            fixed_prediction, target = predict(
                model,
                make_loader(dataset, test_indices, None),
                device,
                target_mean,
                target_std,
            )
            require(target.size == test_indices.size, "Unexpected fixed target size")
            if canonical_target is None:
                canonical_target = target.copy()
            else:
                require(
                    np.array_equal(target, canonical_target),
                    f"Target mismatch for {model_label}/seed{training_seed}",
                )

            fixed_mae_D = mae(fixed_prediction, target)
            stored_test_mae_D = float(result["test_mae_d"])
            fixed_reproduction_error_D = abs(fixed_mae_D - stored_test_mae_D)
            require(
                fixed_reproduction_error_D <= FIXED_MAE_TOLERANCE_D,
                f"{model_label}/seed{training_seed}: fixed MAE {fixed_mae_D:.10f} D "
                f"does not reproduce stored {stored_test_mae_D:.10f} D",
            )

            permutation_predictions: list[np.ndarray] = []
            permutation_maes_D: list[float] = []
            for permutation_id in PERMUTATION_IDS:
                prediction, permuted_target = predict(
                    model,
                    make_loader(dataset, test_indices, permutation_id),
                    device,
                    target_mean,
                    target_std,
                )
                require(
                    np.array_equal(permuted_target, target),
                    f"Target changed for {model_label}/seed{training_seed}/"
                    f"permutation{permutation_id}",
                )
                permutation_mae_D = mae(prediction, target)
                permutation_predictions.append(prediction)
                permutation_maes_D.append(permutation_mae_D)
                permutation_rows.append(
                    {
                        "model": model_label,
                        "training_seed": training_seed,
                        "permutation_id": permutation_id,
                        "permutation_mae_D": permutation_mae_D,
                        "fixed_mae_D": fixed_mae_D,
                        "change_vs_fixed_D": permutation_mae_D - fixed_mae_D,
                        "change_vs_fixed_percent": (
                            100.0 * (permutation_mae_D - fixed_mae_D) / fixed_mae_D
                        ),
                    }
                )
                print(
                    f"{model_label:18s} seed {training_seed:4d} "
                    f"perm {permutation_id:2d}/10 | MAE {permutation_mae_D:.10f} D"
                )

            permutation_matrix = np.stack(permutation_predictions, axis=0)
            require(
                permutation_matrix.shape == (len(PERMUTATION_IDS), test_indices.size),
                "Unexpected permutation prediction matrix shape",
            )
            prediction_mean_D = np.mean(permutation_matrix, axis=0, dtype=np.float64)
            prediction_sd_D = np.std(permutation_matrix, axis=0, ddof=0)
            mean_absolute_shift_D = np.mean(
                np.abs(permutation_matrix - fixed_prediction[None, :]),
                axis=0,
                dtype=np.float64,
            )
            maximum_absolute_shift_D = np.max(
                np.abs(permutation_matrix - fixed_prediction[None, :]), axis=0
            )
            mean_single_permutation_absolute_error_D = np.mean(
                np.abs(permutation_matrix - target[None, :]),
                axis=0,
                dtype=np.float64,
            )
            fixed_absolute_error_D = np.abs(fixed_prediction - target)
            ensemble_absolute_error_D = np.abs(prediction_mean_D - target)
            ordering_error_penalty_D = (
                mean_single_permutation_absolute_error_D - fixed_absolute_error_D
            )
            ensemble_gain_D = (
                mean_single_permutation_absolute_error_D - ensemble_absolute_error_D
            )

            mean_permutation_mae_D = float(np.mean(permutation_maes_D))
            ensemble_mae_D = mae(prediction_mean_D, target)
            seed_row = {
                "model": model_label,
                "training_seed": training_seed,
                "parameters": parameter_count,
                "formal_run_directory": str(formal_run_dir),
                "n_test_molecules": int(target.size),
                "n_permutations": len(PERMUTATION_IDS),
                "stored_test_mae_D": stored_test_mae_D,
                "fixed_mae_D": fixed_mae_D,
                "fixed_reproduction_error_D": fixed_reproduction_error_D,
                "mean_single_permutation_mae_D": mean_permutation_mae_D,
                "single_permutation_mae_sample_sd_D": sample_sd(permutation_maes_D),
                "minimum_single_permutation_mae_D": float(np.min(permutation_maes_D)),
                "maximum_single_permutation_mae_D": float(np.max(permutation_maes_D)),
                "random_only_ensemble_mae_D": ensemble_mae_D,
                "mean_random_change_vs_fixed_D": mean_permutation_mae_D - fixed_mae_D,
                "mean_random_change_vs_fixed_percent": (
                    100.0 * (mean_permutation_mae_D - fixed_mae_D) / fixed_mae_D
                ),
                "ensemble_change_vs_fixed_D": ensemble_mae_D - fixed_mae_D,
                "ensemble_change_vs_fixed_percent": (
                    100.0 * (ensemble_mae_D - fixed_mae_D) / fixed_mae_D
                ),
                "mean_per_molecule_prediction_sd_D": float(np.mean(prediction_sd_D)),
                "mean_per_molecule_absolute_shift_D": float(
                    np.mean(mean_absolute_shift_D)
                ),
            }
            seed_rows.append(seed_row)

            np.savez_compressed(
                output_run_dir / "predictions_and_targets.npz",
                test_indices=test_indices,
                target=target,
                fixed_prediction=fixed_prediction,
                permutation_ids=np.asarray(PERMUTATION_IDS, dtype=np.int64),
                permutation_predictions=permutation_matrix,
                permutation_maes_D=np.asarray(permutation_maes_D, dtype=np.float64),
                per_molecule_prediction_sd_D=prediction_sd_D,
                per_molecule_mean_absolute_shift_D=mean_absolute_shift_D,
            )
            per_molecule_rows = [
                {
                    "test_position": position,
                    "global_dataset_index": int(test_indices[position]),
                    "target_D": float(target[position]),
                    "fixed_prediction_D": float(fixed_prediction[position]),
                    "fixed_absolute_error_D": float(fixed_absolute_error_D[position]),
                    "permutation_prediction_mean_D": float(prediction_mean_D[position]),
                    "permutation_prediction_sd_D": float(prediction_sd_D[position]),
                    "mean_absolute_prediction_shift_D": float(
                        mean_absolute_shift_D[position]
                    ),
                    "maximum_absolute_prediction_shift_D": float(
                        maximum_absolute_shift_D[position]
                    ),
                    "mean_single_permutation_absolute_error_D": float(
                        mean_single_permutation_absolute_error_D[position]
                    ),
                    "ordering_error_penalty_D": float(
                        ordering_error_penalty_D[position]
                    ),
                    "random_only_ensemble_prediction_D": float(
                        prediction_mean_D[position]
                    ),
                    "random_only_ensemble_absolute_error_D": float(
                        ensemble_absolute_error_D[position]
                    ),
                    "ensemble_gain_D": float(ensemble_gain_D[position]),
                }
                for position in range(target.size)
            ]
            write_gzip_csv(
                output_run_dir / "per_molecule_sensitivity.csv.gz",
                per_molecule_rows,
            )
            write_json(output_run_dir / "run_summary.json", seed_row)

            print(
                f"{model_label:18s} seed {training_seed:4d} SUMMARY | "
                f"fixed {fixed_mae_D:.10f} | mean random "
                f"{mean_permutation_mae_D:.10f} | ensemble {ensemble_mae_D:.10f} D"
            )
            del model
            torch.cuda.empty_cache()

    require(canonical_target is not None, "No predictions were produced")
    require(len(seed_rows) == 12, "Expected 12 seed-level audit rows")
    require(len(permutation_rows) == 120, "Expected 120 permutation-level rows")

    write_csv(OUTPUT_DIR / "permutation_level_results.csv", permutation_rows)
    write_csv(OUTPUT_DIR / "seed_level_results.csv", seed_rows)

    model_rows: list[dict[str, Any]] = []
    model_metrics = (
        "fixed_mae_D",
        "mean_single_permutation_mae_D",
        "single_permutation_mae_sample_sd_D",
        "random_only_ensemble_mae_D",
        "mean_random_change_vs_fixed_D",
        "mean_random_change_vs_fixed_percent",
        "ensemble_change_vs_fixed_D",
        "ensemble_change_vs_fixed_percent",
        "mean_per_molecule_prediction_sd_D",
        "mean_per_molecule_absolute_shift_D",
    )
    for model_label in MODEL_SPECS:
        group = [row for row in seed_rows if row["model"] == model_label]
        require(len(group) == 3, f"Expected three training seeds for {model_label}")
        record: dict[str, Any] = {
            "model": model_label,
            "parameters": int(group[0]["parameters"]),
            "n_training_seeds": len(group),
            "n_test_molecules": int(group[0]["n_test_molecules"]),
            "n_permutations_per_seed": int(group[0]["n_permutations"]),
        }
        for metric in model_metrics:
            values = [float(row[metric]) for row in group]
            record[f"{metric}_mean_across_seeds"] = float(np.mean(values))
            record[f"{metric}_sample_sd_across_seeds"] = sample_sd(values)
        model_rows.append(record)
    write_csv(OUTPUT_DIR / "model_level_summary.csv", model_rows)

    created_files = sorted(
        path
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file() and path.name != "audit_manifest.json"
    )
    manifest = {
        "audit_version": protocol["audit_version"],
        "status": "QM9_MU_ORDERING_AUDIT_PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "integrity": {
            "all_12_fixed_maes_reproduced_within_tolerance": True,
            "all_targets_identical_across_fixed_permuted_and_model_runs": True,
            "shared_deterministic_permutations": True,
            "test_indices_sha256": sha256_array(test_indices),
            "target_sha256": sha256_array(canonical_target),
        },
        "model_level_results": model_rows,
        "files_sha256": {
            str(path.relative_to(OUTPUT_DIR)): sha256_file(path)
            for path in created_files
        },
    }
    write_json(OUTPUT_DIR / "audit_manifest.json", manifest)

    print("\n===== SEED-LEVEL ORDERING RESULTS =====")
    for row in seed_rows:
        print(
            f"{row['model']:18s} seed {int(row['training_seed']):4d} | "
            f"fixed {float(row['fixed_mae_D']):.10f} | "
            f"mean random {float(row['mean_single_permutation_mae_D']):.10f} | "
            f"ensemble {float(row['random_only_ensemble_mae_D']):.10f} D"
        )

    print("\n===== MODEL-LEVEL ORDERING RESULTS =====")
    for row in model_rows:
        print(
            f"{row['model']:18s} | fixed "
            f"{float(row['fixed_mae_D_mean_across_seeds']):.10f} +/- "
            f"{float(row['fixed_mae_D_sample_sd_across_seeds']):.10f} | "
            f"mean random "
            f"{float(row['mean_single_permutation_mae_D_mean_across_seeds']):.10f} "
            f"+/- {float(row['mean_single_permutation_mae_D_sample_sd_across_seeds']):.10f} | "
            f"ensemble "
            f"{float(row['random_only_ensemble_mae_D_mean_across_seeds']):.10f} "
            f"+/- {float(row['random_only_ensemble_mae_D_sample_sd_across_seeds']):.10f} D"
        )

    print("\n===== OUTPUT FILES =====")
    for name in (
        "audit_protocol.json",
        "permutation_level_results.csv",
        "seed_level_results.csv",
        "model_level_summary.csv",
        "audit_manifest.json",
    ):
        path = OUTPUT_DIR / name
        print(f"{name}: {path.stat().st_size} bytes")
    print("Per-run prediction packages: 12")
    print("\nQM9_MU_ORDERING_AUDIT_PASS")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nQM9_MU_ORDERING_AUDIT_FAIL: {exc}", file=sys.stderr)
        raise
