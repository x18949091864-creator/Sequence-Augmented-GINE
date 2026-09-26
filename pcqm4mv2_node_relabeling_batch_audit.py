#!/usr/bin/env python3
"""Batch node-relabeling audit for the PCQM4Mv2-50k controlled study.

The script audits the 15 locked formal checkpoints:
GINE-128, GINE-Wide, GINE-GRU, GINE-Transformer (Fixed-Index), and
GINE-Mamba (Fixed-Index), each with seeds 42, 123, and 3407.

Phase A is a one-molecule identity/sensitivity gate.  Only if every checkpoint
passes checkpoint identity, strict state loading, deterministic repeat, valid
graph relabeling, and the locked Mamba seed-42 output anchors does Phase B
evaluate O00 and P01-P10 on the complete 10,000-molecule controlled validation
subset.  Legacy tensor digests are retained as provenance diagnostics because
their original digest framing was not preserved; they are not used alone as a
hard data-identity decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from ogb.lsc import PygPCQM4Mv2Dataset
from torch import Tensor, nn
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


ATOL = 1.0e-6
RTOL = 1.0e-6
AUDIT_SCRIPT_VERSION = "2.3"
CONTROLLED_VALIDATION_SIZE = 10_000
PERMUTATION_SEEDS = tuple(range(42_001, 42_011))
PERMUTATION_STRIDE = 1_000_003
EXPECTED_EVALUATOR_SOURCES = {
    "gru": {
        "relative_path": (
            "models/dual_branch/SequenceComparisons/"
            "dualbranch_crossattn_index_gru_masked.py"
        ),
        "class": "DualBranchCrossAttnRandomGRUMasked",
        # No independently preserved pre-audit digest exists for this file.
        # Its actual digest is recorded, while behavioral compatibility is
        # enforced by strict state loading and complete O00 MAE reproduction.
        "sha256": None,
    },
    "transformer": {
        "relative_path": (
            "models/dual_branch/SequenceComparisons/"
            "dualbranch_crossattn_index_transformer_masked.py"
        ),
        "class": "DualBranchCrossAttnRandomTransformerMasked",
        # See the GRU note above.
        "sha256": None,
    },
    "mamba": {
        "relative_path": (
            "models/dual_branch/CrossAttn/"
            "dualbranch_crossattn_index_masked.py"
        ),
        "class": "DualBranchCrossAttnRandomMasked",
        "sha256": (
            "360b7361c8e7557dae9cc5010b4c3612bd61d10f7d84d40ddefea325f16d3ea9"
        ),
    },
}
EXPECTED_BASE_SOURCE_SHA256 = (
    "e196f214a008b79e911ec73ca33cb038bccbb46e6355a5be1bfec9725c4f91e4"
)

EXPECTED_OFFICIAL_VALIDATION_INDEX = 3_378_606
EXPECTED_SAMPLE = {
    "num_nodes": 17,
    "num_directed_edges": 34,
    "target": 4.587839603424072,
    "x_shape": (17, 9),
    "edge_index_shape": (2, 34),
    "edge_attr_shape": (34, 3),
    "y_shape": (1,),
    "x_dtype": "torch.int64",
    "edge_index_dtype": "torch.int64",
    "edge_attr_dtype": "torch.int64",
    "y_dtype": "torch.float32",
    # These three values were copied from the prior one-model audit.  That
    # audit did not preserve the exact digest framing (raw bytes versus bytes
    # plus dtype/shape/name), so v2 records comparisons to them as diagnostic
    # provenance only.  Exact representation compatibility is instead locked
    # below by four independent Mamba seed-42 output anchors and then by each
    # checkpoint's complete O00 validation-MAE reproduction before any
    # permuted full-validation result is accepted.
    "legacy_x_sha256": "1aeb9c2f16f7487b1d266cdd6cf2fefbdccafaed5b9ce7514a7eb5a40f6317e7",
    "legacy_edge_index_sha256": "97f64d6c0d4db56e4ea998f5e4537286b3425bb6afe3786bb515967799d2c206",
    "legacy_edge_attr_sha256": "c0d4f8ab6e2adaa273704bf78decfe1d75b6ed9873cd63dc2fea05f6e2008650",
    # The previous audit reported this composite digest.  Its exact framing
    # was script-specific, so identity is enforced below using the three
    # component tensor digests rather than assuming the same framing.
    "prior_audit_graph_sha256": "8c8551d0cbc49f03a2ba263705e93f7ee3167f26f3e96292d9ac35b665d3ed26",
}

EXPECTED_MAMBA_SEED42_ANCHOR = {
    "o00_prediction": 5.311175346375,
    "mean_absolute_delta": 8.096523284912e-02,
    "maximum_absolute_delta": 1.741809844971e-01,
    "num_permutations_outside_tolerance": 10,
}

EXPECTED_FIRST_SAMPLE_PERMUTATIONS = (
    (12, 7, 15, 1, 6, 5, 3, 11, 0, 10, 2, 14, 4, 8, 13, 16, 9),
    (5, 12, 2, 16, 1, 9, 15, 11, 0, 6, 3, 7, 14, 13, 8, 10, 4),
    (5, 10, 16, 3, 12, 1, 11, 13, 0, 9, 4, 8, 7, 2, 15, 14, 6),
    (4, 8, 15, 14, 11, 13, 1, 5, 9, 3, 7, 0, 16, 10, 12, 6, 2),
    (10, 2, 11, 3, 4, 6, 14, 5, 16, 15, 7, 1, 9, 12, 0, 8, 13),
    (16, 0, 7, 4, 12, 3, 11, 6, 1, 5, 10, 15, 9, 2, 13, 8, 14),
    (6, 10, 14, 1, 2, 7, 16, 11, 13, 12, 4, 0, 5, 8, 15, 3, 9),
    (12, 14, 3, 9, 10, 16, 0, 6, 2, 4, 11, 7, 15, 13, 8, 1, 5),
    (7, 4, 1, 12, 16, 3, 9, 8, 13, 11, 10, 6, 0, 2, 14, 15, 5),
    (14, 6, 10, 4, 7, 5, 12, 8, 0, 11, 15, 1, 2, 16, 9, 3, 13),
)


@dataclass(frozen=True)
class CheckpointSpec:
    model_label: str
    family: str
    seed: int
    relative_path: str
    sha256: str
    expected_best_val_mae: float
    expected_hidden_dim: int
    expected_trainable_parameters: int
    expected_state_tensor_numel: int
    evaluation_source: str

    @property
    def run_id(self) -> str:
        return f"{self.model_label}_seed{self.seed}"


CHECKPOINT_SPECS = (
    CheckpointSpec(
        "GINE-128", "gine", 42,
        "baseline/gine_only_50k_20260722_003155/checkpoints/best_model.pt",
        "3c9e30d32e6f31a8974dc0e5dc07f963389e15d980d05913d0cf5bfc41ae295d",
        0.2913091513156891, 128, 173_697, 174_729, "native_gine",
    ),
    CheckpointSpec(
        "GINE-128", "gine", 123,
        "baseline/gine_only_50k_seed123_20260722_015856/checkpoints/best_model.pt",
        "34db9a82ce2b4d6e5a6c17aef21b83286d2d31e4c70133ed508597723e5daf24",
        0.28209833670854567, 128, 173_697, 174_729, "native_gine",
    ),
    CheckpointSpec(
        "GINE-128", "gine", 3407,
        "baseline/gine_only_50k_seed3407_20260722_041829/checkpoints/best_model.pt",
        "b3099a49f90082b80e5a6c32b4ffaacd3c3e8b3c11871a149adedd45098b3c34",
        0.2814564664721489, 128, 173_697, 174_729, "native_gine",
    ),
    CheckpointSpec(
        "GINE-Wide", "gine", 42,
        "baseline_wide/gine_wide_50k_seed42_20260810_152822/checkpoints/best_model.pt",
        "45da47106685c7e33871c03436484e033b59dd745f0698272f4c75f218cf94bd",
        0.2622765230059624, 192, 371_137, 372_681, "native_gine",
    ),
    CheckpointSpec(
        "GINE-Wide", "gine", 123,
        "baseline_wide/gine_wide_50k_seed123_20260810_160515/checkpoints/best_model.pt",
        "2c775a14d7709777414740f5736a318ac26219695ae1086cc44b56dacfbe26a4",
        0.2568030346035957, 192, 371_137, 372_681, "native_gine",
    ),
    CheckpointSpec(
        "GINE-Wide", "gine", 3407,
        "baseline_wide/gine_wide_50k_seed3407_20260810_162359/checkpoints/best_model.pt",
        "1a400324cfd6ef168bd6a94091fd176dcfae29b487c40602f3ad304994bdb9a7",
        0.26565197080373765, 192, 371_137, 372_681, "native_gine",
    ),
    CheckpointSpec(
        "GINE-GRU", "gru", 42,
        "matched_gru/matched_gru_50k_20260724_164204/checkpoints/best_model.pt",
        "30ad6d4d92e0ef1dc2e2b54e2333cadb9b580a0c34d226e1c496e571e3e291cd",
        0.2233241686820984, 128, 355_201, 356_233, "fixed_index_evaluator",
    ),
    CheckpointSpec(
        "GINE-GRU", "gru", 123,
        "matched_gru/matched_gru_50k_seed123_20260724_172605/checkpoints/best_model.pt",
        "5a55cd7ece5e7f231724087eb385af68faa49d7d0579816f67ac0c13dc21852f",
        0.23075993573665618, 128, 355_201, 356_233, "fixed_index_evaluator",
    ),
    CheckpointSpec(
        "GINE-GRU", "gru", 3407,
        "matched_gru/matched_gru_50k_seed3407_20260724_180740/checkpoints/best_model.pt",
        "ef98b5fc45a17ca92e625c24a729ea40aa1a0afa0d004a2e6ec03af2d0d9df70",
        0.23477622945308685, 128, 355_201, 356_233, "fixed_index_evaluator",
    ),
    CheckpointSpec(
        "GINE-Transformer", "transformer", 42,
        "matched_transformer/matched_transformer_index_50k_seed42_20260722_080531/checkpoints/best_model.pt",
        "20a6062a60af37540cf62475649ce36586ebaa3fa77b1cbf207b5de521bdabf3",
        0.2382601622223854, 128, 388_609, 389_641, "fixed_index_evaluator",
    ),
    CheckpointSpec(
        "GINE-Transformer", "transformer", 123,
        "matched_transformer/matched_transformer_index_50k_seed123_20260723_115001/checkpoints/best_model.pt",
        "437428dd140541bbe1f79689c7433f0c2bdc6fb0caccd0804a634fbedacba7b6",
        0.23521494839191437, 128, 388_609, 389_641, "fixed_index_evaluator",
    ),
    CheckpointSpec(
        "GINE-Transformer", "transformer", 3407,
        "matched_transformer/matched_transformer_index_50k_seed3407_20260723_122649/checkpoints/best_model.pt",
        "62484443a15e4760f56913d22c490dd652b7145019938222d29030727a08ad63",
        0.23520422633886337, 128, 388_609, 389_641, "fixed_index_evaluator",
    ),
    CheckpointSpec(
        "GINE-Mamba", "mamba", 42,
        "masked_mamba/masked_mamba_index_50k_seed42_20260722_061035/checkpoints/best_model.pt",
        "ea6f491c350d5a3f360ef48da1b6a53eb56909190e3071823ae8967cc4a0fd6e",
        0.23216088560819625, 128, 372_609, 373_641, "fixed_index_evaluator",
    ),
    CheckpointSpec(
        "GINE-Mamba", "mamba", 123,
        "masked_mamba/masked_mamba_index_50k_seed123_20260722_064617/checkpoints/best_model.pt",
        "63f3accbf0fbaf434e872cc517c5867a85a353c1dd1167591df545a9da67b32e",
        0.23612797057628632, 128, 372_609, 373_641, "fixed_index_evaluator",
    ),
    CheckpointSpec(
        "GINE-Mamba", "mamba", 3407,
        "masked_mamba/masked_mamba_index_50k_seed3407_20260722_072143/checkpoints/best_model.pt",
        "1f5e5fd7a1e3d6e1f911544d7a3c91905bc83ddec2935d3ca91ce9f73679db49",
        0.23133581598997116, 128, 372_609, 373_641, "fixed_index_evaluator",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--validation-size", type=int, default=CONTROLLED_VALIDATION_SIZE
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Run only Phase A. The default also runs the full 10k Phase B.",
    )
    parser.add_argument(
        "--run-id",
        choices=tuple(spec.run_id for spec in CHECKPOINT_SPECS),
        default=None,
        help=(
            "Audit exactly one locked checkpoint in an isolated process. "
            "Omit this option to retain the original 15-checkpoint batch mode."
        ),
    )
    parser.add_argument(
        "--mae-match-tolerance",
        type=float,
        default=2.0e-5,
        help="Allowed absolute difference between recomputed O00 MAE and checkpoint metadata.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def tensor_inventory(tensor: Tensor) -> dict[str, Any]:
    """Return auditable metadata without changing the tensor or its dtype."""
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "minimum": value.min().item() if value.numel() else None,
        "maximum": value.max().item() if value.numel() else None,
        "raw_bytes_sha256": tensor_sha256(value),
    }


def graph_sha256(data: Data) -> str:
    digest = hashlib.sha256()
    for name in ("x", "edge_index", "edge_attr", "y"):
        value = getattr(data, name, None)
        if torch.is_tensor(value):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: json_ready(row.get(key)) for key in fieldnames} for row in rows])
    temporary.replace(path)


def checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor]:
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint has no model_state_dict/state_dict mapping")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def config_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {}


def checkpoint_best_metric(checkpoint: Mapping[str, Any]) -> float:
    for key in ("best_val_mae", "best_valid_mae", "val_mae", "best_metric"):
        if key in checkpoint:
            return float(checkpoint[key])
    raise KeyError("Checkpoint contains no recognized best-validation metric")


def construct_permutation(num_nodes: int, permutation_number: int, local_index: int) -> Tensor:
    seed = PERMUTATION_SEEDS[permutation_number - 1] + local_index * PERMUTATION_STRIDE
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(num_nodes, generator=generator)


def relabel_graph(data: Data, new_to_old: Tensor) -> tuple[Data, Tensor]:
    num_nodes = int(data.num_nodes)
    new_to_old = new_to_old.detach().cpu().to(torch.long)
    if tuple(new_to_old.shape) != (num_nodes,):
        raise ValueError("Permutation shape does not match graph node count")
    if not torch.equal(torch.sort(new_to_old).values, torch.arange(num_nodes)):
        raise ValueError("new_to_old is not a valid permutation")

    old_to_new = torch.empty_like(new_to_old)
    old_to_new[new_to_old] = torch.arange(num_nodes)

    relabeled = data.clone()
    relabeled.x = data.x[new_to_old]
    relabeled.edge_index = old_to_new[data.edge_index.detach().cpu()].to(data.edge_index.device)
    relabeled.num_nodes = num_nodes
    return relabeled, old_to_new


def relabeling_recovers_original(original: Data, relabeled: Data, old_to_new: Tensor, new_to_old: Tensor) -> bool:
    recovered_x = relabeled.x[old_to_new.to(relabeled.x.device)]
    recovered_edge_index = new_to_old.to(relabeled.edge_index.device)[relabeled.edge_index]
    return (
        torch.equal(recovered_x, original.x)
        and torch.equal(recovered_edge_index, original.edge_index)
        and torch.equal(relabeled.edge_attr, original.edge_attr)
        and torch.equal(relabeled.y, original.y)
    )


class ValidationView(Dataset):
    def __init__(
        self,
        dataset: PygPCQM4Mv2Dataset,
        official_indices: Sequence[int],
        permutation_number: int,
    ) -> None:
        self.dataset = dataset
        self.official_indices = official_indices
        self.permutation_number = permutation_number

    def __len__(self) -> int:
        return len(self.official_indices)

    def __getitem__(self, local_index: int) -> Data:
        official_index = int(self.official_indices[local_index])
        graph = self.dataset[official_index]
        if self.permutation_number:
            new_to_old = construct_permutation(
                int(graph.num_nodes), self.permutation_number, local_index
            )
            graph, _ = relabel_graph(graph, new_to_old)
        return graph


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_cudnn_for_spec(
    spec: CheckpointSpec, default_enabled: bool
) -> dict[str, Any]:
    """Apply the evaluator runtime policy used by the locked model family.

    The GRU comparison was trained with cuDNN disabled.  Keeping it disabled
    for both device migration and inference also avoids invoking cuDNN's RNN
    weight-flattening path, which is not part of the locked GRU runtime.
    Other families retain the process's initial cuDNN setting.
    """
    enabled = False if spec.family == "gru" else bool(default_enabled)
    torch.backends.cudnn.enabled = enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    policy = {
        "family": spec.family,
        "enabled": bool(torch.backends.cudnn.enabled),
        "benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic": bool(torch.backends.cudnn.deterministic),
        "reason": (
            "match_locked_gru_training_runtime"
            if spec.family == "gru"
            else "retain_process_initial_setting"
        ),
    }
    print(
        "CUDNN_RUNTIME_POLICY: "
        f"family={policy['family']} enabled={policy['enabled']} "
        f"benchmark={policy['benchmark']} "
        f"deterministic={policy['deterministic']} "
        f"reason={policy['reason']}",
        flush=True,
    )
    return policy


def state_tensor_numel(state: Mapping[str, Tensor]) -> int:
    return sum(value.numel() for value in state.values() if torch.is_tensor(value))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def import_module_from_path(project_root: Path, source_path: Path):
    relative = source_path.resolve().relative_to(project_root.resolve()).with_suffix("")
    module_name = ".".join(relative.parts)
    return importlib.import_module(module_name)


def source_priority(path: Path, family: str) -> tuple[int, str]:
    text = str(path).lower()
    if family == "gine":
        score = 0
        if "gine_only" in text or "gineonly" in text:
            score -= 100
        if "baseline" in text:
            score -= 40
        if "dual" in text or "mamba" in text or "transformer" in text or "gru" in text:
            score += 100
        return score, text
    if path.name == "dualbranch_crossattn_index_masked.py":
        return -1000, text
    if "index_masked" in path.name:
        return -900, text
    if "crossattn" in text:
        return -100, text
    return 0, text


def candidate_source_paths(project_root: Path, family: str) -> list[Path]:
    models_root = project_root / "models"
    if not models_root.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_root}")

    if family != "gine":
        if family not in EXPECTED_EVALUATOR_SOURCES:
            raise ValueError(f"Unsupported sequence-model family: {family}")
        locked = project_root / EXPECTED_EVALUATOR_SOURCES[family]["relative_path"]
        if not locked.is_file():
            raise FileNotFoundError(
                f"Locked {family} Fixed-Index evaluator source is missing: {locked}"
            )
        return [locked]

    paths = [
        path
        for path in models_root.rglob("*.py")
        if "gine" in str(path).lower() and path.name != "__init__.py"
    ]
    return sorted(paths, key=lambda path: source_priority(path, family))


def constructor_attempts(cls: type[nn.Module], config: Mapping[str, Any]) -> Iterable[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    signature = inspect.signature(cls.__init__)
    parameters = [
        parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
    ]
    aliases: dict[str, Any] = {
        "hidden_channels": config.get("hidden_dim"),
        "emb_dim": config.get("hidden_dim"),
        "embedding_dim": config.get("hidden_dim"),
        "gnn_hidden_dim": config.get("hidden_dim"),
        "n_layers": config.get("num_layers"),
        "num_gnn_layers": config.get("num_layers"),
        "gnn_num_layers": config.get("num_layers"),
        "heads": config.get("num_heads"),
        "attention_heads": config.get("num_heads"),
        "out_dim": 1,
        "output_dim": 1,
        "num_tasks": 1,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    }

    kwargs: dict[str, Any] = {}
    unresolved = []
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
    for parameter in parameters:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.name in config:
            kwargs[parameter.name] = config[parameter.name]
        elif parameter.name in aliases and aliases[parameter.name] is not None:
            kwargs[parameter.name] = aliases[parameter.name]
        elif parameter.default is inspect.Parameter.empty:
            unresolved.append(parameter.name)

    if not unresolved:
        if has_var_keyword:
            merged = dict(config)
            merged.update(kwargs)
            yield "filtered_config_plus_kwargs", (), merged
        yield "filtered_config", (), kwargs

    namespace = SimpleNamespace(**config)
    for parameter_name in ("config", "args", "cfg"):
        if len(parameters) == 1 and parameters[0].name == parameter_name:
            yield "namespace", (namespace,), {}
            yield "mapping", (dict(config),), {}


def resolve_model(
    project_root: Path,
    spec: CheckpointSpec,
    checkpoint: Mapping[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    state = checkpoint_state_dict(checkpoint)
    config = config_dict(checkpoint)
    errors: list[str] = []

    state_keys = tuple(str(key) for key in state)
    family_markers = {
        "gru": ("mamba.gru.",),
        "transformer": ("mamba.encoder.layers.",),
        "mamba": ("mamba.A_log", "mamba.in_proj."),
    }
    if spec.family in family_markers:
        markers = family_markers[spec.family]
        if not any(
            key == marker or key.startswith(marker)
            for key in state_keys
            for marker in markers
        ):
            raise RuntimeError(
                f"Checkpoint state for {spec.run_id} has no "
                f"{spec.family} family marker among {markers}"
            )

    for source_path in candidate_source_paths(project_root, spec.family):
        try:
            module = import_module_from_path(project_root, source_path)
        except Exception as error:
            errors.append(f"IMPORT {source_path}: {type(error).__name__}: {error}")
            continue

        classes = []
        expected_class = (
            EXPECTED_EVALUATOR_SOURCES[spec.family]["class"]
            if spec.family != "gine"
            else None
        )
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, nn.Module)
                and obj is not nn.Module
                and obj.__module__ == module.__name__
                and (expected_class is None or obj.__name__ == expected_class)
            ):
                classes.append(obj)

        if expected_class is not None and not classes:
            errors.append(
                f"CLASS {source_path}: expected {expected_class} was not defined"
            )
            continue

        classes.sort(key=lambda cls: (0 if "gine" in cls.__name__.lower() else 1, cls.__name__))
        for cls in classes:
            for attempt_name, args, kwargs in constructor_attempts(cls, config):
                try:
                    model = cls(*args, **kwargs)
                    incompatible = model.load_state_dict(state, strict=True)
                    if incompatible.missing_keys or incompatible.unexpected_keys:
                        raise RuntimeError(str(incompatible))
                    if parameter_count(model) != spec.expected_trainable_parameters:
                        raise RuntimeError(
                            f"trainable parameter count {parameter_count(model)} != "
                            f"{spec.expected_trainable_parameters}"
                        )
                    metadata = {
                        "source_path": str(source_path.resolve()),
                        "source_relative_path": str(
                            source_path.resolve().relative_to(project_root.resolve())
                        ),
                        "source_sha256": sha256_file(source_path),
                        "module": module.__name__,
                        "class": cls.__name__,
                        "constructor_attempt": attempt_name,
                        "trainable_parameters": parameter_count(model),
                    }
                    if spec.family != "gine":
                        base_source = (
                            project_root
                            / "models"
                            / "dual_branch"
                            / "CrossAttn"
                            / "dualbranch_crossattn_base_masked.py"
                        )
                        if not base_source.is_file():
                            raise FileNotFoundError(base_source)
                        metadata["base_source_path"] = str(base_source.resolve())
                        metadata["base_source_sha256"] = sha256_file(base_source)
                    return model, metadata
                except Exception as error:
                    errors.append(
                        f"BUILD {module.__name__}.{cls.__name__}/{attempt_name}: "
                        f"{type(error).__name__}: {str(error)[:400]}"
                    )

    diagnostic = "\n".join(errors[-30:])
    raise RuntimeError(
        f"No source class strictly reconstructed {spec.run_id}.\n"
        f"Last candidate diagnostics:\n{diagnostic}"
    )


def model_forward(model: nn.Module, batch: Batch) -> Tensor:
    signature = inspect.signature(model.forward)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name != "self"
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if len(parameters) == 1:
        output = model(batch)
    else:
        available = {
            "data": batch,
            "batch_data": batch,
            "x": batch.x,
            "edge_index": batch.edge_index,
            "edge_attr": batch.edge_attr,
            "batch": batch.batch,
        }
        kwargs: dict[str, Any] = {}
        unresolved = []
        for parameter in parameters:
            if parameter.name in available:
                kwargs[parameter.name] = available[parameter.name]
            elif parameter.default is inspect.Parameter.empty:
                unresolved.append(parameter.name)
        if unresolved:
            raise TypeError(
                f"Unsupported forward signature for {type(model).__name__}: "
                f"unresolved {unresolved}"
            )
        output = model(**kwargs)

    if isinstance(output, Mapping):
        for key in ("prediction", "pred", "out", "y_pred"):
            if key in output:
                output = output[key]
                break
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Model output is not a tensor: {type(output)}")
    return output.reshape(-1)


def extract_first_tensor(value: Any) -> Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            result = extract_first_tensor(item)
            if result is not None:
                return result
    if isinstance(value, Mapping):
        for item in value.values():
            result = extract_first_tensor(item)
            if result is not None:
                return result
    return None


def sequence_module(model: nn.Module, family: str) -> tuple[str, nn.Module] | None:
    if family == "gine":
        return None
    tokens = {
        "gru": ("gru",),
        "transformer": ("transformer",),
        "mamba": ("mamba",),
    }[family]
    candidates = []
    for name, module in model.named_modules():
        if not name:
            continue
        identity = f"{name} {type(module).__name__}".lower()
        if any(token in identity for token in tokens):
            candidates.append((name.count("."), len(name), name, module))
    if not candidates:
        return None
    _, _, name, module = sorted(candidates)[0]
    return name, module


def find_named_module(model: nn.Module, token: str) -> tuple[str, nn.Module] | None:
    candidates = []
    for name, module in model.named_modules():
        if token in name.lower():
            candidates.append((name.count("."), len(name), name, module))
    if not candidates:
        return None
    _, _, name, module = sorted(candidates)[0]
    return name, module


def traced_single_forward(
    model: nn.Module,
    graph: Data,
    device: torch.device,
    family: str,
) -> tuple[Tensor, dict[str, Tensor], dict[str, str]]:
    trace: dict[str, Tensor] = {}
    trace_modules: dict[str, str] = {}
    hooks = []

    seq = sequence_module(model, family)
    if seq is not None:
        seq_name, seq_layer = seq
        trace_modules["sequence"] = seq_name

        def seq_pre_hook(_module, inputs):
            tensor = extract_first_tensor(inputs)
            if tensor is not None:
                trace["sequence_input"] = tensor.detach().cpu().clone()

        def seq_post_hook(_module, _inputs, output):
            tensor = extract_first_tensor(output)
            if tensor is not None:
                trace["sequence_output"] = tensor.detach().cpu().clone()

        hooks.append(seq_layer.register_forward_pre_hook(seq_pre_hook))
        hooks.append(seq_layer.register_forward_hook(seq_post_hook))

    attention = find_named_module(model, "cross_attn")
    if attention is not None:
        attention_name, attention_layer = attention
        trace_modules["cross_attention"] = attention_name

        def attention_pre_hook(_module, inputs):
            if len(inputs) >= 1 and torch.is_tensor(inputs[0]):
                trace["graph_query"] = inputs[0].detach().cpu().clone()
            if len(inputs) >= 2 and torch.is_tensor(inputs[1]):
                trace["sequence_key"] = inputs[1].detach().cpu().clone()

        hooks.append(attention_layer.register_forward_pre_hook(attention_pre_hook))

    head = find_named_module(model, "pred_head")
    if head is not None:
        head_name, head_layer = head
        trace_modules["prediction_head"] = head_name

        def head_pre_hook(_module, inputs):
            tensor = extract_first_tensor(inputs)
            if tensor is not None:
                trace["prediction_head_input"] = tensor.detach().cpu().clone()

        hooks.append(head_layer.register_forward_pre_hook(head_pre_hook))

    try:
        batch = Batch.from_data_list([graph]).to(device)
        with torch.inference_mode():
            prediction = model_forward(model, batch).detach().cpu()
    finally:
        for hook in hooks:
            hook.remove()

    if prediction.numel() != 1:
        raise ValueError(f"Expected one prediction, received shape {tuple(prediction.shape)}")
    return prediction.reshape(()), trace, trace_modules


def max_abs_delta(first: Tensor | None, second: Tensor | None) -> float | None:
    if first is None or second is None or tuple(first.shape) != tuple(second.shape):
        return None
    return float((first.to(torch.float64) - second.to(torch.float64)).abs().max().item())


def align_node_tensor(tensor: Tensor | None, old_to_new: Tensor, num_nodes: int) -> Tensor | None:
    if tensor is None:
        return None
    index = old_to_new.to(torch.long)
    if tensor.ndim >= 2 and tensor.shape[1] == num_nodes:
        return tensor.index_select(1, index)
    if tensor.ndim >= 1 and tensor.shape[0] == num_nodes:
        return tensor.index_select(0, index)
    return None


def validate_checkpoint_identity(
    spec: CheckpointSpec,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    actual_sha = sha256_file(checkpoint_path)
    config = config_dict(checkpoint)
    state = checkpoint_state_dict(checkpoint)
    actual_metric = checkpoint_best_metric(checkpoint)
    checks = {
        "sha256_match": actual_sha == spec.sha256,
        "seed_match": int(config.get("seed")) == spec.seed,
        "train_size_match": int(config.get("train_size")) == 50_000,
        "val_size_match": int(config.get("val_size")) == 10_000,
        "hidden_dim_match": int(config.get("hidden_dim")) == spec.expected_hidden_dim,
        "best_metric_match": math.isclose(
            actual_metric, spec.expected_best_val_mae, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "state_tensor_numel_match": state_tensor_numel(state) == spec.expected_state_tensor_numel,
    }
    return {
        "checkpoint_path": str(checkpoint_path),
        "actual_sha256": actual_sha,
        "expected_sha256": spec.sha256,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_mae": actual_metric,
        "experiment_name": config.get("experiment_name"),
        "sequence_model": config.get("sequence_model"),
        "state_tensor_numel": state_tensor_numel(state),
        "checks": checks,
        "pass": all(checks.values()),
    }


def run_phase_a(
    spec: CheckpointSpec,
    model: nn.Module,
    base_graph: Data,
    device: torch.device,
    identity: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    original_digest = graph_sha256(base_graph)
    o00_a, o00_a_trace, trace_modules = traced_single_forward(
        model, base_graph, device, spec.family
    )
    o00_b, o00_b_trace, _ = traced_single_forward(
        model, base_graph, device, spec.family
    )
    repeat_delta = abs(float(o00_a.item()) - float(o00_b.item()))
    repeat_close = bool(torch.isclose(o00_a, o00_b, atol=ATOL, rtol=RTOL).item())

    permutation_rows = []
    all_recover = True
    for permutation_number in range(1, 11):
        new_to_old = construct_permutation(
            int(base_graph.num_nodes), permutation_number, local_index=0
        )
        expected = torch.tensor(
            EXPECTED_FIRST_SAMPLE_PERMUTATIONS[permutation_number - 1], dtype=torch.long
        )
        if not torch.equal(new_to_old, expected):
            raise AssertionError(f"P{permutation_number:02d} does not match the locked permutation")

        relabeled, old_to_new = relabel_graph(base_graph, new_to_old)
        recovered = relabeling_recovers_original(
            base_graph, relabeled, old_to_new, new_to_old
        )
        all_recover = all_recover and recovered
        prediction, trace, _ = traced_single_forward(
            model, relabeled, device, spec.family
        )
        absolute_delta = abs(float(prediction.item()) - float(o00_a.item()))

        aligned_sequence_input = align_node_tensor(
            trace.get("sequence_input"), old_to_new, int(base_graph.num_nodes)
        )
        aligned_sequence_output = align_node_tensor(
            trace.get("sequence_output"), old_to_new, int(base_graph.num_nodes)
        )
        aligned_sequence_key = align_node_tensor(
            trace.get("sequence_key"), old_to_new, int(base_graph.num_nodes)
        )
        permutation_rows.append(
            {
                "permutation": f"P{permutation_number:02d}",
                "permutation_seed": PERMUTATION_SEEDS[permutation_number - 1],
                "new_to_old": new_to_old.tolist(),
                "prediction": float(prediction.item()),
                "absolute_delta_from_o00": absolute_delta,
                "bitwise_equal_to_o00": bool(torch.equal(prediction, o00_a)),
                "numerically_close_to_o00": bool(
                    torch.isclose(prediction, o00_a, atol=ATOL, rtol=RTOL).item()
                ),
                "relabeling_recovers_original": recovered,
                "sequence_input_direct_max_delta": max_abs_delta(
                    trace.get("sequence_input"), o00_a_trace.get("sequence_input")
                ),
                "sequence_input_aligned_max_delta": max_abs_delta(
                    aligned_sequence_input, o00_a_trace.get("sequence_input")
                ),
                "sequence_output_direct_max_delta": max_abs_delta(
                    trace.get("sequence_output"), o00_a_trace.get("sequence_output")
                ),
                "sequence_output_aligned_max_delta": max_abs_delta(
                    aligned_sequence_output, o00_a_trace.get("sequence_output")
                ),
                "graph_query_max_delta": max_abs_delta(
                    trace.get("graph_query"), o00_a_trace.get("graph_query")
                ),
                "sequence_key_aligned_max_delta": max_abs_delta(
                    aligned_sequence_key, o00_a_trace.get("sequence_key")
                ),
                "prediction_head_input_max_delta": max_abs_delta(
                    trace.get("prediction_head_input"),
                    o00_a_trace.get("prediction_head_input"),
                ),
            }
        )

    deltas = [row["absolute_delta_from_o00"] for row in permutation_rows]
    outside = [row for row in permutation_rows if not row["numerically_close_to_o00"]]
    original_not_mutated = graph_sha256(base_graph) == original_digest
    gate_checks = {
        "checkpoint_identity": bool(identity["pass"]),
        "parameter_count_reconciliation": parameter_count(model)
        == spec.expected_trainable_parameters,
        "repeat_numerically_close": repeat_close,
        "original_graph_not_mutated": original_not_mutated,
        "all_relabelings_recover_original": all_recover,
        "expected_permutations_match": True,
    }
    if spec.family != "gine":
        evaluator_spec = EXPECTED_EVALUATOR_SOURCES[spec.family]
        gate_checks["fixed_index_evaluator_source_path_identity"] = (
            source_metadata.get("source_relative_path")
            == evaluator_spec["relative_path"]
        )
        expected_source_sha256 = evaluator_spec["sha256"]
        if expected_source_sha256 is not None:
            gate_checks["fixed_index_evaluator_source_hash_identity"] = (
                source_metadata.get("source_sha256") == expected_source_sha256
            )
        else:
            # GRU/Transformer did not have an independently preserved source
            # digest before this audit.  Record the current digest in metadata;
            # do not misrepresent a newly observed value as a prior identity
            # anchor.  Strict state loading plus Phase-B O00 MAE reproduction
            # provides the hard compatibility gate for these two families.
            source_digest = source_metadata.get("source_sha256")
            gate_checks["fixed_index_evaluator_source_hash_recorded"] = bool(
                isinstance(source_digest, str) and len(source_digest) == 64
            )
        gate_checks["base_model_source_identity"] = (
            source_metadata.get("base_source_sha256")
            == EXPECTED_BASE_SOURCE_SHA256
        )
    if spec.model_label == "GINE-Mamba" and spec.seed == 42:
        gate_checks["prior_mamba_seed42_o00_anchor"] = math.isclose(
            float(o00_a.item()),
            EXPECTED_MAMBA_SEED42_ANCHOR["o00_prediction"],
            rel_tol=0.0,
            abs_tol=ATOL,
        )
        gate_checks["prior_mamba_seed42_mean_drift_anchor"] = math.isclose(
            statistics.mean(deltas),
            EXPECTED_MAMBA_SEED42_ANCHOR["mean_absolute_delta"],
            rel_tol=0.0,
            abs_tol=ATOL,
        )
        gate_checks["prior_mamba_seed42_max_drift_anchor"] = math.isclose(
            max(deltas),
            EXPECTED_MAMBA_SEED42_ANCHOR["maximum_absolute_delta"],
            rel_tol=0.0,
            abs_tol=ATOL,
        )
        gate_checks["prior_mamba_seed42_outside_count_anchor"] = (
            len(outside)
            == EXPECTED_MAMBA_SEED42_ANCHOR["num_permutations_outside_tolerance"]
        )
    return {
        "run_id": spec.run_id,
        "model_label": spec.model_label,
        "family": spec.family,
        "seed": spec.seed,
        "checkpoint_identity": identity,
        "source": source_metadata,
        "trace_modules": trace_modules,
        "o00_a_prediction": float(o00_a.item()),
        "o00_b_prediction": float(o00_b.item()),
        "o00_repeat_absolute_delta": repeat_delta,
        "o00_repeat_bitwise_equal": bool(torch.equal(o00_a, o00_b)),
        "o00_repeat_numerically_close": repeat_close,
        "o00_sequence_input_repeat_max_delta": max_abs_delta(
            o00_a_trace.get("sequence_input"), o00_b_trace.get("sequence_input")
        ),
        "o00_graph_query_repeat_max_delta": max_abs_delta(
            o00_a_trace.get("graph_query"), o00_b_trace.get("graph_query")
        ),
        "permutations": permutation_rows,
        "sensitivity_summary": {
            "maximum_absolute_delta": max(deltas),
            "mean_absolute_delta": statistics.mean(deltas),
            "median_absolute_delta": statistics.median(deltas),
            "permuted_prediction_range": max(
                row["prediction"] for row in permutation_rows
            )
            - min(row["prediction"] for row in permutation_rows),
            "permuted_prediction_population_std": statistics.pstdev(
                row["prediction"] for row in permutation_rows
            ),
            "num_permutations_outside_tolerance": len(outside),
            "outside_tolerance_labels": [row["permutation"] for row in outside],
            "node_relabeling_sensitivity_detected": bool(outside),
        },
        "gate_checks": gate_checks,
        "audit_status": "PASS" if all(gate_checks.values()) else "FAIL",
    }


def evaluate_validation_order(
    model: nn.Module,
    dataset: PygPCQM4Mv2Dataset,
    official_indices: Sequence[int],
    permutation_number: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    view = ValidationView(dataset, official_indices, permutation_number)
    loader = DataLoader(
        view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    predictions = []
    targets = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            prediction = model_forward(model, batch)
            target = batch.y.reshape(-1)
            if prediction.numel() != target.numel():
                raise ValueError(
                    f"Prediction/target size mismatch: {prediction.numel()} vs {target.numel()}"
                )
            predictions.append(prediction.detach().cpu().to(torch.float64))
            targets.append(target.detach().cpu().to(torch.float64))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    prediction_array = torch.cat(predictions).numpy()
    target_array = torch.cat(targets).numpy()
    runtime = {
        "elapsed_seconds": elapsed,
        "samples_per_second": len(view) / elapsed,
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else None
        ),
    }
    return prediction_array, target_array, runtime


def drift_summary(original: np.ndarray, permuted: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(permuted - original)
    threshold = ATOL + RTOL * np.abs(original)
    outside = absolute > threshold
    return {
        "mean_absolute_prediction_drift": float(np.mean(absolute)),
        "median_absolute_prediction_drift": float(np.median(absolute)),
        "p95_absolute_prediction_drift": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_prediction_drift": float(np.max(absolute)),
        "prediction_drift_population_std": float(np.std(permuted - original)),
        "fraction_molecules_outside_tolerance": float(np.mean(outside)),
        "num_molecules_outside_tolerance": int(np.sum(outside)),
    }


def run_phase_b(
    spec: CheckpointSpec,
    model: nn.Module,
    dataset: PygPCQM4Mv2Dataset,
    official_indices: Sequence[int],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"\n===== PHASE B FULL VALIDATION: {spec.run_id} =====", flush=True)
    predictions_by_order = []
    targets_reference = None
    order_rows = []

    for permutation_number in range(0, 11):
        label = "O00" if permutation_number == 0 else f"P{permutation_number:02d}"
        predictions, targets, runtime = evaluate_validation_order(
            model=model,
            dataset=dataset,
            official_indices=official_indices,
            permutation_number=permutation_number,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if targets_reference is None:
            targets_reference = targets
        elif not np.array_equal(targets, targets_reference):
            raise AssertionError(f"Target vector changed under {label}")

        predictions_by_order.append(predictions)
        mae = float(np.mean(np.abs(predictions - targets)))
        row = {
            "run_id": spec.run_id,
            "model_label": spec.model_label,
            "family": spec.family,
            "seed": spec.seed,
            "order": label,
            "permutation_seed": (
                None if permutation_number == 0 else PERMUTATION_SEEDS[permutation_number - 1]
            ),
            "validation_mae": mae,
            "delta_mae_from_o00": None,
            **runtime,
        }

        if permutation_number == 0:
            metadata_delta = abs(mae - spec.expected_best_val_mae)
            row["checkpoint_metadata_mae"] = spec.expected_best_val_mae
            row["absolute_delta_from_checkpoint_metadata"] = metadata_delta
            row["checkpoint_mae_reproduced"] = metadata_delta <= args.mae_match_tolerance
            if not row["checkpoint_mae_reproduced"]:
                raise RuntimeError(
                    f"{spec.run_id} O00 MAE {mae:.12f} does not reproduce checkpoint "
                    f"metadata {spec.expected_best_val_mae:.12f}; delta={metadata_delta:.3e}"
                )
        else:
            original = predictions_by_order[0]
            original_mae = order_rows[0]["validation_mae"]
            row["delta_mae_from_o00"] = mae - original_mae
            row.update(drift_summary(original, predictions))

        order_rows.append(row)
        print(
            f"{spec.run_id} {label}: MAE={mae:.12f} "
            f"time={runtime['elapsed_seconds']:.2f}s",
            flush=True,
        )

    matrix = np.stack(predictions_by_order, axis=0)
    npz_path = args.output_dir / f"{spec.run_id}_predictions.npz"
    np.savez_compressed(
        npz_path,
        official_validation_indices=np.asarray(official_indices, dtype=np.int64),
        targets=targets_reference,
        order_labels=np.asarray(["O00"] + [f"P{i:02d}" for i in range(1, 11)]),
        permutation_seeds=np.asarray([0] + list(PERMUTATION_SEEDS), dtype=np.int64),
        predictions=matrix,
    )

    all_drifts = np.abs(matrix[1:] - matrix[0:1])
    all_thresholds = ATOL + RTOL * np.abs(matrix[0:1])
    aggregate = {
        "run_id": spec.run_id,
        "model_label": spec.model_label,
        "family": spec.family,
        "seed": spec.seed,
        "o00_validation_mae": order_rows[0]["validation_mae"],
        "mean_permuted_validation_mae": float(
            np.mean([row["validation_mae"] for row in order_rows[1:]])
        ),
        "mean_delta_mae_from_o00": float(
            np.mean([row["delta_mae_from_o00"] for row in order_rows[1:]])
        ),
        "worst_absolute_delta_mae_from_o00": float(
            np.max(np.abs([row["delta_mae_from_o00"] for row in order_rows[1:]]))
        ),
        "mean_absolute_prediction_drift_all_pairs": float(np.mean(all_drifts)),
        "median_absolute_prediction_drift_all_pairs": float(np.median(all_drifts)),
        "p95_absolute_prediction_drift_all_pairs": float(np.quantile(all_drifts, 0.95)),
        "maximum_absolute_prediction_drift_all_pairs": float(np.max(all_drifts)),
        "fraction_sample_permutation_pairs_outside_tolerance": float(
            np.mean(all_drifts > all_thresholds)
        ),
        "prediction_npz": str(npz_path),
    }
    return aggregate, order_rows


def group_full_results(checkpoint_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in checkpoint_rows:
        grouped.setdefault(str(row["model_label"]), []).append(row)
    output = []
    for model_label, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        o00_values = [float(row["o00_validation_mae"]) for row in rows]
        output.append(
            {
                "model_label": model_label,
                "seeds": [int(row["seed"]) for row in rows],
                "o00_mae_mean": statistics.mean(o00_values),
                "o00_mae_sample_sd": (
                    statistics.stdev(o00_values) if len(o00_values) > 1 else None
                ),
                "mean_delta_mae_from_o00_across_checkpoints": statistics.mean(
                    float(row["mean_delta_mae_from_o00"]) for row in rows
                ),
                "mean_absolute_prediction_drift_all_pairs_across_checkpoints": statistics.mean(
                    float(row["mean_absolute_prediction_drift_all_pairs"]) for row in rows
                ),
                "p95_absolute_prediction_drift_all_pairs_mean_across_checkpoints": statistics.mean(
                    float(row["p95_absolute_prediction_drift_all_pairs"]) for row in rows
                ),
                "fraction_pairs_outside_tolerance_mean_across_checkpoints": statistics.mean(
                    float(row["fraction_sample_permutation_pairs_outside_tolerance"])
                    for row in rows
                ),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.log_root = args.log_root.resolve()
    args.output_dir = args.output_dir.resolve()
    selected_specs = tuple(
        spec
        for spec in CHECKPOINT_SPECS
        if args.run_id is None or spec.run_id == args.run_id
    )
    if not selected_specs:
        raise ValueError(f"No locked checkpoint matched --run-id {args.run_id!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected_outputs = (
        "manifest.json",
        "phase_a_single_sample_results.json",
        "phase_b_checkpoint_summary.csv",
        "final_results.json",
    )
    collisions = [name for name in protected_outputs if (args.output_dir / name).exists()]
    if collisions:
        raise FileExistsError(
            f"Output directory already contains audit results: {collisions}"
        )

    if args.validation_size != CONTROLLED_VALIDATION_SIZE and not args.single_only:
        raise ValueError(
            "Formal Phase B requires --validation-size 10000. Use --single-only for a gate-only run."
        )
    if str(args.project_root) not in sys.path:
        sys.path.insert(0, str(args.project_root))

    set_reproducibility(42)
    cudnn_initially_enabled = bool(torch.backends.cudnn.enabled)
    if all(spec.family == "gru" for spec in selected_specs):
        # Apply the locked GRU policy before any GRU module is moved to CUDA.
        torch.backends.cudnn.enabled = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked audit")
    device = torch.device("cuda:0")

    script_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "audit_name": (
            "PCQM4Mv2-50k node-relabeling controlled-comparison batch audit"
            if args.run_id is None
            else "PCQM4Mv2-50k node-relabeling isolated checkpoint audit"
        ),
        "audit_script_version": AUDIT_SCRIPT_VERSION,
        "started_at_utc": utc_now(),
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "python": sys.executable,
        "python_version": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "visible_gpu_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "project_root": str(args.project_root),
        "dataset_root": str(args.dataset_root),
        "log_root": str(args.log_root),
        "output_dir": str(args.output_dir),
        "validation_size": args.validation_size,
        "atol": ATOL,
        "rtol": RTOL,
        "permutation_seeds": PERMUTATION_SEEDS,
        "permutation_stride": PERMUTATION_STRIDE,
        "single_only": args.single_only,
        "run_id_filter": args.run_id,
        "checkpoint_count": len(selected_specs),
        "locked_checkpoint_count": len(CHECKPOINT_SPECS),
        "cudnn_initially_enabled": cudnn_initially_enabled,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_runtime_policy": {
            "gru": "disabled_to_match_locked_training_runtime",
            "other_families": "retain_process_initial_setting",
        },
        "status": "RUNNING",
    }
    write_json(args.output_dir / "manifest.json", manifest)

    print("===== PCQM4MV2-50K BATCH NODE-RELABELING AUDIT =====", flush=True)
    for key in (
        "audit_script_version", "script_sha256", "python", "torch", "cuda", "device", "visible_gpu_name",
        "cuda_visible_devices", "project_root", "dataset_root", "log_root", "output_dir",
    ):
        print(f"{key.upper()}: {manifest[key]}", flush=True)
    print("LOCKED_CHECKPOINT_COUNT:", len(CHECKPOINT_SPECS), flush=True)
    print("SELECTED_CHECKPOINT_COUNT:", len(selected_specs), flush=True)
    print("RUN_ID_FILTER:", args.run_id, flush=True)
    print("CUDNN_INITIALLY_ENABLED:", manifest["cudnn_initially_enabled"], flush=True)
    print("CUDNN_AVAILABLE:", manifest["cudnn_available"], flush=True)
    print("CUDNN_ENABLED:", manifest["cudnn_enabled"], flush=True)
    print("CUDNN_VERSION:", manifest["cudnn_version"], flush=True)
    print("TEST_SPLIT_MODEL_EVALUATIONS: 0", flush=True)
    print("TEST_LABELS_ACCESSED: 0", flush=True)
    print("TEST_PREDICTIONS_GENERATED: 0", flush=True)
    print("TRAINING_STEPS: 0", flush=True)
    print("OPTIMIZER_STEPS: 0", flush=True)

    graph_cache_candidates_before = sorted(
        str(path) for path in args.dataset_root.rglob("geometric_data_processed.pt")
    )
    manifest["full_dataset_graph_cache_present_before_load"] = bool(
        graph_cache_candidates_before
    )
    manifest["graph_cache_candidates_before_load"] = graph_cache_candidates_before
    dataset = PygPCQM4Mv2Dataset(root=str(args.dataset_root))
    graph_cache_candidates_after = sorted(
        str(path) for path in args.dataset_root.rglob("geometric_data_processed.pt")
    )
    manifest["graph_cache_candidates_after_load"] = graph_cache_candidates_after
    manifest["full_dataset_graph_preprocessing_triggered_by_this_run"] = bool(
        not graph_cache_candidates_before and graph_cache_candidates_after
    )
    split = dataset.get_idx_split()
    official_indices_tensor = split["valid"][: args.validation_size].detach().cpu()
    official_indices = [int(value) for value in official_indices_tensor.tolist()]
    if len(official_indices) != args.validation_size:
        raise AssertionError("Controlled validation subset has the wrong size")
    if official_indices[0] != EXPECTED_OFFICIAL_VALIDATION_INDEX:
        raise AssertionError(
            f"First official validation index {official_indices[0]} != "
            f"{EXPECTED_OFFICIAL_VALIDATION_INDEX}"
        )

    base_graph = dataset[official_indices[0]]
    tensor_metadata = {
        name: tensor_inventory(getattr(base_graph, name))
        for name in ("x", "edge_index", "edge_attr", "y")
    }
    sample_identity = {
        "official_validation_index": official_indices[0],
        "validation_local_index": 0,
        "validation_subset_size": len(official_indices),
        "num_nodes": int(base_graph.num_nodes),
        "num_directed_edges": int(base_graph.edge_index.shape[1]),
        "target": float(base_graph.y.reshape(-1)[0].item()),
        "tensor_metadata": tensor_metadata,
        "graph_sha256": graph_sha256(base_graph),
        "prior_audit_graph_sha256": EXPECTED_SAMPLE["prior_audit_graph_sha256"],
    }
    hard_sample_checks = {
        "official_validation_index": sample_identity["official_validation_index"]
        == EXPECTED_OFFICIAL_VALIDATION_INDEX,
        "validation_subset_size": sample_identity["validation_subset_size"]
        == CONTROLLED_VALIDATION_SIZE,
        "num_nodes": sample_identity["num_nodes"] == EXPECTED_SAMPLE["num_nodes"],
        "num_directed_edges": sample_identity["num_directed_edges"]
        == EXPECTED_SAMPLE["num_directed_edges"],
        "target": math.isclose(
            sample_identity["target"], EXPECTED_SAMPLE["target"], rel_tol=0.0, abs_tol=0.0
        ),
        "x_shape": tuple(base_graph.x.shape) == EXPECTED_SAMPLE["x_shape"],
        "edge_index_shape": tuple(base_graph.edge_index.shape)
        == EXPECTED_SAMPLE["edge_index_shape"],
        "edge_attr_shape": tuple(base_graph.edge_attr.shape)
        == EXPECTED_SAMPLE["edge_attr_shape"],
        "y_shape": tuple(base_graph.y.shape) == EXPECTED_SAMPLE["y_shape"],
        "x_dtype": str(base_graph.x.dtype) == EXPECTED_SAMPLE["x_dtype"],
        "edge_index_dtype": str(base_graph.edge_index.dtype)
        == EXPECTED_SAMPLE["edge_index_dtype"],
        "edge_attr_dtype": str(base_graph.edge_attr.dtype)
        == EXPECTED_SAMPLE["edge_attr_dtype"],
        "y_dtype": str(base_graph.y.dtype) == EXPECTED_SAMPLE["y_dtype"],
        "node_feature_indices_nonnegative": bool(torch.all(base_graph.x >= 0).item()),
        "edge_indices_in_range": bool(
            torch.all(base_graph.edge_index >= 0).item()
            and torch.all(base_graph.edge_index < int(base_graph.num_nodes)).item()
        ),
        "edge_feature_indices_nonnegative": bool(
            torch.all(base_graph.edge_attr >= 0).item()
        ),
    }
    legacy_digest_checks = {
        "x_sha256": tensor_metadata["x"]["raw_bytes_sha256"]
        == EXPECTED_SAMPLE["legacy_x_sha256"],
        "edge_index_sha256": tensor_metadata["edge_index"]["raw_bytes_sha256"]
        == EXPECTED_SAMPLE["legacy_edge_index_sha256"],
        "edge_attr_sha256": tensor_metadata["edge_attr"]["raw_bytes_sha256"]
        == EXPECTED_SAMPLE["legacy_edge_attr_sha256"],
    }
    sample_identity["hard_checks"] = hard_sample_checks
    sample_identity["legacy_digest_checks_diagnostic_only"] = legacy_digest_checks
    sample_identity["legacy_digest_all_match"] = all(legacy_digest_checks.values())
    sample_identity["hard_metadata_pass"] = all(hard_sample_checks.values())
    sample_identity["model_output_anchor_confirmed"] = False
    sample_identity["pass"] = sample_identity["hard_metadata_pass"]
    sample_identity["status"] = "PENDING_MODEL_OUTPUT_ANCHOR"
    if not sample_identity["pass"]:
        raise AssertionError(
            f"Validation sample hard metadata identity failed: {hard_sample_checks}"
        )
    manifest["validation_sample_identity"] = sample_identity
    write_json(args.output_dir / "manifest.json", manifest)
    print(
        "LEGACY_TENSOR_DIGESTS_MATCH: "
        f"{sample_identity['legacy_digest_all_match']} (diagnostic only)",
        flush=True,
    )

    print(
        f"\n===== PHASE A: {len(selected_specs)}-CHECKPOINT SINGLE-SAMPLE GATE =====",
        flush=True,
    )
    phase_a_results = []
    phase_a_csv_rows = []

    for number, spec in enumerate(selected_specs, start=1):
        print(f"\n[{number:02d}/{len(selected_specs):02d}] {spec.run_id}", flush=True)
        cudnn_policy = configure_cudnn_for_spec(spec, cudnn_initially_enabled)
        checkpoint_path = args.log_root / spec.relative_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"Checkpoint is not a mapping: {checkpoint_path}")
        identity = validate_checkpoint_identity(spec, checkpoint_path, checkpoint)
        if not identity["pass"]:
            raise AssertionError(f"Checkpoint identity failed for {spec.run_id}: {identity['checks']}")
        model, source_metadata = resolve_model(args.project_root, spec, checkpoint)
        source_metadata = dict(source_metadata)
        source_metadata["cudnn_runtime_policy"] = cudnn_policy
        model.to(device).eval()
        result = run_phase_a(
            spec, model, base_graph, device, identity, source_metadata
        )
        phase_a_results.append(result)
        summary = result["sensitivity_summary"]
        phase_a_csv_rows.append(
            {
                "run_id": spec.run_id,
                "model_label": spec.model_label,
                "family": spec.family,
                "seed": spec.seed,
                "checkpoint_best_val_mae": spec.expected_best_val_mae,
                "o00_prediction": result["o00_a_prediction"],
                "repeat_absolute_delta": result["o00_repeat_absolute_delta"],
                "num_permutations_outside_tolerance": summary[
                    "num_permutations_outside_tolerance"
                ],
                "mean_absolute_delta": summary["mean_absolute_delta"],
                "median_absolute_delta": summary["median_absolute_delta"],
                "maximum_absolute_delta": summary["maximum_absolute_delta"],
                "permuted_prediction_range": summary["permuted_prediction_range"],
                "source_path": source_metadata["source_path"],
                "source_sha256": source_metadata["source_sha256"],
                "model_class": source_metadata["class"],
                "audit_status": result["audit_status"],
            }
        )
        write_json(args.output_dir / "phase_a_single_sample_results.json", phase_a_results)
        write_csv(args.output_dir / "phase_a_single_sample_summary.csv", phase_a_csv_rows)
        print(
            f"AUDIT_STATUS={result['audit_status']} "
            f"O00={result['o00_a_prediction']:.12f} "
            f"OUTSIDE={summary['num_permutations_outside_tolerance']}/10 "
            f"MEAN_DRIFT={summary['mean_absolute_delta']:.12e} "
            f"MAX_DRIFT={summary['maximum_absolute_delta']:.12e}",
            flush=True,
        )
        del model, checkpoint
        torch.cuda.empty_cache()

    phase_a_pass = all(result["audit_status"] == "PASS" for result in phase_a_results)
    mamba_seed42_result = next(
        (
            result
            for result in phase_a_results
            if result["model_label"] == "GINE-Mamba" and result["seed"] == 42
        ),
        None,
    )
    mamba_anchor_keys = (
        "prior_mamba_seed42_o00_anchor",
        "prior_mamba_seed42_mean_drift_anchor",
        "prior_mamba_seed42_max_drift_anchor",
        "prior_mamba_seed42_outside_count_anchor",
    )
    model_output_anchor_confirmed = bool(
        mamba_seed42_result is not None
        and all(
            mamba_seed42_result["gate_checks"].get(key, False)
            for key in mamba_anchor_keys
        )
    )
    model_output_anchor_required = bool(
        args.run_id is None or args.run_id == "GINE-Mamba_seed42"
    )
    sample_identity["model_output_anchor_required_this_run"] = (
        model_output_anchor_required
    )
    sample_identity["model_output_anchor_confirmed"] = model_output_anchor_confirmed
    sample_identity["model_output_anchor_checks"] = (
        {
            key: bool(mamba_seed42_result["gate_checks"].get(key, False))
            for key in mamba_anchor_keys
        }
        if mamba_seed42_result is not None
        else {key: False for key in mamba_anchor_keys}
    )
    sample_identity["pass"] = bool(
        sample_identity["hard_metadata_pass"]
        and (model_output_anchor_confirmed or not model_output_anchor_required)
    )
    if model_output_anchor_required:
        sample_identity["status"] = (
            "CONFIRMED_BY_HARD_METADATA_AND_MODEL_OUTPUT_ANCHORS"
            if sample_identity["pass"]
            else "FAILED_MODEL_OUTPUT_ANCHOR"
        )
    else:
        sample_identity["status"] = (
            "HARD_METADATA_CONFIRMED; SELECTED_O00_MAE_GATE_REQUIRED_IN_PHASE_B"
        )
    manifest["validation_sample_identity"] = sample_identity
    manifest["phase_a_status"] = "PASS" if phase_a_pass else "FAIL"
    write_json(args.output_dir / "manifest.json", manifest)
    if not phase_a_pass or not sample_identity["pass"]:
        raise RuntimeError(
            "Phase A or the model-output data-identity anchor failed; Phase B was not started"
        )

    if args.single_only:
        manifest["phase_b_status"] = "SKIPPED_BY_ARGUMENT"
        manifest["status"] = "PASS"
        manifest["completed_at_utc"] = utc_now()
        write_json(args.output_dir / "manifest.json", manifest)
        print("\nFINAL_AUDIT_STATUS: PASS", flush=True)
        print("PHASE_B_STATUS: SKIPPED_BY_ARGUMENT", flush=True)
        return 0

    print(
        f"\n===== PHASE B: {len(selected_specs)} CHECKPOINT(S) x 11 "
        "FULL VALIDATION ORDERS =====",
        flush=True,
    )
    phase_b_checkpoint_rows = []
    phase_b_order_rows = []
    for number, spec in enumerate(selected_specs, start=1):
        print(
            f"\n[{number:02d}/{len(selected_specs):02d}] PREPARING {spec.run_id}",
            flush=True,
        )
        configure_cudnn_for_spec(spec, cudnn_initially_enabled)
        checkpoint_path = args.log_root / spec.relative_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        identity = validate_checkpoint_identity(spec, checkpoint_path, checkpoint)
        if not identity["pass"]:
            raise AssertionError(f"Checkpoint identity changed for {spec.run_id}")
        model, _source_metadata = resolve_model(args.project_root, spec, checkpoint)
        model.to(device).eval()
        checkpoint_row, order_rows = run_phase_b(
            spec, model, dataset, official_indices, device, args
        )
        phase_b_checkpoint_rows.append(checkpoint_row)
        phase_b_order_rows.extend(order_rows)
        write_csv(
            args.output_dir / "phase_b_checkpoint_summary.csv", phase_b_checkpoint_rows
        )
        write_csv(args.output_dir / "phase_b_order_results.csv", phase_b_order_rows)
        write_json(
            args.output_dir / "phase_b_partial_results.json",
            {
                "checkpoint_summary": phase_b_checkpoint_rows,
                "order_results": phase_b_order_rows,
            },
        )
        del model, checkpoint
        torch.cuda.empty_cache()

    group_rows = group_full_results(phase_b_checkpoint_rows)
    write_csv(args.output_dir / "phase_b_model_group_summary.csv", group_rows)
    manifest["phase_b_status"] = "PASS"
    manifest["status"] = "PASS"
    manifest["completed_at_utc"] = utc_now()
    manifest["written_files"] = sorted(
        str(path) for path in args.output_dir.iterdir() if path.is_file()
    )
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(
        args.output_dir / "final_results.json",
        {
            "manifest": manifest,
            "phase_a": phase_a_results,
            "phase_b_checkpoint_summary": phase_b_checkpoint_rows,
            "phase_b_order_results": phase_b_order_rows,
            "phase_b_model_group_summary": group_rows,
        },
    )

    print("\n===== MODEL-GROUP SUMMARY =====", flush=True)
    for row in group_rows:
        sd = row["o00_mae_sample_sd"]
        o00_summary = (
            f"{row['o00_mae_mean']:.6f}±{sd:.6f}"
            if sd is not None
            else f"{row['o00_mae_mean']:.6f} (n=1; SD=N/A)"
        )
        print(
            f"{row['model_label']}: "
            f"O00_MAE={o00_summary} "
            f"MEAN_PAIR_DRIFT={row['mean_absolute_prediction_drift_all_pairs_across_checkpoints']:.6e} "
            f"OUTSIDE_FRACTION={row['fraction_pairs_outside_tolerance_mean_across_checkpoints']:.6f}",
            flush=True,
        )
    print("\nFINAL_AUDIT_STATUS: PASS", flush=True)
    print("CHECKPOINTS_LOADED_IN_PHASE_A:", len(selected_specs), flush=True)
    print("CHECKPOINTS_LOADED_IN_PHASE_B:", len(selected_specs), flush=True)
    print(
        "FULL_VALIDATION_FORWARD_EVALUATIONS:",
        len(selected_specs) * 11,
        flush=True,
    )
    print("TEST_SPLIT_MODEL_EVALUATIONS: 0", flush=True)
    print("TEST_LABELS_ACCESSED: 0", flush=True)
    print("TEST_PREDICTIONS_GENERATED: 0", flush=True)
    print("TRAINING_STEPS: 0", flush=True)
    print("OPTIMIZER_STEPS: 0", flush=True)
    print("OUTPUT_DIR:", args.output_dir, flush=True)
    print("===== NODE-RELABELING AUDIT COMPLETE =====", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("\nFINAL_AUDIT_STATUS: FAIL", flush=True)
        print("ERROR_TYPE:", type(error).__name__, flush=True)
        print("ERROR_MESSAGE:", str(error), flush=True)
        raise
