import argparse
import json
import os
import random
import resource
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Disable cuDNN because GRU initialization raises
# CUDNN_STATUS_NOT_INITIALIZED on the current GPU environment.
torch.backends.cudnn.enabled = False
import torch.nn as nn

from torch.utils.data import Subset
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.dual_branch.TaskModels.qm9_task_models_index import (
    QM9GINEOnlyTask,
    QM9GINEMambaTask,
    QM9GINETransformerTask,
    QM9GINEGRUTask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train two-dimensional GINE and fixed-index "
            "GINE-Mamba models on QM9."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path for resuming training.",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def read_available_memory_gib():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    available_kib = int(line.split()[1])
                    return available_kib / 1024 / 1024
    except OSError:
        pass

    return float("nan")


def read_process_rss_gib():
    max_rss_kib = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    return max_rss_kib / 1024 / 1024


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def subset_indices(indices, requested_size):
    if requested_size is None:
        return indices

    requested_size = int(requested_size)

    if requested_size <= 0:
        raise ValueError(
            "Requested subset size must be positive."
        )

    return indices[:min(requested_size, len(indices))]


def build_model(config):
    sequence_model = config["sequence_model"].lower()

    if sequence_model == "gine":
        return QM9GINEOnlyTask(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            node_feature_dim=config.get(
                "node_feature_dim",
                11,
            ),
            edge_feature_dim=config.get(
                "edge_feature_dim",
                4,
            ),
            output_dim=1,
        )

    if sequence_model == "mamba":
        return QM9GINEMambaTask(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            node_feature_dim=config.get(
                "node_feature_dim",
                11,
            ),
            edge_feature_dim=config.get(
                "edge_feature_dim",
                4,
            ),
            output_dim=1,
        )

    if sequence_model == "transformer":
        return QM9GINETransformerTask(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            transformer_heads=config.get(
                "transformer_heads",
                4,
            ),
            dim_feedforward=config.get(
                "dim_feedforward",
                256,
            ),
            node_feature_dim=config.get(
                "node_feature_dim",
                11,
            ),
            edge_feature_dim=config.get(
                "edge_feature_dim",
                4,
            ),
            output_dim=1,
        )

    if sequence_model == "gru":
        return QM9GINEGRUTask(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            gru_num_layers=config.get(
                "gru_num_layers",
                1,
            ),
            node_feature_dim=config.get(
                "node_feature_dim",
                11,
            ),
            edge_feature_dim=config.get(
                "edge_feature_dim",
                4,
            ),
            output_dim=1,
        )

    raise ValueError(
        "Unsupported sequence_model: "
        f"{sequence_model}. "
        "Expected gine, mamba, transformer, or gru."
    )


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_mae,
    best_epoch,
    early_stop_counter,
    config,
    history,
    target_mean,
    target_std,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "early_stop_counter": early_stop_counter,
            "config": config,
            "history": history,
            "target_mean": target_mean,
            "target_std": target_std,
        },
        path,
    )


def normalize_target(
    target,
    target_mean,
    target_std,
):
    return (target - target_mean) / target_std


def denormalize_prediction(
    prediction,
    target_mean,
    target_std,
):
    return prediction * target_std + target_mean


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    gradient_clip,
    target_index,
    target_mean,
    target_std,
):
    model.train()

    normalized_absolute_error_sum = 0.0
    original_absolute_error_sum = 0.0
    sample_count = 0

    progress_bar = tqdm(
        loader,
        desc="Train",
        leave=False,
    )

    for batch in progress_bar:
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        prediction_normalized = model(batch).view(-1)

        target = batch.y[
            :, target_index
        ].view(-1)

        target_normalized = normalize_target(
            target,
            target_mean,
            target_std,
        )

        loss = criterion(
            prediction_normalized,
            target_normalized,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip,
        )

        optimizer.step()

        prediction_original = denormalize_prediction(
            prediction_normalized.detach(),
            target_mean,
            target_std,
        )

        batch_size = target.numel()

        normalized_absolute_error_sum += (
            loss.item() * batch_size
        )

        original_absolute_error_sum += (
            torch.abs(
                prediction_original - target
            ).sum().item()
        )

        sample_count += batch_size

        progress_bar.set_postfix(
            mae_d=(
                f"{original_absolute_error_sum / sample_count:.5f}"
            )
        )

    normalized_mae = (
        normalized_absolute_error_sum
        / sample_count
    )

    original_mae = (
        original_absolute_error_sum
        / sample_count
    )

    return normalized_mae, original_mae


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    target_index,
    target_mean,
    target_std,
    description,
):
    model.eval()

    normalized_absolute_error_sum = 0.0
    original_absolute_error_sum = 0.0
    sample_count = 0

    for batch in tqdm(
        loader,
        desc=description,
        leave=False,
    ):
        batch = batch.to(device)

        prediction_normalized = model(batch).view(-1)

        target = batch.y[
            :, target_index
        ].view(-1)

        target_normalized = normalize_target(
            target,
            target_mean,
            target_std,
        )

        loss = criterion(
            prediction_normalized,
            target_normalized,
        )

        prediction_original = denormalize_prediction(
            prediction_normalized,
            target_mean,
            target_std,
        )

        batch_size = target.numel()

        normalized_absolute_error_sum += (
            loss.item() * batch_size
        )

        original_absolute_error_sum += (
            torch.abs(
                prediction_original - target
            ).sum().item()
        )

        sample_count += batch_size

    normalized_mae = (
        normalized_absolute_error_sum
        / sample_count
    )

    original_mae = (
        original_absolute_error_sum
        / sample_count
    )

    return normalized_mae, original_mae


def save_curves(history, log_dir):
    epochs = history["epoch"]

    plt.figure(figsize=(8, 6))
    plt.plot(
        epochs,
        history["train_mae_d"],
        label="Train MAE",
    )
    plt.plot(
        epochs,
        history["val_mae_d"],
        label="Validation MAE",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MAE (D)")
    plt.title("QM9 Dipole Moment Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        log_dir / "loss_curve.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        epochs,
        history["learning_rate"],
    )
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        log_dir / "lr_curve.png",
        dpi=200,
    )
    plt.close()


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    set_seed(config["seed"])

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_root = Path(config["log_root"])

    log_dir = log_root / (
        f'{config["experiment_name"]}_{timestamp}'
    )

    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (log_dir / "config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    print("Configuration:", config_path)
    print("Log directory:", log_dir)
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(device),
        )

    print(
        "System available memory:",
        f"{read_available_memory_gib():.2f} GiB",
    )

    # ==================================================
    # Dataset and fixed split
    # ==================================================

    dataset_root = config.get(
        "dataset_root",
        "data/QM9",
    )

    split_path = Path(
        config.get(
            "split_path",
            "experiments/qm9/split_indices.pt",
        )
    )

    dataset = QM9(root=dataset_root)

    split = torch.load(
        split_path,
        map_location="cpu",
    )

    if len(dataset) != split["num_samples"]:
        raise RuntimeError(
            "QM9 dataset size does not match "
            "the saved split."
        )

    target_index = int(
        config.get(
            "target_index",
            split["target_index"],
        )
    )

    if target_index != 0:
        raise ValueError(
            "This experiment is configured for "
            "QM9 dipole moment target index 0."
        )

    target_mean = torch.as_tensor(
        split["target_mean"],
        dtype=torch.float32,
        device=device,
    )

    target_std = torch.as_tensor(
        split["target_std"],
        dtype=torch.float32,
        device=device,
    )

    train_indices = subset_indices(
        split["train"],
        config.get("train_size"),
    )

    valid_indices = subset_indices(
        split["valid"],
        config.get("val_size"),
    )

    test_indices = subset_indices(
        split["test"],
        config.get("test_size"),
    )

    train_dataset = Subset(
        dataset,
        train_indices.tolist(),
    )

    valid_dataset = Subset(
        dataset,
        valid_indices.tolist(),
    )

    test_dataset = Subset(
        dataset,
        test_indices.tolist(),
    )

    print("Dataset:", split["dataset"])
    print("Target:", split["target_name"])
    print("Target index:", target_index)
    print("Target unit:", split["target_unit"])
    print("Full dataset size:", len(dataset))
    print("Train size:", len(train_dataset))
    print("Validation size:", len(valid_dataset))
    print("Test size:", len(test_dataset))
    print(
        "Target mean:",
        float(target_mean.item()),
    )
    print(
        "Target std:",
        float(target_std.item()),
    )

    loader_arguments = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": config["pin_memory"],
        "persistent_workers": (
            config.get(
                "persistent_workers",
                False,
            )
            if config["num_workers"] > 0
            else False
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_arguments,
    )

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        **loader_arguments,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_arguments,
    )

    # ==================================================
    # Model
    # ==================================================

    model = build_model(config).to(device)

    print(
        "Sequence model:",
        config["sequence_model"],
    )

    parameter_count = count_trainable_parameters(
        model
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
        )
    )

    criterion = nn.L1Loss(
        reduction="mean"
    )

    start_epoch = 1
    best_val_mae = float("inf")
    best_epoch = 0
    early_stop_counter = 0

    history = {
        "epoch": [],
        "train_mae_normalized": [],
        "train_mae_d": [],
        "val_mae_normalized": [],
        "val_mae_d": [],
        "learning_rate": [],
        "train_seconds": [],
        "val_seconds": [],
        "peak_gpu_memory_mib": [],
        "process_max_rss_gib": [],
        "system_available_memory_gib": [],
    }

    # ==================================================
    # Resume
    # ==================================================

    if args.resume is not None:
        checkpoint_path = Path(
            args.resume
        ).resolve()

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_val_mae = checkpoint.get(
            "best_val_mae",
            float("inf"),
        )

        best_epoch = checkpoint.get(
            "best_epoch",
            0,
        )

        early_stop_counter = checkpoint.get(
            "early_stop_counter",
            0,
        )

        if "history" in checkpoint:
            history = checkpoint["history"]

        print(
            "Resumed from:",
            checkpoint_path,
        )
        print(
            "Resume epoch:",
            start_epoch,
        )

    train_log_path = (
        log_dir / "train.log"
    )

    # ==================================================
    # Training
    # ==================================================

    total_start_time = time.time()

    for epoch in range(
        start_epoch,
        config["epochs"] + 1,
    ):
        print(
            f"\n===== Epoch "
            f"{epoch}/{config['epochs']} ====="
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(
                device
            )

        train_start = time.time()

        (
            train_mae_normalized,
            train_mae_d,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip=config[
                "gradient_clip"
            ],
            target_index=target_index,
            target_mean=target_mean,
            target_std=target_std,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        train_seconds = (
            time.time() - train_start
        )

        val_start = time.time()

        (
            val_mae_normalized,
            val_mae_d,
        ) = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            target_index=target_index,
            target_mean=target_mean,
            target_std=target_std,
            description="Valid",
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        val_seconds = (
            time.time() - val_start
        )

        scheduler.step(val_mae_d)

        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        if torch.cuda.is_available():
            peak_gpu_memory_mib = (
                torch.cuda.max_memory_allocated(
                    device
                )
                / 1024**2
            )
        else:
            peak_gpu_memory_mib = 0.0

        process_max_rss_gib = (
            read_process_rss_gib()
        )

        available_memory_gib = (
            read_available_memory_gib()
        )

        history["epoch"].append(epoch)

        history[
            "train_mae_normalized"
        ].append(
            train_mae_normalized
        )

        history[
            "train_mae_d"
        ].append(
            train_mae_d
        )

        history[
            "val_mae_normalized"
        ].append(
            val_mae_normalized
        )

        history[
            "val_mae_d"
        ].append(
            val_mae_d
        )

        history[
            "learning_rate"
        ].append(
            learning_rate
        )

        history[
            "train_seconds"
        ].append(
            train_seconds
        )

        history[
            "val_seconds"
        ].append(
            val_seconds
        )

        history[
            "peak_gpu_memory_mib"
        ].append(
            peak_gpu_memory_mib
        )

        history[
            "process_max_rss_gib"
        ].append(
            process_max_rss_gib
        )

        history[
            "system_available_memory_gib"
        ].append(
            available_memory_gib
        )

        print(
            f"Train MAE (normalized): "
            f"{train_mae_normalized:.8f}"
        )

        print(
            f"Train MAE (D):          "
            f"{train_mae_d:.8f}"
        )

        print(
            f"Validation MAE (D):     "
            f"{val_mae_d:.8f}"
        )

        print(
            f"Learning rate:          "
            f"{learning_rate:.8f}"
        )

        print(
            f"Train time:             "
            f"{train_seconds:.2f} s"
        )

        print(
            f"Validation time:        "
            f"{val_seconds:.2f} s"
        )

        print(
            f"Peak GPU memory:        "
            f"{peak_gpu_memory_mib:.2f} MiB"
        )

        print(
            f"Process max RSS:        "
            f"{process_max_rss_gib:.3f} GiB"
        )

        print(
            f"System available RAM:   "
            f"{available_memory_gib:.2f} GiB"
        )

        log_record = {
            "epoch": epoch,
            "train_mae_normalized": (
                train_mae_normalized
            ),
            "train_mae_d": train_mae_d,
            "val_mae_normalized": (
                val_mae_normalized
            ),
            "val_mae_d": val_mae_d,
            "learning_rate": learning_rate,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "peak_gpu_memory_mib": (
                peak_gpu_memory_mib
            ),
            "process_max_rss_gib": (
                process_max_rss_gib
            ),
            "system_available_memory_gib": (
                available_memory_gib
            ),
        }

        with train_log_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(log_record)
                + "\n"
            )

        if val_mae_d < best_val_mae:
            best_val_mae = val_mae_d
            best_epoch = epoch
            early_stop_counter = 0

            save_checkpoint(
                path=(
                    checkpoint_dir
                    / "best_model.pt"
                ),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_mae=best_val_mae,
                best_epoch=best_epoch,
                early_stop_counter=(
                    early_stop_counter
                ),
                config=config,
                history=history,
                target_mean=target_mean.cpu(),
                target_std=target_std.cpu(),
            )

            print("Best model saved.")

        else:
            early_stop_counter += 1

        save_checkpoint(
            path=(
                checkpoint_dir
                / "last_model.pt"
            ),
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_mae=best_val_mae,
            best_epoch=best_epoch,
            early_stop_counter=(
                early_stop_counter
            ),
            config=config,
            history=history,
            target_mean=target_mean.cpu(),
            target_std=target_std.cpu(),
        )

        with (
            log_dir / "history.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history,
                file,
                indent=2,
            )

        save_curves(
            history,
            log_dir,
        )

        if (
            early_stop_counter
            >= config["patience"]
        ):
            print(
                "Early stopping triggered."
            )
            break

    total_seconds = (
        time.time() - total_start_time
    )

    # ==================================================
    # Final test using best validation checkpoint
    # ==================================================

    best_checkpoint_path = (
        checkpoint_dir / "best_model.pt"
    )

    if not best_checkpoint_path.exists():
        raise RuntimeError(
            "Best model checkpoint was not created."
        )

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    (
        test_mae_normalized,
        test_mae_d,
    ) = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        target_index=target_index,
        target_mean=target_mean,
        target_std=target_std,
        description="Test",
    )

    result = {
        "experiment_name": (
            config["experiment_name"]
        ),
        "dataset": "QM9",
        "target_index": target_index,
        "target_name": "mu",
        "target_unit": "D",
        "sequence_model": (
            config["sequence_model"]
        ),
        "seed": config["seed"],
        "split_seed": split["split_seed"],
        "train_size": len(train_dataset),
        "validation_size": (
            len(valid_dataset)
        ),
        "test_size": len(test_dataset),
        "target_mean": float(
            target_mean.item()
        ),
        "target_std": float(
            target_std.item()
        ),
        "best_val_mae_d": (
            best_val_mae
        ),
        "test_mae_d": test_mae_d,
        "test_mae_normalized": (
            test_mae_normalized
        ),
        "best_epoch": best_epoch,
        "completed_epochs": (
            history["epoch"][-1]
            if history["epoch"]
            else 0
        ),
        "trainable_parameters": (
            parameter_count
        ),
        "total_seconds": total_seconds,
        "log_directory": str(log_dir),
    }

    with (
        log_dir / "result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print("\nTraining finished.")
    print(
        "Best validation MAE (D):",
        best_val_mae,
    )
    print(
        "Test MAE (D):",
        test_mae_d,
    )
    print(
        "Best epoch:",
        best_epoch,
    )
    print(
        "Total time:",
        f"{total_seconds:.2f} s",
    )
    print(
        "Results:",
        log_dir,
    )


if __name__ == "__main__":
    main()
