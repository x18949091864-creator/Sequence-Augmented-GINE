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
import torch.nn as nn

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ogb.lsc import PCQM4Mv2Dataset
from pcqm4m_dataset import PCQM4MPyGDataset
from models.dual_branch.CrossAttn.dualbranch_crossattn_random_masked import (
    DualBranchCrossAttnRandomMasked,
)
from models.dual_branch.SequenceComparisons.dualbranch_crossattn_random_gru_masked import (
    DualBranchCrossAttnRandomGRUMasked,
)
from models.dual_branch.SequenceComparisons.dualbranch_crossattn_index_transformer_masked import (
    DualBranchCrossAttnRandomTransformerMasked,
)
from models.dual_branch.SequenceComparisons.gine_only import (
    GINEOnly,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train padding-aware GINE sequence models on PCQM4Mv2."
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

    # Exact bitwise reproducibility on GPU is not guaranteed,
    # but validation ordering is deterministic.
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
    # On Linux, ru_maxrss is reported in KiB.
    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss_kib / 1024 / 1024


def create_subset(dataset, requested_size):
    if requested_size is None:
        return dataset

    requested_size = int(requested_size)

    if requested_size <= 0:
        raise ValueError("Dataset subset size must be positive.")

    actual_size = min(requested_size, len(dataset))

    return Subset(
        dataset,
        range(actual_size),
    )


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_mae,
    early_stop_counter,
    config,
    history,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mae": best_val_mae,
            "early_stop_counter": early_stop_counter,
            "config": config,
            "history": history,
        },
        path,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    gradient_clip,
):
    model.train()

    absolute_error_sum = 0.0
    sample_count = 0

    progress_bar = tqdm(
        loader,
        desc="Train",
        leave=False,
    )

    for batch in progress_bar:
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(batch).view(-1)
        target = batch.y.view(-1)

        loss = criterion(prediction, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip,
        )

        optimizer.step()

        batch_size = target.numel()

        absolute_error_sum += loss.item() * batch_size
        sample_count += batch_size

        progress_bar.set_postfix(
            mae=f"{loss.item():.5f}"
        )

    return absolute_error_sum / sample_count


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    absolute_error_sum = 0.0
    sample_count = 0

    for batch in tqdm(
        loader,
        desc="Valid",
        leave=False,
    ):
        batch = batch.to(device)

        prediction = model(batch).view(-1)
        target = batch.y.view(-1)

        loss = criterion(prediction, target)

        batch_size = target.numel()

        absolute_error_sum += loss.item() * batch_size
        sample_count += batch_size

    return absolute_error_sum / sample_count


def save_curves(history, log_dir):
    epochs = history["epoch"]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_mae"], label="Train MAE")
    plt.plot(epochs, history["val_mae"], label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Masked GINE-Mamba Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(log_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["learning_rate"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(log_dir / "lr_curve.png", dpi=200)
    plt.close()



def build_model(config):
    sequence_model = config["sequence_model"].lower()

    common_arguments = {
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
    }

    if sequence_model == "gine":
        model = GINEOnly(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
        )

    elif sequence_model == "mamba":
        model = DualBranchCrossAttnRandomMasked(
            **common_arguments,
        )

    elif sequence_model == "gru":
        model = DualBranchCrossAttnRandomGRUMasked(
            **common_arguments,
            gru_num_layers=config.get(
                "gru_num_layers",
                1,
            ),
        )

    elif sequence_model == "transformer":
        model = DualBranchCrossAttnRandomTransformerMasked(
            **common_arguments,
            transformer_heads=config.get(
                "transformer_heads",
                4,
            ),
            dim_feedforward=config.get(
                "dim_feedforward",
                256,
            ),
        )

    else:
        raise ValueError(
            "Unsupported sequence_model: "
            f"{sequence_model}. "
            "Expected one of: gine, mamba, gru, transformer."
        )

    return model

def main():
    args = parse_args()

    config_path = Path(args.config).resolve()

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    set_seed(config["seed"])

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_root = Path(config["log_root"])
    log_dir = log_root / f'{config["experiment_name"]}_{timestamp}'
    checkpoint_dir = log_dir / "checkpoints"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / "config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("Configuration:", config_path)
    print("Log directory:", log_dir)
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(device))

    print(
        "System available memory:",
        f"{read_available_memory_gib():.2f} GiB",
    )

    # ==================================================
    # Dataset
    # ==================================================

    dataset_root = config.get(
        "dataset_root",
        "data/pcqm4m",
    )

    print("Loading one shared PCQM4Mv2 base dataset...")

    shared_base_dataset = PCQM4Mv2Dataset(
        root=dataset_root,
    )

    print(
        "Memory after shared base dataset:",
        f"{read_process_rss_gib():.2f} GiB process peak RSS,",
        f"{read_available_memory_gib():.2f} GiB system available",
    )

    full_train_dataset = PCQM4MPyGDataset(
        split="train",
        dataset=shared_base_dataset,
    )

    full_val_dataset = PCQM4MPyGDataset(
        split="valid",
        dataset=shared_base_dataset,
    )

    if (
        full_train_dataset.dataset
        is not full_val_dataset.dataset
    ):
        raise RuntimeError(
            "Train and validation wrappers are not sharing "
            "the same underlying PCQM4Mv2 dataset."
        )

    print(
        "Shared base dataset check:",
        full_train_dataset.dataset
        is full_val_dataset.dataset,
    )

    train_dataset = create_subset(
        full_train_dataset,
        config["train_size"],
    )

    val_dataset = create_subset(
        full_val_dataset,
        config["val_size"],
    )

    print("Full train size:", len(full_train_dataset))
    print("Full validation size:", len(full_val_dataset))
    print("Used train size:", len(train_dataset))
    print("Used validation size:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=(
            config["persistent_workers"]
            if config["num_workers"] > 0
            else False
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=(
            config["persistent_workers"]
            if config["num_workers"] > 0
            else False
        ),
    )

    # ==================================================
    # Model
    # ==================================================

    if config["sequence_model"].lower() == "gru":
        print(
            "GRU model detected: disabling cuDNN "
            "to avoid cuDNN RNN initialization errors."
        )
        torch.backends.cudnn.enabled = False

    model = build_model(config).to(device)

    print(
        "Sequence model:",
        config["sequence_model"],
    )

    parameter_count = count_trainable_parameters(model)

    print("Trainable parameters:", parameter_count)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )

    criterion = nn.L1Loss(reduction="mean")

    start_epoch = 1
    best_val_mae = float("inf")
    early_stop_counter = 0

    history = {
        "epoch": [],
        "train_mae": [],
        "val_mae": [],
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
        checkpoint_path = Path(args.resume).resolve()

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        start_epoch = checkpoint["epoch"] + 1

        best_val_mae = checkpoint.get(
            "best_val_mae",
            checkpoint.get("val_loss", float("inf")),
        )

        early_stop_counter = checkpoint.get(
            "early_stop_counter",
            0,
        )

        if "history" in checkpoint:
            history = checkpoint["history"]

        print("Resumed from:", checkpoint_path)
        print("Resume epoch:", start_epoch)

    train_log_path = log_dir / "train.log"

    # ==================================================
    # Training
    # ==================================================

    total_start_time = time.time()

    for epoch in range(
        start_epoch,
        config["epochs"] + 1,
    ):
        print(f"\n===== Epoch {epoch}/{config['epochs']} =====")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        train_start = time.time()

        train_mae = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip=config["gradient_clip"],
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        train_seconds = time.time() - train_start

        val_start = time.time()

        val_mae = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        val_seconds = time.time() - val_start

        scheduler.step(val_mae)

        learning_rate = optimizer.param_groups[0]["lr"]

        if torch.cuda.is_available():
            peak_gpu_memory_mib = (
                torch.cuda.max_memory_allocated(device)
                / 1024**2
            )
        else:
            peak_gpu_memory_mib = 0.0

        process_max_rss_gib = read_process_rss_gib()
        available_memory_gib = read_available_memory_gib()

        history["epoch"].append(epoch)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["learning_rate"].append(learning_rate)
        history["train_seconds"].append(train_seconds)
        history["val_seconds"].append(val_seconds)
        history["peak_gpu_memory_mib"].append(
            peak_gpu_memory_mib
        )
        history["process_max_rss_gib"].append(
            process_max_rss_gib
        )
        history["system_available_memory_gib"].append(
            available_memory_gib
        )

        print(f"Train MAE:             {train_mae:.8f}")
        print(f"Validation MAE:        {val_mae:.8f}")
        print(f"Learning rate:         {learning_rate:.8f}")
        print(f"Train time:            {train_seconds:.2f} s")
        print(f"Validation time:       {val_seconds:.2f} s")
        print(
            f"Peak GPU memory:       "
            f"{peak_gpu_memory_mib:.2f} MiB"
        )
        print(
            f"Process max RSS:       "
            f"{process_max_rss_gib:.3f} GiB"
        )
        print(
            f"System available RAM:  "
            f"{available_memory_gib:.2f} GiB"
        )

        log_record = {
            "epoch": epoch,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "learning_rate": learning_rate,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "peak_gpu_memory_mib": peak_gpu_memory_mib,
            "process_max_rss_gib": process_max_rss_gib,
            "system_available_memory_gib": available_memory_gib,
        }

        with train_log_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(log_record) + "\n")

        save_checkpoint(
            path=checkpoint_dir / "last_model.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_mae=best_val_mae,
            early_stop_counter=early_stop_counter,
            config=config,
            history=history,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            early_stop_counter = 0

            save_checkpoint(
                path=checkpoint_dir / "best_model.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_mae=best_val_mae,
                early_stop_counter=early_stop_counter,
                config=config,
                history=history,
            )

            print("Best model saved.")

        else:
            early_stop_counter += 1

        # Rewrite last checkpoint after updating the counter/best value.
        save_checkpoint(
            path=checkpoint_dir / "last_model.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_mae=best_val_mae,
            early_stop_counter=early_stop_counter,
            config=config,
            history=history,
        )

        with (log_dir / "history.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(history, file, indent=2)

        save_curves(history, log_dir)

        if early_stop_counter >= config["patience"]:
            print("Early stopping triggered.")
            break

    total_seconds = time.time() - total_start_time

    result = {
        "experiment_name": config["experiment_name"],
        "best_val_mae": best_val_mae,
        "completed_epochs": (
            history["epoch"][-1]
            if history["epoch"]
            else 0
        ),
        "trainable_parameters": parameter_count,
        "total_seconds": total_seconds,
        "log_directory": str(log_dir),
    }

    with (log_dir / "result.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2)

    print("\nTraining finished.")
    print("Best validation MAE:", best_val_mae)
    print("Total time:", f"{total_seconds:.2f} s")
    print("Results:", log_dir)


if __name__ == "__main__":
    main()
