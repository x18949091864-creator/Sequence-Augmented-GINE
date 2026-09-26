"""Offline split and Prefix50k verification. Never loads or downloads datasets."""
from pathlib import Path
import json
import os
import sys
import torch
from torch.utils.data import Subset

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("Set CUDA_VISIBLE_DEVICES to an empty string.")
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from pcqm4m_dataset import PCQM4MPyGDataset
from experiments.pcqm4mv2.common.train_sequence_model_index import create_subset

class FakeOGB:
    def get_idx_split(self):
        return {
            "train": torch.arange(200000, 260000),
            "valid": torch.arange(700000, 712000),
        }

fake = FakeOGB()
train = create_subset(PCQM4MPyGDataset(split="train", dataset=fake), 50000)
valid = create_subset(PCQM4MPyGDataset(split="valid", dataset=fake), 10000)
assert isinstance(train, Subset) and isinstance(valid, Subset)
assert len(train) == 50000 and len(valid) == 10000
assert list(train.indices[:3]) == [0, 1, 2]
assert list(valid.indices[:3]) == [0, 1, 2]
assert [int(train.dataset.indices[i]) for i in (0,1,49999)] == [200000,200001,249999]
assert [int(valid.dataset.indices[i]) for i in (0,1,9999)] == [700000,700001,709999]

pcqm_configs = sorted((root / "experiments/pcqm4mv2").rglob("config*.json"))
assert len(pcqm_configs) == 30
for path in pcqm_configs:
    config = json.loads(path.read_text())
    assert config["train_size"] == 50000 and config["val_size"] == 10000
    assert config["seed"] in (7,42,123,2024,3407)

mu = torch.load(root / "experiments/qm9/split_indices.pt", map_location="cpu", weights_only=True)
gap = torch.load(root / "experiments/qm9/split_indices_gap_verified.pt", map_location="cpu", weights_only=True)
assert mu["target_index"] == 0 and mu["target_name"] == "mu"
assert gap["target_index"] == 4 and gap["target_name"] == "gap"
assert mu["split_seed"] == gap["split_seed"] == 42
assert mu["num_samples"] == gap["num_samples"] == 130831
for name, expected in (("train",110000),("valid",10000),("test",10831)):
    assert len(mu[name]) == len(gap[name]) == expected
    assert torch.equal(mu[name], gap[name])
assert torch.cat([mu[name] for name in ("train","valid","test")]).unique().numel() == 130831

lines = [
    "PASS PCQM official split wrapper preserves original indices.",
    "PASS Prefix50k uses first 50000 official training and first 10000 official validation split positions; no random/scaffold subset construction.",
    "PASS all 30 formal PCQM configs use train_size=50000 and val_size=10000.",
    "PASS QM9 mu target index 0, gap target index 4.",
    "PASS QM9 split seed 42 and sizes 110000/10000/10831; all 130831 indices unique.",
    "PASS mu and verified gap splits have identical train/valid/test indices.",
]
(root / "reproducibility/prefix50k_check.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
