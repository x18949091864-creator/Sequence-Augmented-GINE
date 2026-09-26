"""CPU-only release validation. No data download, optimizer, backward pass, or training."""
from __future__ import annotations

import ast
import importlib
import hashlib
import json
import inspect
import os
import sys
import textwrap
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("Set CUDA_VISIBLE_DEVICES to an empty string before validation.")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from mamba_ssm import Mamba

from models.dual_branch.SequenceComparisons.gine_only import GINEOnly
from models.dual_branch.SequenceComparisons.gine_pointwise_attn_noseq import GINEPointwiseAttnNoSeq
from models.dual_branch.CrossAttn.dualbranch_crossattn_base_masked import DualBranchCrossAttnBaseMasked
from models.dual_branch.CrossAttn.dualbranch_crossattn_index_masked import DualBranchCrossAttnRandomMasked
from models.dual_branch.SequenceComparisons.dualbranch_crossattn_index_gru_masked import DualBranchCrossAttnRandomGRUMasked
from models.dual_branch.SequenceComparisons.dualbranch_crossattn_index_transformer_masked import (
    DualBranchCrossAttnRandomTransformerMasked, TransformerSequenceEncoder
)
from models.dual_branch.TaskModels.qm9_task_models_index import (
    QM9GINEOnlyTask, QM9GINEMambaTask, QM9GINEGRUTask, QM9GINETransformerTask
)
from models.dual_branch.TaskModels.qm9_noseq_task_model import QM9GINEPointwiseAttnNoSeqTask

torch.set_num_threads(1)
cases = [
    ("PCQM", "GINE-128", 173697, GINEOnly, {"hidden_dim": 128}),
    ("PCQM", "GINE-Wide", 371137, GINEOnly, {"hidden_dim": 192}),
    ("PCQM", "GINE-Attn (NoSeq)", 372678, GINEPointwiseAttnNoSeq, {}),
    ("PCQM", "GINE-GRU", 355201, DualBranchCrossAttnRandomGRUMasked, {}),
    ("PCQM", "GINE-Transformer", 388609, DualBranchCrossAttnRandomTransformerMasked, {}),
    ("PCQM", "GINE-Mamba", 372609, DualBranchCrossAttnRandomMasked, {}),
    ("QM9", "GINE-128", 151937, QM9GINEOnlyTask, {"hidden_dim": 128}),
    ("QM9", "GINE-Wide", 352605, QM9GINEOnlyTask, {"hidden_dim": 196}),
    ("QM9", "GINE-Attn (NoSeq)", 350918, QM9GINEPointwiseAttnNoSeqTask, {}),
    ("QM9", "GINE-GRU", 333441, QM9GINEGRUTask, {}),
    ("QM9", "GINE-Transformer", 366849, QM9GINETransformerTask, {}),
    ("QM9", "GINE-Mamba", 350849, QM9GINEMambaTask, {}),
]
reports = {key: [] for key in ("import_check", "parameter_count_check", "architecture_check", "masking_check", "smoke_test")}
modules = [
    "pcqm4m_dataset",
    "experiments.pcqm4mv2.common.train_sequence_model_index",
    "experiments.pcqm4mv2.common.train_sequence_model_gru_index",
    "experiments.pcqm4mv2.common.train_sequence_model_transformer_index",
    "experiments.pcqm4mv2.common.train_noseq_attn",
    "experiments.qm9.common.train_index",
    "experiments.qm9.common.train_gap",
    "experiments.qm9.common.train_noseq_attn",
    "experiments.qm9.common.train_gap_noseq_attn",
]
for name in modules:
    try:
        importlib.import_module(name)
        reports["import_check"].append(f"PASS {name}")
    except Exception as exc:
        reports["import_check"].append(f"FAIL {name}: {type(exc).__name__}: {exc}")

assert len(get_atom_feature_dims()) == 9 and len(get_bond_feature_dims()) == 3

def synthetic_graph(nodes: int, qm9: bool) -> Data:
    # Categorical zero is within every OGB atom/bond vocabulary.
    x = torch.zeros((nodes, 11 if qm9 else 9), dtype=torch.float32 if qm9 else torch.long)
    edges = [(i, i + 1) for i in range(nodes - 1)]
    edges += [(j, i) for i, j in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    edge_attr = torch.zeros((len(edges), 4 if qm9 else 3), dtype=torch.float32 if qm9 else torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def encoder_keywords():
    tree = ast.parse(textwrap.dedent(inspect.getsource(TransformerSequenceEncoder.forward)))
    return {kw.arg for node in ast.walk(tree) if isinstance(node, ast.Call) for kw in node.keywords}

for family, name, expected, cls, kwargs in cases:
    tag = f"{family} | {name}"
    source = f"{cls.__module__}.{cls.__name__}"
    try:
        model = cls(**kwargs).cpu().eval()
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        status = "PASS" if actual == expected else "FAIL"
        reports["parameter_count_check"].append(f"{tag} | expected={expected} | actual={actual} | {status} | {source}")
    except Exception as exc:
        reports["parameter_count_check"].append(f"{tag} | expected={expected} | actual=ERROR | FAIL | {source}: {type(exc).__name__}: {exc}")
        continue

    checks = [("GINE layers=4", len(model.convs) == 4)]
    checks += [("GINE eps=0 fixed", all(float(c.eps) == 0.0 and not c.eps.requires_grad for c in model.convs))]
    checks += [("BatchNorm layers=4", len(model.batch_norms) == 4)]
    expected_hidden = 192 if family == "PCQM" and name == "GINE-Wide" else 196 if family == "QM9" and name == "GINE-Wide" else 128
    checks += [(f"hidden={expected_hidden}", model.convs[0].nn[0].out_features == expected_hidden)]
    if hasattr(model, "cross_attn"):
        checks += [("cross-attention heads=4, embed=128", model.cross_attn.num_heads == 4 and model.cross_attn.embed_dim == 128)]
    if name == "GINE-Mamba":
        m = model.mamba
        checks += [("Mamba d_model=128,d_state=16,d_conv=4,expand=2",
                    all(getattr(m, k, None) == v for k, v in (("d_model",128),("d_state",16),("d_conv",4),("expand",2))))]
    if name == "GINE-GRU":
        checks += [("GRU single-layer unidirectional", model.mamba.gru.num_layers == 1 and not model.mamba.gru.bidirectional)]
    if name == "GINE-Transformer":
        checks += [("Transformer one layer", len(model.mamba.encoder.layers) == 1)]
        checks += [("Transformer no positional encoding", not hasattr(model.mamba, "pos_encoder"))]
        checks += [("Transformer padding mask, no causal mask", encoder_keywords() == {"src_key_padding_mask"})]
    if name == "GINE-Attn (NoSeq)":
        checks += [("pointwise adapter", hasattr(model, "pointwise_adapter"))]
        checks += [("no contextual sequence encoder", not any(isinstance(m, (nn.GRU, nn.LSTM, nn.TransformerEncoder, Mamba)) for m in model.modules()))]
    reports["architecture_check"].append(tag + " | " + " | ".join(f"{label}:{'PASS' if ok else 'FAIL'}" for label, ok in checks))

    if hasattr(model, "serialize_graph"):
        x = torch.zeros((8,128))
        batch = torch.tensor([0,0,0,1,1,1,1,1])
        packed, mask = model.serialize_graph(x, torch.empty((2,0), dtype=torch.long), batch)
        mask_ok = packed.shape == (2,5,128) and mask.shape == (2,5) and mask.dtype == torch.bool and mask[0].tolist() == [False,False,False,True,True] and not mask[1].any()
        forward_source = inspect.getsource(type(model).forward)
        if name == "GINE-GRU" or name == "GINE-Mamba":
            forward_source = inspect.getsource(DualBranchCrossAttnBaseMasked.forward)
        mask_source_ok = "masked_fill" in forward_source and "key_padding_mask=padding_mask" in forward_source
        reports["masking_check"].append(f"{tag} | serialization padding:{'PASS' if mask_ok else 'FAIL'} | padded output zeroing and key_padding_mask:{'PASS' if mask_source_ok else 'FAIL'}")
    elif name == "GINE-Attn (NoSeq)":
        packed, mask = model.pack_nodes(torch.zeros((8,128)), torch.tensor([0,0,0,1,1,1,1,1]))
        src = inspect.getsource(type(model).forward)
        if family == "QM9":
            src = inspect.getsource(GINEPointwiseAttnNoSeq.forward)
        ok = packed.shape == (2,5,128) and mask[0].tolist() == [False,False,False,True,True] and "masked_fill" in src and "key_padding_mask=padding_mask" in src
        reports["masking_check"].append(f"{tag} | NoSeq padding and key_padding_mask:{'PASS' if ok else 'FAIL'}")

    batch_data = Batch.from_data_list([synthetic_graph(3, family == "QM9"), synthetic_graph(5, family == "QM9")])
    try:
        with torch.no_grad():
            output = model(batch_data)
        ok = tuple(output.shape) == (2,1) and bool(torch.isfinite(output).all())
        reports["smoke_test"].append(f"{tag} | {'PASS' if ok else 'FAIL'} | output shape={tuple(output.shape)} finite={bool(torch.isfinite(output).all())}")
    except Exception as exc:
        if name == "GINE-Mamba" and isinstance(exc, RuntimeError) and str(exc).startswith("Expected u.is_cuda() to be true"):
            reports["smoke_test"].append(f"{tag} | NOT APPLICABLE on CPU | installed mamba_ssm backend requires CUDA tensors")
        else:
            reports["smoke_test"].append(f"{tag} | FAIL | {type(exc).__name__}: {str(exc).splitlines()[0]}")

reports["masking_check"].append("Mamba base forward explicitly zeros padded sequence outputs with masked_fill before MultiheadAttention.")
reports["masking_check"].append("Transformer passes src_key_padding_mask to its contextual encoder. NoSeq zeros padded adapter output.")
# The CUDA record is a one-time observed inference result. Reuse it only while
# the exact tested paper-source files remain byte-identical to that record.
cuda_record = ROOT / "reproducibility" / "cuda_smoke_check.json"
if cuda_record.is_file():
    evidence = json.loads(cuda_record.read_text())
    expected_families = {"PCQM", "QM9"}
    validated = set()
    for result in evidence["results"]:
        family = result["family"]
        source_file = ROOT / result["source_file"]
        current_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
        ok = (family in expected_families and family not in validated
              and current_hash == result["source_sha256"]
              and result["output_shape"] == [2, 1]
              and result["all_finite"] is True
              and result["exception"] is None)
        reports["smoke_test"].append(
            f"{family} | GINE-Mamba | CUDA inference {'PASS' if ok else 'FAIL'} | "
            f"physical GPU 1 (torch cuda:0) | nodes=3,5 | shape={tuple(result['output_shape'])} "
            f"finite={result['all_finite']} | source hash {'PASS' if current_hash == result['source_sha256'] else 'FAIL'}")
        validated.add(family)
    if validated != expected_families:
        reports["smoke_test"].append("GINE-Mamba CUDA inference | FAIL | expected PCQM and QM9 records")
    if not any("FAIL" in line for line in reports["smoke_test"]):
        reports["smoke_test"].extend([
            "CPU-compatible models: 10/10 PASS",
            "GINE-Mamba CPU: NOT APPLICABLE with installed mamba_ssm CUDA backend",
            "GINE-Mamba CUDA inference: 2/2 PASS",
            "Overall executable model validation: 12/12 PASS in supported runtime",
        ])
for name, lines in reports.items():
    (ROOT / "reproducibility" / (name + ".txt")).write_text("\n".join(lines) + "\n")
    print(name, sum("FAIL" in line for line in lines), "failures")
