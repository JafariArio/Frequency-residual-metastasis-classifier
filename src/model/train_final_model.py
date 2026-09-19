#!/usr/bin/env python
r"""
Phase 7 11M scaling test for MetaPath / PathEOM-Net on PCam patch-level classification.

Purpose:
- Test whether the successful Phase 7 R2 frequency-residual design remains useful when scaled to approximately 11M parameters.
- Keep the same regularized training logic and specificity-constrained validation selection used by the successful compact Phase 7 models.
- This is an exploratory scaling control, not an automatic replacement for the compact final model.

Paper-1 scope: PCam patch-level only. No whole-slide learning.

Variants:
  p7_r2_reg_11m      : ~11M R2-like frequency residual with regularized training.
  p7_r2_sens_reg_11m : ~11M R2-like frequency residual with mild positive weighting and constrained thresholding.

Architecture scaling note:
  The compact final Phase 7 model used width=96 (~3.90M parameters).
  This script uses width=168, yielding approximately 10.96M parameters.

Outputs per run:
  - run_config.json
  - history.json
  - summary.json
  - best_validation_constraint.pt
  - best_valid_default.pt

Recommended first run:
py src/model/train_final_model.py --root "<path-to-PCam-data>" --variant p7_r2_sens_reg_11m --outdir "<output-directory>" --epochs 5 --device cpu --num-workers 0
"""

from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# PCam H5 dataset
# -----------------------------
def _find_file(root: Path, split: str, kind: str) -> Path:
    patterns = [
        f"*split_{split}_{kind}.h5",
        f"*{split}_{kind}.h5",
        f"split_{split}_{kind}.h5",
        f"{split}_{kind}.h5",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(root.glob(pat))
        hits.extend(root.glob(f"**/{pat}"))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(f"Could not find PCam {split}_{kind}.h5 under {root}")
    return hits[0]


def _first_dataset_key(h5: h5py.File) -> str:
    keys = list(h5.keys())
    if not keys:
        raise ValueError("H5 file contains no datasets")
    for k in ["x", "y", "X", "Y", "data", "labels"]:
        if k in h5:
            return k
    return keys[0]


def augment_pcam_image(x: np.ndarray) -> np.ndarray:
    """Histology-safe light augmentation for 96x96 RGB patches.
    Input/output is CHW float32 in [0, 1].
    """
    # Flips and rotations are usually safe for patch-level histology.
    if np.random.rand() < 0.5:
        x = x[:, :, ::-1]
    if np.random.rand() < 0.5:
        x = x[:, ::-1, :]
    k = np.random.randint(0, 4)
    if k:
        x = np.rot90(x, k=k, axes=(1, 2))

    # Mild channel-wise stain/color perturbation. Kept deliberately small.
    if np.random.rand() < 0.80:
        gain = np.random.uniform(0.90, 1.10, size=(3, 1, 1)).astype(np.float32)
        bias = np.random.uniform(-0.03, 0.03, size=(3, 1, 1)).astype(np.float32)
        x = x * gain + bias

    # Mild contrast perturbation around per-image mean.
    if np.random.rand() < 0.50:
        mean = x.mean(axis=(1, 2), keepdims=True)
        contrast = np.random.uniform(0.90, 1.10)
        x = (x - mean) * contrast + mean

    # Very small noise; disabled most of the time.
    if np.random.rand() < 0.15:
        x = x + np.random.normal(0.0, 0.01, size=x.shape).astype(np.float32)

    return np.ascontiguousarray(np.clip(x, 0.0, 1.0).astype(np.float32))


class PCamH5(Dataset):
    def __init__(self, root: str, split: str, augment: bool = False):
        self.root = Path(root)
        self.split = split
        self.augment = bool(augment and split == "train")
        self.x_path = _find_file(self.root, split, "x")
        self.y_path = _find_file(self.root, split, "y")
        with h5py.File(self.x_path, "r") as fx, h5py.File(self.y_path, "r") as fy:
            self.x_key = _first_dataset_key(fx)
            self.y_key = _first_dataset_key(fy)
            self.n = int(fx[self.x_key].shape[0])
        self._fx = None
        self._fy = None

    def _open(self):
        if self._fx is None:
            self._fx = h5py.File(self.x_path, "r")
            self._fy = h5py.File(self.y_path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        self._open()
        x = np.asarray(self._fx[self.x_key][idx])
        y = self._fy[self.y_key][idx]
        if x.ndim == 3 and x.shape[-1] == 3:
            x = np.transpose(x, (2, 0, 1))
        x = x.astype(np.float32) / 255.0
        if self.augment:
            x = augment_pcam_image(x)
        else:
            x = np.ascontiguousarray(x)
        y = float(np.asarray(y).reshape(-1)[0])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# -----------------------------
# Model blocks
# -----------------------------
class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None, groups=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, c, expand=2, drop_path=0.0):
        super().__init__()
        h = c * expand
        self.drop_path = float(drop_path)
        self.net = nn.Sequential(
            ConvBNAct(c, h, 1, 1, 0),
            ConvBNAct(h, h, 3, 1, 1, groups=max(1, h // 16)),
            nn.Conv2d(h, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.net(x)
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            mask = torch.empty((x.shape[0], 1, 1, 1), device=x.device).bernoulli_(keep) / keep
            y = y * mask
        return self.act(x + y)


class LearnableFFTResidual(nn.Module):
    """Lightweight frequency residual, deliberately weaker than AFOM.
    Phase 7 reduces residual strength to prevent frequency branch domination.
    """
    def __init__(self, channels: int, hidden: int = 64, strength: float = 0.10, dropout: float = 0.10):
        super().__init__()
        self.strength = float(strength)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, hidden), nn.SiLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        z = torch.fft.rfft2(x.float(), norm="ortho")
        mag = torch.log1p(torch.abs(z))
        low = mag[:, :, : max(1, mag.shape[-2] // 4), : max(1, mag.shape[-1] // 4)].mean(dim=(-2, -1))
        allm = mag.mean(dim=(-2, -1))
        g = self.gate(torch.cat([low, allm], dim=1)).view(x.shape[0], x.shape[1], 1, 1)
        y = self.post(x * g)
        return x + self.strength * y


class SpatialBackbone(nn.Module):
    def __init__(self, width=96, depth=(2, 2, 3), freq=False, freq_strength=0.10, drop_path=0.04):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, width // 2, 3, 1),
            ConvBNAct(width // 2, width, 3, 2),
        )
        stages = []
        c = width
        block_id = 0
        total_blocks = sum(depth)
        for si, n in enumerate(depth):
            if si > 0:
                stages.append(ConvBNAct(c, c * 2, 3, 2))
                c *= 2
            for _ in range(n):
                dp = drop_path * (block_id / max(1, total_blocks - 1))
                stages.append(ResidualBlock(c, drop_path=dp))
                block_id += 1
            if freq and si == len(depth) - 1:
                stages.append(LearnableFFTResidual(c, strength=freq_strength, dropout=0.10))
        self.body = nn.Sequential(*stages)
        self.out_channels = c

    def forward(self, x):
        return self.body(self.stem(x))


class SingleNet(nn.Module):
    def __init__(self, width=96, freq=False, dropout=0.20, freq_strength=0.10):
        super().__init__()
        self.backbone = SpatialBackbone(width=width, depth=(2, 2, 3), freq=freq, freq_strength=freq_strength)
        c = self.backbone.out_channels
        self.head = nn.Sequential(
            nn.Linear(c, 384), nn.SiLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(384, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x).squeeze(1)


class DualScaleFreqNet(nn.Module):
    def __init__(self, dropout=0.20):
        super().__init__()
        self.local = SpatialBackbone(width=72, depth=(2, 2, 2), freq=True, freq_strength=0.08)
        self.context = SpatialBackbone(width=72, depth=(1, 2, 3), freq=True, freq_strength=0.08)
        c = self.local.out_channels + self.context.out_channels
        self.head = nn.Sequential(
            nn.Linear(c, 384), nn.SiLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(384, 1),
        )

    def forward(self, x):
        a = F.adaptive_avg_pool2d(self.local(x), 1).flatten(1)
        b = F.adaptive_max_pool2d(self.context(x), 1).flatten(1)
        return self.head(torch.cat([a, b], dim=1)).squeeze(1)


def build_model(variant: str) -> nn.Module:
    # width=168 gives ~10.96M trainable parameters for the R2-like single-branch design.
    if variant == "p7_r2_reg_11m":
        return SingleNet(width=168, freq=True, dropout=0.20, freq_strength=0.10)
    if variant == "p7_r2_sens_reg_11m":
        return SingleNet(width=168, freq=True, dropout=0.20, freq_strength=0.08)
    raise ValueError(f"Unknown 11M variant: {variant}")


# -----------------------------
# Metrics and thresholds
# -----------------------------
def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def expected_calibration_error(y_true, prob, bins=15):
    y_true = np.asarray(y_true).astype(np.float32)
    prob = np.asarray(prob).astype(np.float32)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(y_true[mask].mean()) - float(prob[mask].mean()))
    return float(ece)


def compute_metrics(y, logits, threshold=0.5):
    y = np.asarray(y).astype(np.int32)
    prob = sigmoid_np(logits)
    pred = (prob >= threshold).astype(np.int32)
    out: Dict[str, float] = {}
    if SKLEARN_OK and len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["pr_auc"] = float(average_precision_score(y, prob))
        out["accuracy"] = float(accuracy_score(y, pred))
        out["positive_f1"] = float(f1_score(y, pred, pos_label=1, zero_division=0))
        out["weighted_f1"] = float(f1_score(y, pred, average="weighted", zero_division=0))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    else:
        out["accuracy"] = float((pred == y).mean())
        out["roc_auc"] = out["pr_auc"] = out["positive_f1"] = out["weighted_f1"] = out["balanced_accuracy"] = float("nan")
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out["sensitivity"] = float(tp / max(1, tp + fn))
    out["specificity"] = float(tn / max(1, tn + fp))
    out["brier"] = float(np.mean((prob - y) ** 2))
    out["ece"] = expected_calibration_error(y, prob)
    out["threshold"] = float(threshold)
    out["tp"] = tp; out["fp"] = fp; out["tn"] = tn; out["fn"] = fn
    return out


def choose_threshold(y, logits, mode="bacc", min_specificity=0.91):
    best_t, best_m, best_score = 0.5, None, -1e9
    fallback_t, fallback_m, fallback_score = 0.5, None, -1e9
    for t in np.linspace(0.05, 0.95, 181):
        m = compute_metrics(y, logits, float(t))
        if m["balanced_accuracy"] > fallback_score:
            fallback_t, fallback_m, fallback_score = float(t), m, m["balanced_accuracy"]
        ok = True
        if mode == "bacc_spec":
            ok = m["specificity"] >= min_specificity
            score = m["balanced_accuracy"]
        elif mode == "f1_spec":
            ok = m["specificity"] >= min_specificity
            score = m["positive_f1"]
        elif mode == "bacc":
            score = m["balanced_accuracy"]
        else:
            raise ValueError(mode)
        if ok and score > best_score:
            best_t, best_m, best_score = float(t), m, score
    if best_m is None:
        return fallback_t, fallback_m, False
    return best_t, best_m, True


class SmoothedBCEWithLogits(nn.Module):
    def __init__(self, label_smoothing=0.02, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.pos_weight = pos_weight
    def forward(self, logits, target):
        if self.label_smoothing > 0:
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)


# -----------------------------
# Train/eval
# -----------------------------
def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * x.shape[0]
        n += x.shape[0]
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    ys, logits_all = [], []
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total += float(loss.item()) * x.shape[0]
        n += x.shape[0]
        ys.append(y.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
    y_np = np.concatenate(ys)
    l_np = np.concatenate(logits_all)
    m = compute_metrics(y_np, l_np, threshold=threshold)
    m["loss"] = total / max(1, n)
    return m, y_np, l_np


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--variant", required=True, choices=["p7_r2_reg_11m", "p7_r2_sens_reg_11m"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--pos-weight", type=float, default=1.10)
    ap.add_argument("--min-specificity", type=float, default=0.91)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    train_ds = PCamH5(args.root, "train", augment=not args.no_augment)
    valid_ds = PCamH5(args.root, "valid", augment=False)
    test_ds = PCamH5(args.root, "test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = build_model(args.variant).to(device)

    use_pos = args.variant == "p7_r2_sens_reg_11m"
    posw = torch.tensor([args.pos_weight], dtype=torch.float32, device=device) if use_pos else None
    criterion = SmoothedBCEWithLogits(label_smoothing=args.label_smoothing, pos_weight=posw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    meta = {
        "variant": args.variant,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "root": args.root,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "augment_train": not args.no_augment,
        "use_pos_weight": bool(use_pos),
        "pos_weight": args.pos_weight if use_pos else None,
        "min_specificity": args.min_specificity,
        "selection_rule": "maximize validation balanced accuracy subject to specificity >= min_specificity; fallback to unconstrained balanced accuracy",
        "scaling_test": "Phase 7 R2 design widened from width=96 (~3.90M) to width=168 (~10.96M)",
    }
    (outdir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

    history = []
    best_constraint_score = -1e9
    best_default_score = -1e9
    best_constraint_epoch = None
    bad_epochs = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        valid_m, valid_y, valid_logits = evaluate(model, valid_loader, criterion, device, threshold=0.5)
        t_bacc, m_bacc, ok_bacc = choose_threshold(valid_y, valid_logits, "bacc")
        t_spec, m_spec, ok_spec = choose_threshold(valid_y, valid_logits, "bacc_spec", args.min_specificity)
        t_f1spec, m_f1spec, ok_f1spec = choose_threshold(valid_y, valid_logits, "f1_spec", args.min_specificity)

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"valid_default_{k}": v for k, v in valid_m.items() if k not in ["tp", "tn", "fp", "fn"]})
        row.update({f"valid_bacc_tuned_{k}": v for k, v in m_bacc.items() if k not in ["tp", "tn", "fp", "fn"]})
        row.update({f"valid_spec91_tuned_{k}": v for k, v in m_spec.items() if k not in ["tp", "tn", "fp", "fn"]})
        row["valid_spec91_threshold_found"] = bool(ok_spec)
        row["valid_f1spec_threshold_found"] = bool(ok_f1spec)
        history.append(row)
        print(json.dumps(row, indent=2))

        # Default checkpoint: for comparison only.
        if valid_m["balanced_accuracy"] > best_default_score:
            best_default_score = valid_m["balanced_accuracy"]
            torch.save({"model_state_dict": model.state_dict(), "variant": args.variant, "epoch": epoch, "config": meta}, outdir / "best_valid_default.pt")

        # Main checkpoint: specificity-constrained validation selection.
        candidate_score = m_spec["balanced_accuracy"] if ok_spec else m_bacc["balanced_accuracy"]
        if candidate_score > best_constraint_score:
            best_constraint_score = candidate_score
            best_constraint_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "variant": args.variant,
                "epoch": epoch,
                "config": meta,
                "valid_selected_threshold": float(m_spec["threshold"] if ok_spec else m_bacc["threshold"]),
                "valid_selected_metrics": m_spec if ok_spec else m_bacc,
                "valid_threshold_type": "spec91" if ok_spec else "bacc_fallback",
            }, outdir / "best_validation_constraint.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} after {bad_epochs} non-improving epochs.")
                break

    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    def load_and_eval(ckpt_path: Path, tag: str):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        valid_m, valid_y, valid_logits = evaluate(model, valid_loader, criterion, device, threshold=0.5)
        t_bacc, m_bacc, _ = choose_threshold(valid_y, valid_logits, "bacc")
        t_spec, m_spec, ok_spec = choose_threshold(valid_y, valid_logits, "bacc_spec", args.min_specificity)
        selected_t = float(t_spec if ok_spec else t_bacc)
        selected_type = "spec91" if ok_spec else "bacc_fallback"
        test_default, _, _ = evaluate(model, test_loader, criterion, device, threshold=0.5)
        test_bacc, _, _ = evaluate(model, test_loader, criterion, device, threshold=t_bacc)
        test_selected, _, _ = evaluate(model, test_loader, criterion, device, threshold=selected_t)
        return {
            "tag": tag,
            "epoch": int(ckpt.get("epoch", -1)),
            "valid_default_threshold": valid_m,
            "valid_bacc_threshold": float(t_bacc),
            "valid_bacc_tuned_metrics": m_bacc,
            "valid_spec91_threshold": float(t_spec),
            "valid_spec91_threshold_found": bool(ok_spec),
            "valid_selected_threshold": selected_t,
            "valid_selected_threshold_type": selected_type,
            "test_default_threshold": test_default,
            "test_bacc_tuned_threshold": test_bacc,
            "test_selected_threshold": test_selected,
        }

    summary = {
        "model": "MetaPath_Phase7_Regularized_Redesign_11M_Scaling_Test",
        "variant": args.variant,
        "main_selection": "best_validation_constraint.pt",
        "constraint_rule": f"validation threshold maximizes balanced accuracy with specificity >= {args.min_specificity}; fallback to unconstrained balanced accuracy",
        "constraint_checkpoint": load_and_eval(outdir / "best_validation_constraint.pt", "best_validation_constraint"),
        "default_checkpoint": load_and_eval(outdir / "best_valid_default.pt", "best_valid_default"),
        "parameter_count": count_parameters(model),
        "elapsed_sec": time.time() - start,
        "config": meta,
        "resnet18_targets_to_beat": {
            "balanced_accuracy": 0.8541,
            "positive_f1": 0.8447,
            "sensitivity": 0.7939,
            "specificity_minimum": 0.91,
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FINAL_SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


