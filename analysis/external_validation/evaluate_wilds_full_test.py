#!/usr/bin/env python
r"""
Phase 13D — Frozen final 11M PathEOM-Net-R transfer evaluation on the
full CAMELYON17-WILDS test split created in Phase 13C.

Scientific role
---------------
This is an out-of-source patch-level transfer stress test.
It is NOT whole-slide inference, NOT clinical deployment validation, and NOT a
new threshold-tuned external benchmark.

Frozen decisions
----------------
- Model: final p7_r2_sens_reg_11m PathEOM-Net-R
- Checkpoint: existing final Phase 7 checkpoint
- Threshold: frozen PCam-selected operating threshold = 0.220
- No retraining
- No recalibration on the external subset
- No external threshold retuning

Default inputs
--------------
Subset directory:
  <repository>/data/raw/wilds_full_test

Training script used to reconstruct the model:
  <repository>/src/model/train_final_model.py

Final checkpoint:
  <repository>/checkpoints/final_model/best_validation_constraint.pt

Outputs
-------
Default output directory:
  <repository>/results/wilds_full_test

Key exports:
- predictions\test_predictions.csv
- summary_stats\performance_summary.json
- summary_stats\performance_summary.csv
- summary_stats\bootstrap_95ci.csv
- summary_stats\confusion_matrix_counts.csv
- summary_stats\confusion_matrix_normalized.csv
- summary_stats\per_class_metrics.csv
- summary_stats\roc_curve_points.csv
- summary_stats\pr_curve_points.csv
- figures\roc_curve.png
- figures\pr_curve.png
- figures\confusion_matrix.png
- figures\reliability_diagram.png
- wilds_full_test_evaluation_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        matthews_corrcoef,
        cohen_kappa_score,
        log_loss,
        roc_curve,
        precision_recall_curve,
        brier_score_loss,
    )
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Install with: py -m pip install scikit-learn"
    ) from exc


KEY_BOOTSTRAP_METRICS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision_positive",
    "positive_f1",
    "weighted_f1",
    "mcc",
    "brier_score",
    "ece_15bin",
    "nll_log_loss",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def status(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k)) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen final 11M PathEOM-Net-R transfer evaluation on CAMELYON17-WILDS subset."
    )
    parser.add_argument(
        "--subset-dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "raw" / "wilds_full_test"),
        help="Phase 13A subset directory containing x/y HDF5 and metadata CSV.",
    )
    parser.add_argument(
        "--x-h5",
        default="",
        help="Optional explicit x.h5 path. Defaults to subset-dir/camelyon17_wilds_full_test_x.h5.",
    )
    parser.add_argument(
        "--y-h5",
        default="",
        help="Optional explicit y.h5 path. Defaults to subset-dir/camelyon17_wilds_full_test_y.h5.",
    )
    parser.add_argument(
        "--metadata-csv",
        default="",
        help="Optional explicit metadata CSV. Defaults to subset-dir/camelyon17_wilds_full_test_metadata.csv if present.",
    )
    parser.add_argument(
        "--train-script",
        default=str(Path(__file__).resolve().parents[2] / "src" / "model" / "train_final_model.py"),
        help="Training script used to reconstruct the final model architecture.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "final_model" / "best_validation_constraint.pt"),
        help="Final 11M checkpoint path.",
    )
    parser.add_argument(
        "--variant",
        default="p7_r2_sens_reg_11m",
        help="Model variant passed to the Phase 7 training-script builder.",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "wilds_full_test"),
        help="Output directory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.220,
        help="Frozen PCam-selected operating threshold. Do not tune on external subset.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto uses CUDA if available, otherwise CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260519)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_input_paths(args: argparse.Namespace) -> Dict[str, Path]:
    subset_dir = Path(args.subset_dir)
    x_h5 = Path(args.x_h5) if args.x_h5 else subset_dir / "camelyon17_wilds_full_test_x.h5"
    y_h5 = Path(args.y_h5) if args.y_h5 else subset_dir / "camelyon17_wilds_full_test_y.h5"
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else subset_dir / "camelyon17_wilds_full_test_metadata.csv"

    for required_path in [x_h5, y_h5, Path(args.train_script), Path(args.checkpoint)]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required input not found: {required_path}")

    return {
        "subset_dir": subset_dir,
        "x_h5": x_h5,
        "y_h5": y_h5,
        "metadata_csv": metadata_csv,
        "train_script": Path(args.train_script),
        "checkpoint": Path(args.checkpoint),
    }


def prepare_outdir(outdir: Path, overwrite: bool) -> Dict[str, Path]:
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {outdir}\n"
            "Use a new --outdir or pass --overwrite intentionally."
        )
    pred_dir = outdir / "predictions"
    stats_dir = outdir / "summary_stats"
    figs_dir = outdir / "figures"
    for d in [outdir, pred_dir, stats_dir, figs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "outdir": outdir,
        "pred_dir": pred_dir,
        "stats_dir": stats_dir,
        "figs_dir": figs_dir,
    }


def first_dataset_key(h5: h5py.File, preferred: Sequence[str]) -> str:
    for key in preferred:
        if key in h5:
            return key
    keys = list(h5.keys())
    if not keys:
        raise ValueError("HDF5 file has no dataset keys.")
    return keys[0]


class TransferSubsetReader:
    def __init__(self, x_h5_path: Path, y_h5_path: Path):
        self.x_path = x_h5_path
        self.y_path = y_h5_path
        self.fx = h5py.File(self.x_path, "r")
        self.fy = h5py.File(self.y_path, "r")
        self.x_key = first_dataset_key(self.fx, ["x", "X", "images", "data"])
        self.y_key = first_dataset_key(self.fy, ["y", "Y", "labels"])
        self.x_ds = self.fx[self.x_key]
        self.y_ds = self.fy[self.y_key]
        self.n = int(self.x_ds.shape[0])
        if int(self.y_ds.shape[0]) != self.n:
            raise ValueError(f"x/y row-count mismatch: x={self.n}, y={self.y_ds.shape[0]}")

    def __len__(self) -> int:
        return self.n

    def get_batch(self, start: int, end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.asarray(self.x_ds[start:end])
        ys = np.asarray(self.y_ds[start:end]).reshape(-1).astype(np.int64)
        if xs.ndim != 4:
            raise ValueError(f"Expected 4D image batch, got shape {xs.shape}")
        if xs.shape[-1] == 3:
            xs = np.transpose(xs, (0, 3, 1, 2))
        xs = np.ascontiguousarray(np.clip(xs.astype(np.float32) / 255.0, 0.0, 1.0))
        idx = np.arange(start, end, dtype=np.int64)
        return xs, ys, idx

    def close(self):
        try:
            self.fx.close()
        except Exception:
            pass
        try:
            self.fy.close()
        except Exception:
            pass


# ============================================================
# Final model reconstruction: copied from Phase 9A logic
# ============================================================

def import_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("final_model_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def try_build_model(mod, variant: str):
    for name in ["build_model_from_variant", "build_model", "make_model", "create_model", "get_model"]:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                sig = inspect.signature(fn)
                if "variant" in sig.parameters:
                    return fn(variant=variant)
                if len(sig.parameters) == 1:
                    return fn(variant)
                if len(sig.parameters) == 0:
                    return fn()
            except Exception:
                pass

    for name in [
        "MetaPathModel", "PathEOMNetR", "PathEOMNet", "MetaPathNet",
        "RegularizedRedesignModel", "PCamModel",
    ]:
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        try:
            sig = inspect.signature(cls)
            if "variant" in sig.parameters:
                return cls(variant=variant)
            if len(sig.parameters) == 1:
                return cls(variant)
            if len(sig.parameters) == 0:
                return cls()
        except Exception:
            pass

    raise RuntimeError(
        "Could not build the model from the training script. "
        "Expected the Phase 7 final 11M training script."
    )


def load_model(train_script: Path, checkpoint: Path, variant: str, device: torch.device):
    mod = import_module_from_path(train_script)
    model = try_build_model(mod, variant)
    model.to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    load_info = {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return model, load_info


def model_logit(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, dict):
        out = out.get("logits", out)
    if out.ndim > 1:
        out = out[:, 0]
    return out


def run_predictions(
    model,
    reader: TransferSubsetReader,
    device: torch.device,
    batch_size: int,
    threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = len(reader)
    batch_size = max(1, int(batch_size))
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        xb, yb, idxs = reader.get_batch(start, end)
        xt = torch.tensor(xb, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = torch.sigmoid(model_logit(model, xt)).detach().cpu().numpy()
        preds = (probs >= threshold).astype(np.int64)
        for idx, y, p, pred in zip(idxs.tolist(), yb.tolist(), probs.tolist(), preds.tolist()):
            rows.append(
                {
                    "subset_index": int(idx),
                    "y_true": int(y),
                    "probability": float(p),
                    "selected_threshold": float(threshold),
                    "pred_selected": int(pred),
                }
            )
        if (start // batch_size) % 20 == 0 or end == n:
            status(f"predictions {end:,}/{n:,}")
    return pd.DataFrame(rows)


def label_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    y = out["y_true"].astype(int)
    pred = out["pred_selected"].astype(int)
    out["category"] = np.select(
        [
            (y == 1) & (pred == 1),
            (y == 0) & (pred == 0),
            (y == 0) & (pred == 1),
            (y == 1) & (pred == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="NA",
    )
    return out


# ============================================================
# Metrics and figures
# ============================================================

def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def ece_binary(y_true: np.ndarray, prob: np.ndarray, bins: int) -> float:
    y = y_true.astype(np.float64)
    p = prob.astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def compute_metrics(df: pd.DataFrame, ece_bins: int) -> Dict[str, Any]:
    y = df["y_true"].astype(int).to_numpy()
    p = df["probability"].astype(float).to_numpy()
    pred = df["pred_selected"].astype(int).to_numpy()

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)

    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision_pos = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    metrics = {
        "n_examples": int(len(df)),
        "n_negative": int((y == 0).sum()),
        "n_positive": int((y == 1).sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision_positive": float(precision_pos),
        "npv": float(npv),
        "positive_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y, pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y, pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_15bin": float(ece_binary(y, p, ece_bins)),
        "nll_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    return metrics


def confusion_counts_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    tp, tn, fp, fn = metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]
    return [
        {"actual_class": "Negative", "pred_negative": tn, "pred_positive": fp, "row_total": tn + fp},
        {"actual_class": "Positive", "pred_negative": fn, "pred_positive": tp, "row_total": fn + tp},
        {"actual_class": "column_total", "pred_negative": tn + fn, "pred_positive": fp + tp, "row_total": tn + fp + fn + tp},
    ]


def confusion_normalized_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    tp, tn, fp, fn = metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]
    neg_total = tn + fp
    pos_total = fn + tp
    total = tn + fp + fn + tp
    return [
        {
            "normalization": "by_actual_class",
            "actual_class": "Negative",
            "pred_negative": safe_div(tn, neg_total),
            "pred_positive": safe_div(fp, neg_total),
        },
        {
            "normalization": "by_actual_class",
            "actual_class": "Positive",
            "pred_negative": safe_div(fn, pos_total),
            "pred_positive": safe_div(tp, pos_total),
        },
        {
            "normalization": "by_total",
            "actual_class": "Negative",
            "pred_negative": safe_div(tn, total),
            "pred_positive": safe_div(fp, total),
        },
        {
            "normalization": "by_total",
            "actual_class": "Positive",
            "pred_negative": safe_div(fn, total),
            "pred_positive": safe_div(tp, total),
        },
    ]


def per_class_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    y = df["y_true"].astype(int).to_numpy()
    pred = df["pred_selected"].astype(int).to_numpy()
    precision = precision_score(y, pred, average=None, labels=[0, 1], zero_division=0)
    recall = recall_score(y, pred, average=None, labels=[0, 1], zero_division=0)
    f1 = f1_score(y, pred, average=None, labels=[0, 1], zero_division=0)
    support = [(y == 0).sum(), (y == 1).sum()]
    return [
        {"class_label": 0, "class_name": "Negative", "precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
        {"class_label": 1, "class_name": "Positive", "precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
    ]


def save_curve_tables_and_figures(df: pd.DataFrame, stats_dir: Path, figs_dir: Path) -> None:
    y = df["y_true"].astype(int).to_numpy()
    p = df["probability"].astype(float).to_numpy()

    fpr, tpr, roc_thr = roc_curve(y, p)
    prec, rec, pr_thr = precision_recall_curve(y, p)
    pr_thr_pad = np.concatenate([pr_thr, np.array([np.nan])])

    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(
        stats_dir / "roc_curve_points.csv", index=False
    )
    pd.DataFrame({"recall": rec, "precision": prec, "threshold": pr_thr_pad}).to_csv(
        stats_dir / "pr_curve_points.csv", index=False
    )

    plt.figure(figsize=(5.6, 4.4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("CAMELYON17-WILDS transfer ROC curve")
    plt.tight_layout()
    plt.savefig(figs_dir / "roc_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(5.6, 4.4))
    plt.plot(rec, prec, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("CAMELYON17-WILDS transfer precision-recall curve")
    plt.tight_layout()
    plt.savefig(figs_dir / "pr_curve.png", dpi=220)
    plt.close()


def save_confusion_figure(metrics: Dict[str, Any], figs_dir: Path) -> None:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), value in np.ndenumerate(cm):
        plt.text(j, i, f"{value:,}", ha="center", va="center")
    plt.title("CAMELYON17-WILDS transfer confusion matrix")
    plt.tight_layout()
    plt.savefig(figs_dir / "confusion_matrix.png", dpi=220)
    plt.close()


def save_reliability_figure(df: pd.DataFrame, bins: int, figs_dir: Path) -> None:
    y = df["y_true"].astype(int).to_numpy()
    p = df["probability"].astype(float).to_numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = []
    observed = []
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            centers.append(float(p[mask].mean()))
            observed.append(float(y[mask].mean()))
            counts.append(int(mask.sum()))
    plt.figure(figsize=(5.6, 4.4))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.plot(centers, observed, marker="o", linewidth=2)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("CAMELYON17-WILDS transfer reliability diagram")
    plt.tight_layout()
    plt.savefig(figs_dir / "reliability_diagram.png", dpi=220)
    plt.close()


def bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    pos_draw = rng.choice(pos, size=len(pos), replace=True)
    neg_draw = rng.choice(neg, size=len(neg), replace=True)
    idx = np.concatenate([pos_draw, neg_draw])
    rng.shuffle(idx)
    return idx


def bootstrap_ci_rows(df: pd.DataFrame, ece_bins: int, n_bootstrap: int, ci_level: float, seed: int, progress_every: int) -> List[Dict[str, Any]]:
    y_all = df["y_true"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in KEY_BOOTSTRAP_METRICS}

    for i in range(n_bootstrap):
        idx = bootstrap_indices(y_all, rng)
        boot_df = df.iloc[idx].reset_index(drop=True)
        m = compute_metrics(boot_df, ece_bins)
        for metric in KEY_BOOTSTRAP_METRICS:
            draws[metric].append(float(m[metric]))
        if progress_every > 0 and (i + 1) % progress_every == 0:
            status(f"bootstrap {i + 1:,}/{n_bootstrap:,}")

    point = compute_metrics(df, ece_bins)
    alpha = 1.0 - ci_level
    rows = []
    for metric in KEY_BOOTSTRAP_METRICS:
        values = np.asarray(draws[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "point_estimate": float(point[metric]),
                "ci_level": float(ci_level),
                "ci_lower": float(np.quantile(values, alpha / 2.0)),
                "ci_upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return rows


def merge_metadata_if_available(pred_df: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        return pred_df
    meta = pd.read_csv(metadata_csv)
    if "subset_index" not in meta.columns:
        return pred_df
    merged = pred_df.merge(meta, on="subset_index", how="left", validate="one_to_one")
    return merged


def main() -> int:
    args = parse_args()
    input_paths = resolve_input_paths(args)
    out_paths = prepare_outdir(Path(args.outdir), args.overwrite)
    device = resolve_device(args.device)

    status("Opening CAMELYON17-WILDS transfer subset.")
    reader = TransferSubsetReader(input_paths["x_h5"], input_paths["y_h5"])
    status(f"subset rows={len(reader):,}; device={device}")

    status("Loading frozen final 11M PathEOM-Net-R model.")
    model, load_info = load_model(
        input_paths["train_script"],
        input_paths["checkpoint"],
        args.variant,
        device,
    )

    start_time = time.time()
    pred_df = run_predictions(model, reader, device, args.batch_size, args.threshold)
    pred_df = label_categories(pred_df)
    pred_df = merge_metadata_if_available(pred_df, input_paths["metadata_csv"])

    pred_csv = out_paths["pred_dir"] / "test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    metrics = compute_metrics(pred_df, args.ece_bins)
    metrics.update(
        {
            "threshold": float(args.threshold),
            "variant": args.variant,
            "checkpoint": str(input_paths["checkpoint"]),
            "train_script": str(input_paths["train_script"]),
            "transfer_subset_dir": str(input_paths["subset_dir"]),
            "n_bootstrap": int(args.n_bootstrap),
            "ci_level": float(args.ci_level),
            "frozen_external_transfer_evaluation": True,
        }
    )

    write_json(out_paths["stats_dir"] / "performance_summary.json", metrics)
    pd.DataFrame([metrics]).to_csv(
        out_paths["stats_dir"] / "performance_summary.csv", index=False
    )

    write_csv(
        out_paths["stats_dir"] / "confusion_matrix_counts.csv",
        confusion_counts_rows(metrics),
    )
    write_csv(
        out_paths["stats_dir"] / "confusion_matrix_normalized.csv",
        confusion_normalized_rows(metrics),
    )
    write_csv(
        out_paths["stats_dir"] / "per_class_metrics.csv",
        per_class_rows(pred_df),
    )

    category_counts = pred_df.groupby("category").size().reset_index(name="count")
    category_counts.to_csv(out_paths["stats_dir"] / "category_counts.csv", index=False)
    pred_df.groupby("category")["probability"].agg(
        ["count", "mean", "std", "median", "min", "max"]
    ).reset_index().to_csv(
        out_paths["stats_dir"] / "probability_summary_by_category.csv",
        index=False,
    )

    save_curve_tables_and_figures(pred_df, out_paths["stats_dir"], out_paths["figs_dir"])
    save_confusion_figure(metrics, out_paths["figs_dir"])
    save_reliability_figure(pred_df, args.ece_bins, out_paths["figs_dir"])

    status("Computing stratified bootstrap confidence intervals.")
    ci_rows = bootstrap_ci_rows(
        pred_df,
        args.ece_bins,
        args.n_bootstrap,
        args.ci_level,
        args.bootstrap_seed,
        args.progress_every,
    )
    write_csv(out_paths["stats_dir"] / "bootstrap_95ci.csv", ci_rows)

    elapsed = time.time() - start_time
    manifest = {
        "phase": "Phase 13D final 11M PathEOM CAMELYON17-WILDS transfer evaluation",
        "status": "COMPLETED",
        "scientific_boundary": (
            "Out-of-source patch-level transfer stress test only; not whole-slide inference, "
            "not clinical validation, and no external threshold tuning."
        ),
        "device": str(device),
        "variant": args.variant,
        "threshold": float(args.threshold),
        "n_predictions": int(len(pred_df)),
        "elapsed_seconds": float(elapsed),
        "input_paths": {k: str(v) for k, v in input_paths.items()},
        "output_paths": {k: str(v) for k, v in out_paths.items()},
        "model_load_info": load_info,
        "metrics_json": str(out_paths["stats_dir"] / "performance_summary.json"),
        "bootstrap_ci_csv": str(out_paths["stats_dir"] / "bootstrap_95ci.csv"),
        "predictions_csv": str(pred_csv),
    }
    write_json(out_paths["outdir"] / "wilds_full_test_evaluation_manifest.json", manifest)

    reader.close()
    status("Phase 13D transfer evaluation completed.")
    print(json.dumps(jsonable(manifest), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        raise


