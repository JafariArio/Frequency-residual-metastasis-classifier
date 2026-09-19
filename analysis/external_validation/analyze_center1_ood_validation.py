#!/usr/bin/env python
r"""
Center-1 OOD-validation analysis — Extract the true CAMELYON17-WILDS OOD validation center from the
OOD-validation analysis prediction export and compute manuscript-ready metrics.

Why this correction exists
--------------------------
The Hugging Face mirror's `validation` split contains:
- center 1: the official WILDS Validation (OOD) hospital,
- centers 0, 3, 4: source-domain validation content.

OOD-validation analysis evaluated the full mirrored validation split. That aggregate is valid,
but it should not be described as a pure OOD-validation result.

Center-1 OOD-validation analysis filters the already-generated OOD-validation analysis predictions to:
  center == 1

and computes:
- full scalar metrics,
- confusion matrix and per-class metrics,
- probability summaries by TP/TN/FP/FN,
- stratified bootstrap 95% confidence intervals,
- ROC/PR curve source tables,
- ROC, PR, confusion matrix, and reliability figures,
- manifest JSON.

No model rerun is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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


BOOTSTRAP_METRICS = [
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


def status(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k)) for k in fields})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Center-1 OOD-validation analysis: extract true OOD validation center=1 metrics from OOD-validation analysis predictions."
    )
    ap.add_argument(
        "--ood-validation-dir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "wilds_ood_validation"),
        help="OOD-validation analysis output folder containing the prediction CSV.",
    )
    ap.add_argument(
        "--predictions-csv",
        default="",
        help="Optional explicit OOD-validation predictions CSV. Defaults to the standard file under --ood-validation-dir.",
    )
    ap.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "center1_ood_validation"),
        help="Output folder for center=1 OOD-validation correction results.",
    )
    ap.add_argument("--ood-center", type=int, default=1)
    ap.add_argument("--ece-bins", type=int, default=15)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--bootstrap-seed", type=int, default=20260520)
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    ood_validation_dir = Path(args.ood_validation_dir)
    pred = (
        Path(args.predictions_csv)
        if args.predictions_csv
        else ood_validation_dir / "predictions" / "validation_predictions.csv"
    )
    if not pred.exists():
        raise FileNotFoundError(f"OOD-validation analysis prediction CSV not found: {pred}")
    outdir = Path(args.outdir)
    return {"ood_validation_dir": ood_validation_dir, "predictions_csv": pred, "outdir": outdir}


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
    return {"outdir": outdir, "pred_dir": pred_dir, "stats_dir": stats_dir, "figs_dir": figs_dir}


def ece_binary(y_true: np.ndarray, prob: np.ndarray, bins: int) -> float:
    y = y_true.astype(float)
    p = prob.astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def compute_metrics(df: pd.DataFrame, ece_bins: int) -> Dict[str, Any]:
    y = df["y_true"].astype(int).to_numpy()
    p = df["probability"].astype(float).to_numpy()
    pred = df["pred_selected"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)
    return {
        "n_examples": int(len(df)),
        "n_negative": int((y == 0).sum()),
        "n_positive": int((y == 1).sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "precision_positive": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
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
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


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
        {"normalization": "by_actual_class", "actual_class": "Negative", "pred_negative": safe_div(tn, neg_total), "pred_positive": safe_div(fp, neg_total)},
        {"normalization": "by_actual_class", "actual_class": "Positive", "pred_negative": safe_div(fn, pos_total), "pred_positive": safe_div(tp, pos_total)},
        {"normalization": "by_total", "actual_class": "Negative", "pred_negative": safe_div(tn, total), "pred_positive": safe_div(fp, total)},
        {"normalization": "by_total", "actual_class": "Positive", "pred_negative": safe_div(fn, total), "pred_positive": safe_div(tp, total)},
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


def probability_summary_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for category, sub in df.groupby("category"):
        s = sub["probability"].astype(float)
        rows.append(
            {
                "category": category,
                "count": int(len(sub)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
                "median": float(s.median()),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    return sorted(rows, key=lambda r: r["category"])


def category_count_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{"category": str(k), "count": int(v)} for k, v in df.groupby("category").size().items()]


def save_curves_and_figures(df: pd.DataFrame, stats_dir: Path, figs_dir: Path) -> None:
    y = df["y_true"].astype(int).to_numpy()
    p = df["probability"].astype(float).to_numpy()
    pred = df["pred_selected"].astype(int).to_numpy()

    fpr, tpr, roc_thr = roc_curve(y, p)
    precision, recall, pr_thr = precision_recall_curve(y, p)
    pr_thr_pad = np.concatenate([pr_thr, np.array([np.nan])])

    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(
        stats_dir / "roc_curve_points.csv", index=False
    )
    pd.DataFrame({"recall": recall, "precision": precision, "threshold": pr_thr_pad}).to_csv(
        stats_dir / "pr_curve_points.csv", index=False
    )

    plt.figure(figsize=(5.6, 4.4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("CAMELYON17-WILDS center-1 OOD validation ROC")
    plt.tight_layout()
    plt.savefig(figs_dir / "roc_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(5.6, 4.4))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("CAMELYON17-WILDS center-1 OOD validation PR curve")
    plt.tight_layout()
    plt.savefig(figs_dir / "pr_curve.png", dpi=220)
    plt.close()

    cm = confusion_matrix(y, pred, labels=[0, 1])
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), value in np.ndenumerate(cm):
        plt.text(j, i, f"{int(value):,}", ha="center", va="center")
    plt.title("Center-1 OOD validation confusion matrix")
    plt.tight_layout()
    plt.savefig(figs_dir / "confusion_matrix.png", dpi=220)
    plt.close()

    bins = 15
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers, observed = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            centers.append(float(p[mask].mean()))
            observed.append(float(y[mask].mean()))
    plt.figure(figsize=(5.6, 4.4))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.plot(centers, observed, marker="o", linewidth=2)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Center-1 OOD validation reliability diagram")
    plt.tight_layout()
    plt.savefig(figs_dir / "reliability_diagram.png", dpi=220)
    plt.close()


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    pos_draw = rng.choice(pos, size=len(pos), replace=True)
    neg_draw = rng.choice(neg, size=len(neg), replace=True)
    idx = np.concatenate([pos_draw, neg_draw])
    rng.shuffle(idx)
    return idx


def bootstrap_ci_rows(
    df: pd.DataFrame,
    ece_bins: int,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    progress_every: int,
) -> List[Dict[str, Any]]:
    y = df["y_true"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in BOOTSTRAP_METRICS}
    for i in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y, rng)
        m = compute_metrics(df.iloc[idx].reset_index(drop=True), ece_bins)
        for metric in BOOTSTRAP_METRICS:
            draws[metric].append(float(m[metric]))
        if progress_every > 0 and (i + 1) % progress_every == 0:
            status(f"bootstrap {i + 1:,}/{n_bootstrap:,}")

    point = compute_metrics(df, ece_bins)
    alpha = 1.0 - ci_level
    rows = []
    for metric in BOOTSTRAP_METRICS:
        vals = np.asarray(draws[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "point_estimate": float(point[metric]),
                "ci_level": float(ci_level),
                "ci_lower": float(np.quantile(vals, alpha / 2.0)),
                "ci_upper": float(np.quantile(vals, 1.0 - alpha / 2.0)),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    dirs = prepare_outdir(paths["outdir"], args.overwrite)

    status("Loading OOD-validation analysis prediction export.")
    df = pd.read_csv(paths["predictions_csv"])
    required = ["subset_index", "y_true", "probability", "pred_selected", "category", "center"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OOD-validation analysis predictions missing required columns: {missing}")
    df["center"] = df["center"].astype(int)
    center_counts = df.groupby("center").size().reset_index(name="count")
    center_counts.to_csv(dirs["stats_dir"] / "validation_center_counts.csv", index=False)

    status(f"Filtering to official OOD validation center={args.ood_center}.")
    ood = df[df["center"] == int(args.ood_center)].copy().reset_index(drop=True)
    if len(ood) == 0:
        raise ValueError(f"No rows found for center={args.ood_center}.")
    ood.to_csv(
        dirs["pred_dir"] / "center1_predictions.csv",
        index=False,
    )

    metrics = compute_metrics(ood, args.ece_bins)
    metrics.update(
        {
            "ood_center": int(args.ood_center),
            "source_predictions_csv": str(paths["predictions_csv"]),
            "scientific_boundary": (
                "True CAMELYON17-WILDS OOD validation hospital center only; extracted "
                "from the combined Hugging Face validation mirror using center==1."
            ),
            "n_bootstrap": int(args.n_bootstrap),
            "ci_level": float(args.ci_level),
        }
    )
    write_json(dirs["stats_dir"] / "performance_summary.json", metrics)
    pd.DataFrame([metrics]).to_csv(
        dirs["stats_dir"] / "performance_summary.csv",
        index=False,
    )
    write_csv(
        dirs["stats_dir"] / "confusion_matrix_counts.csv",
        confusion_counts_rows(metrics),
    )
    write_csv(
        dirs["stats_dir"] / "confusion_matrix_normalized.csv",
        confusion_normalized_rows(metrics),
    )
    write_csv(
        dirs["stats_dir"] / "per_class_metrics.csv",
        per_class_rows(ood),
    )
    write_csv(
        dirs["stats_dir"] / "category_counts.csv",
        category_count_rows(ood),
    )
    write_csv(
        dirs["stats_dir"] / "probability_summary_by_category.csv",
        probability_summary_rows(ood),
    )

    status("Saving ROC/PR/confusion/reliability figures.")
    save_curves_and_figures(ood, dirs["stats_dir"], dirs["figs_dir"])

    status("Computing stratified bootstrap confidence intervals.")
    ci_rows = bootstrap_ci_rows(
        ood,
        args.ece_bins,
        args.n_bootstrap,
        args.ci_level,
        args.bootstrap_seed,
        args.progress_every,
    )
    write_csv(
        dirs["stats_dir"] / "bootstrap_95ci.csv",
        ci_rows,
    )

    manifest = {
        "phase": "Center-1 OOD-validation analysis true CAMELYON17-WILDS OOD validation center-1 correction",
        "status": "COMPLETED",
        "ood_center": int(args.ood_center),
        "n_center1_predictions": int(len(ood)),
        "paths": {k: str(v) for k, v in paths.items()},
        "output_paths": {k: str(v) for k, v in dirs.items()},
        "center_counts_from_ood_validation": center_counts.to_dict(orient="records"),
        "metrics_json": str(dirs["stats_dir"] / "performance_summary.json"),
        "bootstrap_ci_csv": str(dirs["stats_dir"] / "bootstrap_95ci.csv"),
        "scientific_boundary": (
            "This package corrects the combined Hugging Face validation mirror into "
            "the official WILDS OOD validation hospital center only."
        ),
    }
    write_json(dirs["outdir"] / "center1_ood_validation_manifest.json", manifest)
    status("Center-1 OOD-validation analysis completed.")
    print(json.dumps(jsonable(manifest), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        raise


