#!/usr/bin/env python
r"""
paired comparator analysis — Paired statistical comparison:
Final 11M PathEOM-Net-R vs standardized baseline fair/full-budget baselines.

This script is designed for the Paper 1 PathEOM / MetaPath PCam manuscript.

It computes, for each comparator:
1. Paired point-metric differences:
   PathEOM - baseline
2. Stratified paired bootstrap 95% confidence intervals for metric differences
3. DeLong ROC-AUC difference test
4. Continuity-corrected McNemar test for selected thresholded predictions
5. Full CSV / JSON / Markdown audit outputs

Default comparators
-------------------
- resnet18
- efficientnet_b0
- convnext_tiny
- deit_tiny

Default local paths
-------------------
PathEOM final 11M prediction CSV:
  <repository>\predictions\pcam\test_predictions.csv

standardized baseline baseline root:
  <repository>\predictions\baselines

Each baseline CSV is expected at:
  <baseline_predictions_root>\<model>\predictions\test_predictions.csv

Output folder:
  <repository>\results\paired_comparator_statistics

Probability columns
-------------------
PathEOM file:
  probability

standardized baseline baseline files:
  probability_calibrated

Thresholded prediction columns
------------------------------
Both:
  pred_selected

The script validates:
- row counts
- unique indices
- exact index alignment
- exact label alignment
- binary labels and predictions
- finite probabilities
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as exc:
    raise RuntimeError(
        "paired comparator analysis requires pandas for reliable CSV alignment and export. "
        "Install with: py -m pip install pandas"
    ) from exc

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )
except Exception as exc:
    raise RuntimeError(
        "paired comparator analysis requires scikit-learn. Install with: py -m pip install scikit-learn"
    ) from exc


DEFAULT_BASELINES = ["resnet18", "efficientnet_b0", "convnext_tiny", "deit_tiny"]
BOOTSTRAP_METRICS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "positive_f1",
    "weighted_f1",
    "sensitivity",
    "specificity",
    "mcc",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def status(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload: Any) -> None:
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


def write_markdown_table(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows available.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                if field.endswith("_p_value"):
                    values.append(f"{value:.4g}")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="paired comparator analysis paired statistics: final 11M PathEOM vs standardized baseline baselines."
    )
    parser.add_argument(
        "--patheom-csv",
        default=str(Path(__file__).resolve().parents[2] / "predictions" / "pcam" / "test_predictions.csv"),
        help="Final 11M PathEOM official PCam test prediction CSV.",
    )
    parser.add_argument(
        "--baseline-predictions-root",
        default=str(Path(__file__).resolve().parents[2] / "predictions" / "baselines"),
        help="Root folder containing standardized baseline model folders.",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "paired_comparator_statistics"),
        help="Output directory for paired comparator analysis statistical results.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=DEFAULT_BASELINES,
        choices=DEFAULT_BASELINES,
        help="Baseline models to compare against PathEOM.",
    )
    parser.add_argument(
        "--baseline-csv",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help=(
            "Optional explicit override for a baseline prediction CSV. "
            "Example: --baseline-csv efficientnet_b0=E:\\...\\test_predictions.csv"
        ),
    )
    parser.add_argument("--patheom-prob-col", default="probability")
    parser.add_argument("--baseline-prob-col", default="probability_calibrated")
    parser.add_argument("--pred-col", default="pred_selected")
    parser.add_argument("--label-col", default="y_true")
    parser.add_argument("--index-col", default="index")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument(
        "--save-draws",
        action="store_true",
        help="Save compressed per-bootstrap metric-difference draws for reproducibility.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing baseline CSVs and continue with available comparators.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print bootstrap progress every N resamples per comparator.",
    )
    return parser.parse_args()


def parse_baseline_overrides(overrides: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --baseline-csv override; expected MODEL=PATH, got: {item}")
        model, path = item.split("=", 1)
        model = model.strip()
        if model not in DEFAULT_BASELINES:
            raise ValueError(f"Unsupported baseline override model: {model}")
        out[model] = Path(path.strip())
    return out


def default_baseline_csv(baseline_predictions_root: Path, model: str) -> Path:
    return baseline_predictions_root / model / "test_predictions.csv"


def assert_columns(df: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Available columns: {list(df.columns)}")


def load_prediction_table(
    path: Path,
    model_name: str,
    prob_col: str,
    pred_col: str,
    label_col: str,
    index_col: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found for {model_name}: {path}")
    df = pd.read_csv(path)
    assert_columns(df, path, [index_col, label_col, prob_col, pred_col])

    out = df[[index_col, label_col, prob_col, pred_col]].copy()
    out.columns = ["index", "y_true", "prob", "pred"]
    out["index"] = out["index"].astype(np.int64)
    out["y_true"] = out["y_true"].astype(np.int64)
    out["pred"] = out["pred"].astype(np.int64)
    out["prob"] = out["prob"].astype(float)

    if out["index"].duplicated().any():
        dup_count = int(out["index"].duplicated().sum())
        raise ValueError(f"{model_name}: duplicate index rows found: {dup_count}")
    if not set(out["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_name}: y_true is not binary.")
    if not set(out["pred"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_name}: pred_selected is not binary.")
    if not np.isfinite(out["prob"].to_numpy()).all():
        raise ValueError(f"{model_name}: non-finite probability values found.")
    if ((out["prob"] < 0) | (out["prob"] > 1)).any():
        raise ValueError(f"{model_name}: probabilities outside [0, 1] found.")
    return out.sort_values("index").reset_index(drop=True)


def align_tables(patheom: pd.DataFrame, baseline: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    merged = patheom.merge(
        baseline,
        on="index",
        how="inner",
        suffixes=("_patheom", "_baseline"),
        validate="one_to_one",
    )
    if len(merged) != len(patheom) or len(merged) != len(baseline):
        raise ValueError(
            f"{baseline_name}: index alignment failed. "
            f"PathEOM n={len(patheom)}, baseline n={len(baseline)}, merged n={len(merged)}."
        )
    mismatch = merged["y_true_patheom"].to_numpy() != merged["y_true_baseline"].to_numpy()
    if np.any(mismatch):
        count = int(np.sum(mismatch))
        raise ValueError(f"{baseline_name}: y_true mismatch after index alignment: {count} rows.")
    merged = merged.rename(columns={"y_true_patheom": "y_true"})
    merged = merged.drop(columns=["y_true_baseline"])
    return merged


def confusion_stats(y_true: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return int(tp), int(tn), int(fp), int(fn)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    tp, tn, fp, fn = confusion_stats(y_true, pred)
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "positive_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    pos_draw = rng.choice(pos, size=len(pos), replace=True)
    neg_draw = rng.choice(neg, size=len(neg), replace=True)
    out = np.concatenate([pos_draw, neg_draw])
    rng.shuffle(out)
    return out


def bootstrap_paired_differences(
    y_true: np.ndarray,
    path_prob: np.ndarray,
    path_pred: np.ndarray,
    base_prob: np.ndarray,
    base_pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
    progress_every: int,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for i in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y_true, rng)
        y_b = y_true[idx]
        path_metrics = compute_metrics(y_b, path_prob[idx], path_pred[idx])
        base_metrics = compute_metrics(y_b, base_prob[idx], base_pred[idx])
        for metric in BOOTSTRAP_METRICS:
            draws[metric].append(float(path_metrics[metric] - base_metrics[metric]))
        if progress_every > 0 and (i + 1) % progress_every == 0:
            status(f"bootstrap {i + 1}/{n_bootstrap}")
    return draws


def ci_bounds(values: Sequence[float], ci_level: float) -> Tuple[float, float]:
    alpha = 1.0 - ci_level
    lo = float(np.quantile(values, alpha / 2.0))
    hi = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lo, hi


# ------------------------------
# DeLong ROC-AUC difference test
# Standard vectorized implementation
# ------------------------------

def compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1.0
    return T2


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :m]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true))
    predictions_sorted_transposed = np.vstack([prob_a, prob_b])[:, order]
    aucs, cov = fast_delong(predictions_sorted_transposed, label_1_count)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ cov @ contrast.T).reshape(-1)[0])
    diff = float(aucs[0] - aucs[1])
    if not math.isfinite(variance) or variance <= 0:
        return {
            "delong_auc_patheom": float(aucs[0]),
            "delong_auc_baseline": float(aucs[1]),
            "delong_auc_difference": diff,
            "delong_variance": variance,
            "delong_z": float("nan"),
            "delong_p_value": float("nan"),
        }
    z = diff / math.sqrt(variance)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "delong_auc_patheom": float(aucs[0]),
        "delong_auc_baseline": float(aucs[1]),
        "delong_auc_difference": diff,
        "delong_variance": variance,
        "delong_z": float(z),
        "delong_p_value": float(p),
    }


# ------------------------------
# McNemar test
# Continuity-corrected chi-square, df=1
# ------------------------------

def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & (~correct_b)))  # PathEOM correct, baseline wrong
    c = int(np.sum((~correct_a) & correct_b))  # PathEOM wrong, baseline correct
    if b + c == 0:
        chi2_cc = 0.0
        p = 1.0
    else:
        chi2_cc = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p = math.erfc(math.sqrt(chi2_cc / 2.0))
    return {
        "mcnemar_patheom_correct_baseline_wrong": b,
        "mcnemar_patheom_wrong_baseline_correct": c,
        "mcnemar_discordant_total": b + c,
        "mcnemar_chi2_cc": float(chi2_cc),
        "mcnemar_p_value": float(p),
    }


def save_draws(path: Path, baseline_name: str, draws: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = list(draws.keys())
    n = len(draws[metrics[0]]) if metrics else 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", "draw_index", *metrics])
        for i in range(n):
            writer.writerow([baseline_name, i + 1, *[draws[m][i] for m in metrics]])


def process_baseline(
    patheom_df: pd.DataFrame,
    baseline_name: str,
    baseline_csv: Path,
    args: argparse.Namespace,
    outdir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    baseline_df = load_prediction_table(
        baseline_csv,
        model_name=baseline_name,
        prob_col=args.baseline_prob_col,
        pred_col=args.pred_col,
        label_col=args.label_col,
        index_col=args.index_col,
    )
    merged = align_tables(patheom_df, baseline_df, baseline_name)

    y = merged["y_true"].to_numpy(dtype=np.int64)
    path_prob = merged["prob_patheom"].to_numpy(dtype=float)
    path_pred = merged["pred_patheom"].to_numpy(dtype=np.int64)
    base_prob = merged["prob_baseline"].to_numpy(dtype=float)
    base_pred = merged["pred_baseline"].to_numpy(dtype=np.int64)

    path_metrics = compute_metrics(y, path_prob, path_pred)
    base_metrics = compute_metrics(y, base_prob, base_pred)

    status(f"{baseline_name}: computing paired bootstrap differences.")
    draws = bootstrap_paired_differences(
        y_true=y,
        path_prob=path_prob,
        path_pred=path_pred,
        base_prob=base_prob,
        base_pred=base_pred,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + sum((i + 1) * ord(ch) for i, ch in enumerate(baseline_name)) % 100000,
        progress_every=args.progress_every,
    )

    rows: List[Dict[str, Any]] = []
    for metric in BOOTSTRAP_METRICS:
        diff = float(path_metrics[metric] - base_metrics[metric])
        lo, hi = ci_bounds(draws[metric], args.ci_level)
        rows.append(
            {
                "baseline": baseline_name,
                "metric": metric,
                "patheom_point": float(path_metrics[metric]),
                "baseline_point": float(base_metrics[metric]),
                "difference_patheom_minus_baseline": diff,
                "ci_level": float(args.ci_level),
                "ci_lower": lo,
                "ci_upper": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "direction": "PathEOM_higher_is_better" if diff > 0 else "Baseline_higher_is_better",
            }
        )

    tests = {
        "baseline": baseline_name,
        **delong_roc_test(y, path_prob, base_prob),
        **mcnemar_test(y, path_pred, base_pred),
    }

    verification = {
        "baseline": baseline_name,
        "baseline_csv": str(baseline_csv),
        "n_aligned": int(len(merged)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "patheom_point_metrics": path_metrics,
        "baseline_point_metrics": base_metrics,
        "index_alignment": "PASS",
        "label_alignment": "PASS",
    }

    per_baseline_dir = outdir / baseline_name
    write_json(per_baseline_dir / "verification_and_point_metrics.json", verification)
    write_json(per_baseline_dir / "statistical_tests.json", tests)
    write_csv(per_baseline_dir / "paired_metric_differences.csv", rows)
    if args.save_draws:
        save_draws(per_baseline_dir / "bootstrap_metric_difference_draws.csv.gz", baseline_name, draws)

    return rows, tests, verification


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    overrides = parse_baseline_overrides(args.baseline_csv)
    baseline_predictions_root = Path(args.baseline_predictions_root)

    patheom_csv = Path(args.patheom_csv)
    patheom_df = load_prediction_table(
        patheom_csv,
        model_name="patheom_final11m",
        prob_col=args.patheom_prob_col,
        pred_col=args.pred_col,
        label_col=args.label_col,
        index_col=args.index_col,
    )
    # Standardized columns for merge suffixing.
    patheom_df = patheom_df.rename(
        columns={"prob": "prob", "pred": "pred"}
    )

    manifest: Dict[str, Any] = {
        "analysis": "Paired statistics for final model versus standardized baselines",
        "start_time": now(),
        "args": vars(args),
        "patheom_csv": str(patheom_csv),
        "processed_baselines": [],
        "missing_baselines": [],
        "failed_baselines": [],
    }
    write_json(outdir / "paired_comparator_statistics_manifest.json", manifest)

    all_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    verification_rows: List[Dict[str, Any]] = []

    # Rename columns once so merge suffixes become prob_patheom / prob_baseline.
    patheom_df_for_merge = patheom_df.rename(
        columns={"prob": "prob", "pred": "pred"}
    )

    # Pandas merge suffixes only apply on overlapping column names; this is intended.
    for baseline_name in args.baselines:
        baseline_csv = overrides.get(baseline_name, default_baseline_csv(baseline_predictions_root, baseline_name))
        if not baseline_csv.exists():
            item = {"baseline": baseline_name, "expected_csv": str(baseline_csv)}
            manifest["missing_baselines"].append(item)
            write_json(outdir / "paired_comparator_statistics_manifest.json", manifest)
            if args.skip_missing:
                status(f"Skipping missing baseline={baseline_name}: {baseline_csv}")
                continue
            raise FileNotFoundError(f"Missing baseline prediction CSV for {baseline_name}: {baseline_csv}")

        try:
            # Ensure merge creates the expected names by using identical source columns.
            rows, tests, verification = process_baseline(
                patheom_df=patheom_df_for_merge,
                baseline_name=baseline_name,
                baseline_csv=baseline_csv,
                args=args,
                outdir=outdir,
            )
            all_rows.extend(rows)
            test_rows.append(tests)
            verification_rows.append(
                {
                    "baseline": baseline_name,
                    "n_aligned": verification["n_aligned"],
                    "n_positive": verification["n_positive"],
                    "n_negative": verification["n_negative"],
                    "index_alignment": verification["index_alignment"],
                    "label_alignment": verification["label_alignment"],
                }
            )
            manifest["processed_baselines"].append(baseline_name)
            write_json(outdir / "paired_comparator_statistics_manifest.json", manifest)
        except Exception as exc:
            failure = {
                "baseline": baseline_name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            manifest["failed_baselines"].append(failure)
            write_json(outdir / "paired_comparator_statistics_manifest.json", manifest)
            write_json(outdir / baseline_name / "FAILED_paired_statistics_error.json", failure)
            raise

    write_csv(outdir / "all_paired_metric_differences.csv", all_rows)
    write_csv(outdir / "all_statistical_tests.csv", test_rows)
    write_csv(outdir / "alignment_verification.csv", verification_rows)

    write_markdown_table(
        outdir / "all_paired_metric_differences.md",
        all_rows,
        fields=[
            "baseline",
            "metric",
            "patheom_point",
            "baseline_point",
            "difference_patheom_minus_baseline",
            "ci_lower",
            "ci_upper",
            "ci_excludes_zero",
        ],
        title="paired comparator analysis paired bootstrap metric differences",
    )
    write_markdown_table(
        outdir / "all_statistical_tests.md",
        test_rows,
        fields=[
            "baseline",
            "delong_auc_difference",
            "delong_z",
            "delong_p_value",
            "mcnemar_patheom_correct_baseline_wrong",
            "mcnemar_patheom_wrong_baseline_correct",
            "mcnemar_chi2_cc",
            "mcnemar_p_value",
        ],
        title="paired comparator analysis DeLong ROC-AUC and McNemar paired tests",
    )

    manifest["end_time"] = now()
    manifest["outputs"] = {
        "paired_metric_differences_csv": str(outdir / "all_paired_metric_differences.csv"),
        "statistical_tests_csv": str(outdir / "all_statistical_tests.csv"),
        "alignment_verification_csv": str(outdir / "alignment_verification.csv"),
        "paired_metric_differences_md": str(outdir / "all_paired_metric_differences.md"),
        "statistical_tests_md": str(outdir / "all_statistical_tests.md"),
    }
    write_json(outdir / "paired_comparator_statistics_manifest.json", manifest)

    status("paired comparator analysis paired-statistics workflow completed.")
    print(json.dumps(jsonable(manifest), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




