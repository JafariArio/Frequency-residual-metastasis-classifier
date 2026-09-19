#!/usr/bin/env python3
"""
Phase 10A — Bootstrap confidence intervals for the final 11M PathEOM-Net-R model.

Purpose
-------
This script consumes the held-out PCam prediction export produced in final analysis,
for example:
    <path-to-final-PCam-test-predictions.csv>

It computes:
  1) full-sample point estimates,
  2) 95% bootstrap confidence intervals for final-model metrics,
  3) optional verification that the input file reproduces the locked final analysis
     final 11M manuscript metrics.

This script does NOT compare against ResNet18, because paired per-patch ResNet18
predictions were not retained. Its output is the statistically valid Phase 10A
single-model uncertainty audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install it with: py -m pip install scikit-learn"
    ) from exc


LOCKED_FINAL_MODEL_METRICS: Dict[str, float] = {
    "roc_auc": 0.954338717321296,
    "pr_auc_average_precision": 0.9600421857409934,
    "accuracy": 0.88885498046875,
    "positive_f1": 0.8833962989050393,
    "weighted_f1": 0.8886130938155858,
    "balanced_accuracy": 0.8888351416324152,
    "sensitivity_recall_positive": 0.8424009281309153,
    "specificity_recall_negative": 0.935269355133915,
    "weighted_precision": 0.8922267031198469,
    "weighted_recall": 0.88885498046875,
    "macro_f1": 0.8886108659122014,
    "brier": 0.10202052090992542,
    "ece": 0.09503302834076521,
    "matthews_corrcoef": 0.7810699414608782,
    "cohen_kappa": 0.7777011090783587,
    "nll_log_loss": 0.3442876274431107,
}

LOCKED_FINAL_MODEL_COUNTS: Dict[str, int] = {
    "tp": 13796,
    "tn": 15330,
    "fp": 1061,
    "fn": 2581,
    "n_total": 32768,
    "n_positive": 16377,
    "n_negative": 16391,
}


METRIC_ORDER: List[str] = [
    "roc_auc",
    "pr_auc_average_precision",
    "accuracy",
    "balanced_accuracy",
    "sensitivity_recall_positive",
    "specificity_recall_negative",
    "positive_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "macro_f1",
    "brier",
    "ece",
    "matthews_corrcoef",
    "cohen_kappa",
    "nll_log_loss",
]


METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "roc_auc": "ROC-AUC",
    "pr_auc_average_precision": "PR-AUC / Average Precision",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "sensitivity_recall_positive": "Sensitivity / Positive recall",
    "specificity_recall_negative": "Specificity / Negative recall",
    "positive_f1": "Positive-class F1",
    "weighted_precision": "Weighted precision",
    "weighted_recall": "Weighted recall",
    "weighted_f1": "Weighted F1",
    "macro_f1": "Macro F1",
    "brier": "Brier score",
    "ece": "ECE (15-bin unless changed)",
    "matthews_corrcoef": "Matthews correlation coefficient",
    "cohen_kappa": "Cohen's kappa",
    "nll_log_loss": "NLL / Log loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap 95% confidence intervals for the final 11M PathEOM-Net-R "
            "official PCam test predictions."
        )
    )
    parser.add_argument(
        "--pred-csv",
        required=True,
        type=Path,
        help="Prediction CSV containing y_true and probability columns.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Output directory for CSV/JSON/XLSX result files.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap replicates. Default: 2000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260517,
        help="Random seed for reproducible bootstrap resampling. Default: 20260517.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level. Default: 0.95.",
    )
    parser.add_argument(
        "--bootstrap-mode",
        choices=["stratified", "ordinary"],
        default="stratified",
        help=(
            "Bootstrap sampling mode. 'stratified' preserves the observed positive/negative "
            "class counts in each replicate and is recommended here. Default: stratified."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional binary threshold if pred_selected/selected_threshold are absent. "
            "The final analysis final prediction file already includes the selected threshold."
        ),
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=15,
        help="Number of equal-width bins for ECE. Default: 15, matching final analysis.",
    )
    parser.add_argument(
        "--save-draws",
        action="store_true",
        help="Save all per-bootstrap metric draws to CSV and XLSX. Recommended for reproducibility.",
    )
    parser.add_argument(
        "--write-xlsx",
        action="store_true",
        help="Attempt to write an Excel workbook. Requires openpyxl.",
    )
    parser.add_argument(
        "--verify-final-model",
        action="store_true",
        help=(
            "Verify that the input reproduces the locked final analysis final 11M metrics and counts. "
            "Recommended for the current authoritative prediction file."
        ),
    )
    parser.add_argument(
        "--verify-tol",
        type=float,
        default=5e-8,
        help="Absolute tolerance for locked-metric verification. Default: 5e-8.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N bootstrap replicates. Default: 100; set 0 to silence.",
    )
    return parser.parse_args()


def validate_probability_vector(probs: np.ndarray) -> None:
    if probs.ndim != 1:
        raise ValueError("Probability vector must be one-dimensional.")
    if np.any(~np.isfinite(probs)):
        raise ValueError("Probability vector contains NaN or infinite values.")
    if np.any((probs < 0.0) | (probs > 1.0)):
        bad_min = float(np.min(probs))
        bad_max = float(np.max(probs))
        raise ValueError(
            f"Probability values must lie in [0, 1]. Observed min={bad_min}, max={bad_max}."
        )


def validate_binary_labels(y_true: np.ndarray, name: str) -> None:
    unique = set(np.unique(y_true).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0/1 values. Observed: {sorted(unique)}")
    if len(unique) < 2:
        raise ValueError(f"{name} must contain both classes 0 and 1.")


def read_predictions(
    pred_csv: Path,
    threshold_override: float | None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float | None, List[str]]:
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    df = pd.read_csv(pred_csv)
    required = {"y_true", "probability"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    warnings: List[str] = []
    y_true = df["y_true"].to_numpy(dtype=int)
    probs = df["probability"].to_numpy(dtype=float)
    validate_binary_labels(y_true, "y_true")
    validate_probability_vector(probs)

    threshold: float | None = None
    if "selected_threshold" in df.columns:
        unique_thresholds = pd.to_numeric(df["selected_threshold"], errors="coerce").dropna().unique()
        if len(unique_thresholds) == 1:
            threshold = float(unique_thresholds[0])
        elif len(unique_thresholds) > 1:
            warnings.append(
                "selected_threshold contains more than one value; the script will rely on pred_selected if present."
            )

    if threshold_override is not None:
        if not (0.0 <= threshold_override <= 1.0):
            raise ValueError("--threshold must lie in [0, 1].")
        threshold = float(threshold_override)

    if "pred_selected" in df.columns:
        pred_selected = df["pred_selected"].to_numpy(dtype=int)
        validate_binary_labels_with_single_class_ok(pred_selected, "pred_selected")
        if threshold is not None:
            derived = (probs >= threshold).astype(int)
            mismatch_count = int(np.sum(derived != pred_selected))
            if mismatch_count != 0:
                warnings.append(
                    f"pred_selected and probability>=threshold disagree for {mismatch_count} rows. "
                    "The script will use pred_selected exactly as exported."
                )
    else:
        if threshold is None:
            raise ValueError(
                "Prediction CSV lacks pred_selected, and no usable threshold was found. "
                "Provide --threshold or use a CSV with pred_selected/selected_threshold."
            )
        pred_selected = (probs >= threshold).astype(int)
        warnings.append(
            "pred_selected was absent; binary predictions were derived from probability>=threshold."
        )

    return df, y_true, probs, pred_selected, threshold, warnings


def validate_binary_labels_with_single_class_ok(arr: np.ndarray, name: str) -> None:
    unique = set(np.unique(arr).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0/1 values. Observed: {sorted(unique)}")


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Equal-width positive-rate-vs-probability ECE.

    This matches the final analysis final summary definition:
    '15-bin positive-rate-vs-probability ECE matching Phase 7 training script'.
    """
    if n_bins < 2:
        raise ValueError("ECE requires at least 2 bins.")

    y_true = y_true.astype(float, copy=False)
    probs = probs.astype(float, copy=False)
    n = probs.size
    if n == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= left) & (probs < right)
        else:
            mask = (probs >= left) & (probs <= right)
        count = int(np.sum(mask))
        if count == 0:
            continue
        bin_confidence = float(np.mean(probs[mask]))
        bin_positive_rate = float(np.mean(y_true[mask]))
        ece += (count / n) * abs(bin_positive_rate - bin_confidence)

    return float(ece)


def confusion_counts(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    pred: np.ndarray,
    ece_bins: int,
    include_counts: bool = False,
) -> Dict[str, float | int]:
    validate_binary_labels(y_true, "y_true")
    validate_probability_vector(probs)
    validate_binary_labels_with_single_class_ok(pred, "pred")

    counts = confusion_counts(y_true, pred)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    metrics: Dict[str, float | int] = {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity_recall_positive": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "specificity_recall_negative": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "positive_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "weighted_precision": float(precision_score(y_true, pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": float(expected_calibration_error(y_true, probs, n_bins=ece_bins)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
        "nll_log_loss": float(
            log_loss(y_true, np.column_stack([1.0 - probs, probs]), labels=[0, 1])
        ),
    }

    if include_counts:
        metrics.update(counts)
        metrics.update(
            {
                "n_total": int(y_true.size),
                "n_positive": int(np.sum(y_true == 1)),
                "n_negative": int(np.sum(y_true == 0)),
                "predicted_positive_rate": float(np.mean(pred == 1)),
                "prevalence_positive": float(np.mean(y_true == 1)),
            }
        )

    return metrics


def make_bootstrap_indices(
    rng: np.random.Generator,
    y_true: np.ndarray,
    mode: str,
) -> np.ndarray:
    n = y_true.size
    if mode == "ordinary":
        return rng.integers(0, n, size=n, endpoint=False)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    sampled_pos = rng.choice(pos_idx, size=pos_idx.size, replace=True)
    sampled_neg = rng.choice(neg_idx, size=neg_idx.size, replace=True)
    sampled = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(sampled)
    return sampled


def bootstrap_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
    ece_bins: int,
    mode: str,
    progress_every: int,
) -> pd.DataFrame:
    if n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be at least 1.")

    rng = np.random.default_rng(seed)
    records: List[Dict[str, float | int]] = []
    t0 = time.time()

    for b in range(1, n_bootstrap + 1):
        sample_idx = make_bootstrap_indices(rng, y_true, mode)
        metrics = compute_metrics(
            y_true=y_true[sample_idx],
            probs=probs[sample_idx],
            pred=pred[sample_idx],
            ece_bins=ece_bins,
            include_counts=False,
        )
        metrics["bootstrap_replicate"] = b
        records.append(metrics)

        if progress_every > 0 and (b % progress_every == 0 or b == n_bootstrap):
            elapsed = time.time() - t0
            print(f"[bootstrap] {b}/{n_bootstrap} replicates completed | elapsed={elapsed:.1f}s", flush=True)

    draws = pd.DataFrame.from_records(records)
    ordered_cols = ["bootstrap_replicate"] + [m for m in METRIC_ORDER if m in draws.columns]
    draws = draws[ordered_cols]
    return draws


def summarize_bootstrap(
    point_metrics: Dict[str, float | int],
    draws: pd.DataFrame,
    ci_level: float,
) -> pd.DataFrame:
    if not (0.0 < ci_level < 1.0):
        raise ValueError("--ci-level must lie between 0 and 1.")

    alpha = 1.0 - ci_level
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    rows: List[Dict[str, float | int | str]] = []

    for metric in METRIC_ORDER:
        if metric not in draws.columns:
            continue
        values = pd.to_numeric(draws[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        rows.append(
            {
                "metric": metric,
                "metric_display": METRIC_DISPLAY_NAMES.get(metric, metric),
                "point_estimate": float(point_metrics[metric]),
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                f"ci{int(round(ci_level * 100))}_lower": float(np.quantile(values, q_low)),
                f"ci{int(round(ci_level * 100))}_upper": float(np.quantile(values, q_high)),
                "n_valid_bootstrap": int(values.size),
            }
        )

    return pd.DataFrame(rows)


def point_metrics_to_frame(point_metrics: Dict[str, float | int]) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    preferred = METRIC_ORDER + [
        "tp",
        "tn",
        "fp",
        "fn",
        "n_total",
        "n_positive",
        "n_negative",
        "prevalence_positive",
        "predicted_positive_rate",
    ]
    for key in preferred:
        if key in point_metrics:
            rows.append(
                {
                    "metric": key,
                    "metric_display": METRIC_DISPLAY_NAMES.get(key, key),
                    "value": point_metrics[key],
                }
            )
    for key, value in point_metrics.items():
        if key not in preferred:
            rows.append({"metric": key, "metric_display": key, "value": value})
    return pd.DataFrame(rows)


def verify_final_model(
    point_metrics: Dict[str, float | int],
    tol: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric, expected in LOCKED_FINAL_MODEL_METRICS.items():
        observed = float(point_metrics[metric])
        abs_diff = abs(observed - expected)
        rows.append(
            {
                "item": metric,
                "expected_locked": expected,
                "observed": observed,
                "absolute_difference": abs_diff,
                "status": "PASS" if abs_diff <= tol else "FAIL",
            }
        )
    for item, expected_count in LOCKED_FINAL_MODEL_COUNTS.items():
        observed_count = int(point_metrics[item])
        rows.append(
            {
                "item": item,
                "expected_locked": expected_count,
                "observed": observed_count,
                "absolute_difference": abs(observed_count - expected_count),
                "status": "PASS" if observed_count == expected_count else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def try_write_excel(
    out_xlsx: Path,
    point_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    draws_df: pd.DataFrame | None,
    verification_df: pd.DataFrame | None,
    config_df: pd.DataFrame,
) -> Tuple[bool, str]:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return False, "openpyxl is not installed; Excel workbook skipped."

    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            point_df.to_excel(writer, sheet_name="Point metrics", index=False)
            ci_df.to_excel(writer, sheet_name="Bootstrap CIs", index=False)
            if draws_df is not None:
                draws_df.to_excel(writer, sheet_name="Bootstrap draws", index=False)
            if verification_df is not None:
                verification_df.to_excel(writer, sheet_name="Final model verification", index=False)
            config_df.to_excel(writer, sheet_name="Run config", index=False)
        return True, f"Excel workbook written: {out_xlsx}"
    except Exception as exc:  # pragma: no cover - user machine variability
        return False, f"Excel workbook could not be written: {exc}"


def make_config_frame(config: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([{"field": str(k), "value": json.dumps(v) if isinstance(v, (dict, list)) else v} for k, v in config.items()])


def write_json(path: Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    start_time = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df, y_true, probs, pred, threshold, input_warnings = read_predictions(
        pred_csv=args.pred_csv,
        threshold_override=args.threshold,
    )

    print("[input] Prediction CSV loaded:", args.pred_csv)
    print(f"[input] Rows={len(df):,} | positives={int(np.sum(y_true == 1)):,} | negatives={int(np.sum(y_true == 0)):,}")
    if threshold is not None:
        print(f"[input] Selected threshold available/used: {threshold}")
    else:
        print("[input] No scalar threshold available; pred_selected was used exactly as exported.")
    for warning in input_warnings:
        print(f"[warning] {warning}", file=sys.stderr)

    point_metrics = compute_metrics(
        y_true=y_true,
        probs=probs,
        pred=pred,
        ece_bins=args.ece_bins,
        include_counts=True,
    )
    point_df = point_metrics_to_frame(point_metrics)

    verification_df: pd.DataFrame | None = None
    verification_status = None
    if args.verify_final_model:
        verification_df = verify_final_model(point_metrics, tol=args.verify_tol)
        verification_status = "PASS" if bool((verification_df["status"] == "PASS").all()) else "FAIL"
        print(f"[verify] Locked final analysis final 11M verification: {verification_status}")
        if verification_status != "PASS":
            failed = verification_df.loc[verification_df["status"] != "PASS", ["item", "expected_locked", "observed", "absolute_difference"]]
            print("[verify] Failed items:")
            print(failed.to_string(index=False))

    print(
        f"[bootstrap] Starting {args.n_bootstrap} {args.bootstrap_mode} bootstrap replicates "
        f"with seed={args.seed}."
    )
    draws_df = bootstrap_metrics(
        y_true=y_true,
        probs=probs,
        pred=pred,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        ece_bins=args.ece_bins,
        mode=args.bootstrap_mode,
        progress_every=args.progress_every,
    )
    ci_df = summarize_bootstrap(
        point_metrics=point_metrics,
        draws=draws_df,
        ci_level=args.ci_level,
    )

    ci_suffix = int(round(args.ci_level * 100))
    point_csv = args.outdir / "pcam_test_point_metrics.csv"
    ci_csv = args.outdir / f"pcam_test_bootstrap_ci{ci_suffix}.csv"
    draws_csv = args.outdir / "pcam_test_bootstrap_draws.csv"
    verify_csv = args.outdir / "final_model_metric_verification.csv"
    config_json = args.outdir / "bootstrap_run_config.json"
    xlsx_path = args.outdir / f"pcam_test_bootstrap_ci{ci_suffix}_workbook.xlsx"

    point_df.to_csv(point_csv, index=False)
    ci_df.to_csv(ci_csv, index=False)
    if args.save_draws:
        draws_df.to_csv(draws_csv, index=False)
    if verification_df is not None:
        verification_df.to_csv(verify_csv, index=False)

    elapsed = time.time() - start_time
    run_config: Dict[str, object] = {
        "analysis_name": "Phase 10A final 11M PathEOM bootstrap confidence intervals",
        "pred_csv": str(args.pred_csv),
        "outdir": str(args.outdir),
        "n_rows": int(len(df)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "threshold_available_or_used": threshold,
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_mode": args.bootstrap_mode,
        "seed": int(args.seed),
        "ci_level": float(args.ci_level),
        "ece_bins": int(args.ece_bins),
        "save_draws": bool(args.save_draws),
        "write_xlsx": bool(args.write_xlsx),
        "verify_final_model": bool(args.verify_final_model),
        "verification_status": verification_status,
        "input_warnings": input_warnings,
        "elapsed_seconds": float(elapsed),
        "output_files": {
            "point_metrics_csv": str(point_csv),
            "bootstrap_ci_csv": str(ci_csv),
            "bootstrap_draws_csv": str(draws_csv) if args.save_draws else None,
            "verification_csv": str(verify_csv) if verification_df is not None else None,
            "excel_workbook": str(xlsx_path) if args.write_xlsx else None,
        },
    }
    write_json(config_json, run_config)
    config_df = make_config_frame(run_config)

    if args.write_xlsx:
        wrote, message = try_write_excel(
            out_xlsx=xlsx_path,
            point_df=point_df,
            ci_df=ci_df,
            draws_df=draws_df if args.save_draws else None,
            verification_df=verification_df,
            config_df=config_df,
        )
        print(f"[xlsx] {message}")
        run_config["excel_write_success"] = wrote
        run_config["excel_write_message"] = message
        write_json(config_json, run_config)

    print("[done] Point metrics CSV:", point_csv)
    print("[done] Bootstrap CI CSV:", ci_csv)
    if args.save_draws:
        print("[done] Bootstrap draws CSV:", draws_csv)
    if verification_df is not None:
        print("[done] Locked Phase 9 verification CSV:", verify_csv)
    print("[done] Run config JSON:", config_json)
    print(f"[done] Total elapsed time: {elapsed:.1f}s")

    # Return nonzero only when explicit locked verification was requested and failed.
    if args.verify_final_model and verification_status != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





