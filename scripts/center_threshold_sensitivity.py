#!/usr/bin/env python
"""
Center-specific threshold sensitivity analysis without external dependencies.

This script reads a prediction CSV with binary labels and positive-class scores,
sweeps decision thresholds, and reports operating-point metrics. It uses only
the Python standard library, so numpy/pandas are not required.

Example
-------
python scripts/center_threshold_sensitivity_stdlib.py ^
  --input data/center1/phase17c_center1_true_ood_validation_predictions.csv ^
  --out-dir outputs/center1_threshold_sensitivity ^
  --fixed-threshold 0.220 ^
  --target-specificity 0.91
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


LABEL_CANDIDATES = [
    "y_true", "true_label", "label", "target", "class", "ground_truth",
    "gt", "metastasis", "is_metastasis", "y"
]

SCORE_CANDIDATES = [
    "y_score", "score", "prob", "probability", "positive_probability",
    "p_positive", "p_pos", "pred_prob", "prediction_probability",
    "metastasis_probability", "prob_pos", "pred_score", "logit"
]


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {c.lower().strip(): c for c in columns}
    for name in candidates:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def parse_label(value: str) -> int:
    text = str(value).strip().lower()
    mapping = {
        "0": 0, "1": 1,
        "0.0": 0, "1.0": 1,
        "false": 0, "true": 1,
        "negative": 0, "positive": 1,
        "neg": 0, "pos": 1,
        "normal": 0, "tumor": 1,
        "non-metastatic": 0, "metastatic": 1,
        "non_metastatic": 0,
    }
    if text in mapping:
        return mapping[text]
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"Could not convert label value to binary label: {value!r}")
    if number == 0:
        return 0
    if number == 1:
        return 1
    raise ValueError(f"Label must be 0/1. Found: {value!r}")


def logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_predictions(path: Path, label_col: Optional[str], score_col: Optional[str]) -> Tuple[List[int], List[float], str, str]:
    y_true: List[int] = []
    scores_raw: List[float] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        columns = list(reader.fieldnames)
        label_name = label_col or find_column(columns, LABEL_CANDIDATES)
        score_name = score_col or find_column(columns, SCORE_CANDIDATES)

        if label_name is None:
            raise ValueError(f"Could not auto-detect label column. Available columns: {columns}. Use --label-col.")
        if score_name is None:
            raise ValueError(f"Could not auto-detect score column. Available columns: {columns}. Use --score-col.")

        for row in reader:
            label_value = row.get(label_name, "")
            score_value = row.get(score_name, "")
            if score_value is None or str(score_value).strip() == "":
                continue
            y_true.append(parse_label(label_value))
            scores_raw.append(float(score_value))

    if not y_true:
        raise ValueError("No valid predictions were loaded.")

    score_lower = score_name.lower()
    min_score = min(scores_raw)
    max_score = max(scores_raw)
    if "logit" in score_lower or min_score < 0.0 or max_score > 1.0:
        scores = [logistic(x) for x in scores_raw]
    else:
        scores = scores_raw

    if min(scores) < 0.0 or max(scores) > 1.0:
        raise ValueError("Score values must be probabilities in [0, 1] or logits.")

    return y_true, scores, label_name, score_name


def metrics_at_threshold(y_true: List[int], scores: List[float], threshold: float) -> Dict[str, float]:
    tp = tn = fp = fn = 0

    for y, s in zip(y_true, scores):
        pred = 1 if s >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    accuracy = (tp + tn) / len(y_true) if y_true else float("nan")
    balanced_accuracy = (sensitivity + specificity) / 2.0

    if math.isnan(precision) or math.isnan(sensitivity) or (precision + sensitivity) == 0:
        f1 = float("nan")
    else:
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": float(precision),
        "npv": float(npv),
        "positive_f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def make_threshold_grid(scores: List[float], n_grid: int) -> List[float]:
    uniform = [i / (n_grid - 1) for i in range(n_grid)] if n_grid > 1 else [0.0, 1.0]
    observed = sorted(set(scores))

    if len(observed) > 5000:
        step = (len(observed) - 1) / 4999
        observed = [observed[round(i * step)] for i in range(5000)]

    grid = sorted(set(max(0.0, min(1.0, x)) for x in (uniform + observed)))
    return grid


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = [
        "threshold", "accuracy", "balanced_accuracy", "sensitivity", "specificity",
        "precision_ppv", "npv", "positive_f1", "tp", "tn", "fp", "fn"
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold-sensitivity analysis for a held-out center.")
    parser.add_argument("--input", required=True, help="Input prediction CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--label-col", default=None, help="Binary label column. Auto-detected if omitted.")
    parser.add_argument("--score-col", default=None, help="Positive-class score/probability column. Auto-detected if omitted.")
    parser.add_argument("--fixed-threshold", type=float, default=0.220, help="Original fixed threshold.")
    parser.add_argument("--target-specificity", type=float, default=0.91, help="Specificity target to recover.")
    parser.add_argument("--n-grid", type=int, default=1001, help="Uniform threshold grid size.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true, scores, label_col, score_col = load_predictions(input_path, args.label_col, args.score_col)

    grid = make_threshold_grid(scores, args.n_grid)
    sweep = [metrics_at_threshold(y_true, scores, t) for t in grid]
    fixed = metrics_at_threshold(y_true, scores, args.fixed_threshold)

    eligible = [row for row in sweep if row["specificity"] >= args.target_specificity]
    if eligible:
        best = sorted(
            eligible,
            key=lambda r: (r["balanced_accuracy"], r["sensitivity"], r["specificity"]),
            reverse=True,
        )[0]
    else:
        best = sorted(
            sweep,
            key=lambda r: (r["specificity"], r["balanced_accuracy"]),
            reverse=True,
        )[0]

    sweep_path = out_dir / "center1_threshold_sensitivity.csv"
    summary_path = out_dir / "center1_threshold_sensitivity_summary.json"
    fixed_path = out_dir / "center1_fixed_threshold_metrics.csv"
    best_path = out_dir / "center1_specificity_constrained_threshold_metrics.csv"

    write_csv(sweep_path, sweep)
    write_csv(fixed_path, [fixed])
    write_csv(best_path, [best])

    summary = {
        "input_file": str(input_path),
        "n_samples": int(len(y_true)),
        "n_positive": int(sum(1 for y in y_true if y == 1)),
        "n_negative": int(sum(1 for y in y_true if y == 0)),
        "label_column": label_col,
        "score_column": score_col,
        "fixed_threshold": fixed,
        "target_specificity": float(args.target_specificity),
        "best_threshold_at_or_above_target_specificity": best,
        "note": (
            "This is a diagnostic threshold-sensitivity analysis on frozen prediction scores. "
            "It does not train, recalibrate, or modify the model."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Completed threshold-sensitivity analysis.")
    print(f"Input: {input_path}")
    print(f"Detected label column: {label_col}")
    print(f"Detected score column: {score_col}")
    print(f"Fixed threshold metrics: {fixed_path}")
    print(f"Threshold sweep: {sweep_path}")
    print(f"Specificity-constrained threshold metrics: {best_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
