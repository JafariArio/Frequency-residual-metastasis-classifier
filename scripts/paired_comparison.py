#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from freqres_pathology.eval.io import ID_CANDIDATES, infer_column, load_prediction_csv
from freqres_pathology.eval.metrics import binary_metrics
from freqres_pathology.eval.paired import mcnemar_continuity


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare aligned prediction CSV files at fixed thresholds.")
    parser.add_argument("--final", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--final-threshold", type=float, required=True)
    parser.add_argument("--comparator-threshold", type=float, required=True)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    final_frame, final_label, final_score = load_prediction_csv(args.final)
    comp_frame, comp_label, comp_score = load_prediction_csv(args.comparator)
    id_col = args.id_col or infer_column(final_frame.columns, ID_CANDIDATES)

    if id_col and id_col in comp_frame.columns:
        merged = final_frame[[id_col, final_label, final_score]].merge(
            comp_frame[[id_col, comp_label, comp_score]],
            on=id_col,
            suffixes=("_final", "_comp"),
        )
        y_true = merged[f"{final_label}_final"].astype(int).to_numpy()
        final_scores = merged[f"{final_score}_final"].astype(float).to_numpy()
        comp_scores = merged[f"{comp_score}_comp"].astype(float).to_numpy()
    else:
        if len(final_frame) != len(comp_frame):
            raise ValueError("No shared ID column found and row counts differ.")
        y_true = final_frame[final_label].astype(int).to_numpy()
        final_scores = final_frame[final_score].astype(float).to_numpy()
        comp_scores = comp_frame[comp_score].astype(float).to_numpy()

    final_metrics = binary_metrics(y_true, final_scores, args.final_threshold)
    comp_metrics = binary_metrics(y_true, comp_scores, args.comparator_threshold)
    metric_names = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "positive_f1", "mcc"]
    rows = [
        {
            "metric": name,
            "final": final_metrics[name],
            "comparator": comp_metrics[name],
            "delta_final_minus_comparator": final_metrics[name] - comp_metrics[name],
        }
        for name in metric_names
    ]

    final_pred = (final_scores >= args.final_threshold).astype(int)
    comp_pred = (comp_scores >= args.comparator_threshold).astype(int)
    test = mcnemar_continuity(y_true, final_pred, comp_pred)
    output = pd.DataFrame(rows)
    for key, value in test.items():
        output[key] = value

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
