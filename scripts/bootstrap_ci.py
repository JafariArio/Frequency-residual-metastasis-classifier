#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from freqres_pathology.eval.bootstrap import stratified_bootstrap_ci
from freqres_pathology.eval.io import load_prediction_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stratified bootstrap confidence intervals.")
    parser.add_argument("--input", required=True, help="Prediction CSV file")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    parser.add_argument("--boot-out", default=None)
    parser.add_argument("--y-true-col", default=None)
    parser.add_argument("--y-score-col", default=None)
    args = parser.parse_args()

    frame, label_col, score_col = load_prediction_csv(args.input, args.y_true_col, args.y_score_col)
    intervals, draws = stratified_bootstrap_ci(
        frame[label_col],
        frame[score_col],
        threshold=args.threshold,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(args.out, index=False)
    if args.boot_out:
        Path(args.boot_out).parent.mkdir(parents=True, exist_ok=True)
        draws.to_csv(args.boot_out, index=False)
    print(intervals.to_string(index=False))


if __name__ == "__main__":
    main()
