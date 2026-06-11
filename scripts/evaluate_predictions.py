#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from freqres_pathology.eval.io import load_prediction_csv
from freqres_pathology.eval.metrics import binary_metrics, metrics_to_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary prediction scores at a fixed threshold.")
    parser.add_argument("--input", required=True, help="Prediction CSV file")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--y-true-col", default=None)
    parser.add_argument("--y-score-col", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, label_col, score_col = load_prediction_csv(args.input, args.y_true_col, args.y_score_col)
    metrics = binary_metrics(frame[label_col], frame[score_col], threshold=args.threshold)
    metrics_to_frame(metrics).to_csv(out_dir / "metrics.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
