#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a prediction or metadata CSV by CAMELYON17 center.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--center", type=int, required=True)
    parser.add_argument("--center-col", default="center")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.center_col not in frame.columns:
        raise ValueError(f"Column '{args.center_col}' not found. Available columns: {list(frame.columns)}")
    filtered = frame[frame[args.center_col].astype(int) == args.center].copy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.out, index=False)
    print(f"Wrote {len(filtered):,} rows to {args.out}")


if __name__ == "__main__":
    main()
