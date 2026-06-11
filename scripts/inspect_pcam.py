#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from freqres_pathology.data import PCamH5Dataset, compute_mean_std


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PCam HDF5 splits and estimate channel statistics.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default="outputs/pcam_summary.csv")
    parser.add_argument("--max-items", type=int, default=2048)
    args = parser.parse_args()

    rows = []
    for split in ["train", "valid", "test"]:
        dataset = PCamH5Dataset(args.data_root, split=split, transform=None)
        mean, std = compute_mean_std(dataset, max_items=args.max_items)
        rows.append(
            {
                "split": split,
                "n": len(dataset),
                "mean_r": mean[0],
                "mean_g": mean[1],
                "mean_b": mean[2],
                "std_r": std[0],
                "std_g": std[1],
                "std_b": std[2],
            }
        )
        dataset.close()

    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
