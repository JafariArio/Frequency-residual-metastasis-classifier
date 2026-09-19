#!/usr/bin/env python
r"""
Phase 15B — SI-style summary figures, tables, and gallery pages for the
CAMELYON17-WILDS full-test external attribution/map audit.

Reads Phase 15A outputs and creates:
- category-count figure,
- probability distribution by TP/TN/FP/FN,
- attribution-statistics summary figures,
- flattened summary tables,
- multi-panel gallery pages from Phase 15A curated map candidates.

Scientific boundary:
These are patch-level attribution-distribution summaries and curated map galleries.
They are not segmentation masks or pixel-level tumor-probability outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread


def status(msg: str) -> None:
    print(msg, flush=True)


def chunked(seq: Sequence[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def figure_counts(counts: pd.DataFrame, out_png: Path) -> None:
    fig = plt.figure(figsize=(5.5, 4.0))
    ax = plt.gca()
    ax.bar(counts["category"].astype(str), counts["count"].astype(float))
    ax.set_title("CAMELYON17-WILDS transfer outcome-category counts")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    for x, y in zip(range(len(counts)), counts["count"]):
        ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def figure_probabilities(pred_df: pd.DataFrame, out_png: Path) -> None:
    order = ["TP", "TN", "FP", "FN"]
    data = [pred_df.loc[pred_df["category"] == cat, "probability"].astype(float).to_numpy() for cat in order]
    fig = plt.figure(figsize=(6.8, 4.4))
    ax = plt.gca()
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title("External transfer predicted-probability distribution by category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Predicted probability for positive class")
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def figure_attr_metric(attr_df: pd.DataFrame, metric: str, ylabel: str, title: str, out_png: Path) -> None:
    order = ["TP", "TN", "FP", "FN"]
    data = [attr_df.loc[attr_df["category"] == cat, metric].astype(float).to_numpy() for cat in order]
    fig = plt.figure(figsize=(6.8, 4.4))
    ax = plt.gca()
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def figure_center_vs_border(attr_df: pd.DataFrame, out_png: Path) -> None:
    order = ["TP", "TN", "FP", "FN"]
    centers = [attr_df.loc[attr_df["category"] == cat, "center_saliency_mean"].astype(float).mean() for cat in order]
    borders = [attr_df.loc[attr_df["category"] == cat, "border_saliency_mean"].astype(float).mean() for cat in order]
    x = np.arange(len(order))
    width = 0.36
    fig = plt.figure(figsize=(7.0, 4.4))
    ax = plt.gca()
    ax.bar(x - width / 2, centers, width, label="Center")
    ax.bar(x + width / 2, borders, width, label="Border")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_title("External transfer center versus border attribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Mean normalized attribution strength")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def flatten_summary(attr_df: pd.DataFrame, group_col: str, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for group, sub in attr_df.groupby(group_col):
        row = {group_col: group, "n_maps_analyzed": int(len(sub))}
        for metric in metrics:
            vals = sub[metric].astype(float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            row[f"{metric}_median"] = float(vals.median())
        rows.append(row)
    return pd.DataFrame(rows)


def make_gallery_pages(gallery_df: pd.DataFrame, gallery_dir: Path, out_dir: Path, page_size: int, ncols: int) -> pd.DataFrame:
    if gallery_df.empty:
        return pd.DataFrame()
    page_rows = []
    entries = gallery_df.to_dict(orient="records")
    ncols = max(1, int(ncols))
    page_size = max(1, int(page_size))
    for page_idx, page_entries in enumerate(chunked(entries, page_size), start=1):
        n = len(page_entries)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        for ax in axes.ravel():
            ax.axis("off")
        for i, entry in enumerate(page_entries):
            ax = axes.ravel()[i]
            image_path = Path(entry["map_png"])
            if not image_path.exists():
                candidate = gallery_dir / str(entry["category"]) / image_path.name
                image_path = candidate
            img = imread(image_path)
            ax.imshow(img)
            cat = entry.get("category", "")
            idx = int(entry.get("index", -1))
            prob = float(entry.get("probability", float("nan")))
            ax.set_title(f"{cat} idx={idx} p={prob:.3f}", fontsize=8)
            ax.axis("off")
        fig.suptitle("CAMELYON17-WILDS external curated attribution gallery", fontsize=13, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_png = out_dir / f"external_gallery_page_{page_idx:02d}.png"
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        page_rows.append({"page": page_idx, "n_maps": n, "gallery_page_png": str(out_png)})
    return pd.DataFrame(page_rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 15B SI-style summary figures for CAMELYON17-WILDS external attribution audit."
    )
    ap.add_argument(
        "--attribution-dir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "wilds_attribution"),
        help="Output directory produced by Phase 15A.",
    )
    ap.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "wilds_attribution_galleries"),
        help="Phase 15B output directory.",
    )
    ap.add_argument("--gallery-page-size", type=int, default=16)
    ap.add_argument("--gallery-ncols", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    attribution_dir = Path(args.attribution_dir)
    outdir = Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is non-empty: {outdir}. Use --overwrite or choose another folder.")

    fig_dir = outdir / "summary_figures"
    gal_page_dir = outdir / "si_gallery_pages"
    table_dir = outdir / "summary_tables"
    for d in [outdir, fig_dir, gal_page_dir, table_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pred_csv = attribution_dir / "predictions" / "wilds_full_test_predictions_aligned.csv"
    attr_csv = attribution_dir / "summary_stats" / "all_external_map_statistics.csv"
    gallery_csv = attribution_dir / "summary_stats" / "external_gallery_candidates.csv"
    gallery_dir = attribution_dir / "gallery_candidate_maps"

    if not pred_csv.exists():
        raise FileNotFoundError(f"Missing Phase 15A prediction CSV: {pred_csv}")
    if not attr_csv.exists():
        raise FileNotFoundError(
            f"Missing Phase 15A all-map statistics CSV: {attr_csv}. "
            "Run Phase 15A with --generate-all-map-statistics."
        )

    pred_df = pd.read_csv(pred_csv)
    attr_df = pd.read_csv(attr_csv)
    gallery_df = pd.read_csv(gallery_csv) if gallery_csv.exists() else pd.DataFrame()

    counts = pred_df.groupby("category").size().reset_index(name="count")
    prob_summary = pred_df.groupby("category")["probability"].agg(
        ["count", "mean", "std", "median", "min", "max"]
    ).reset_index()

    metrics = [
        "saliency_mean",
        "saliency_std",
        "saliency_max",
        "hotspot_frac_gt_050",
        "hotspot_frac_gt_075",
        "center_saliency_mean",
        "border_saliency_mean",
        "abs_probability_difference",
    ]
    attr_summary = flatten_summary(attr_df, "category", metrics)

    counts.to_csv(table_dir / "external_category_counts.csv", index=False)
    prob_summary.to_csv(table_dir / "external_probability_summary_by_category.csv", index=False)
    attr_summary.to_csv(table_dir / "external_attribution_summary_by_category.csv", index=False)

    figure_counts(counts, fig_dir / "external_outcome_category_counts.png")
    figure_probabilities(pred_df, fig_dir / "external_probability_distribution_by_category.png")
    figure_attr_metric(
        attr_df,
        "saliency_mean",
        "Mean normalized attribution strength",
        "External transfer attribution strength by outcome category",
        fig_dir / "external_attribution_strength_by_category.png",
    )
    figure_attr_metric(
        attr_df,
        "hotspot_frac_gt_050",
        "Fraction of pixels with normalized attribution >= 0.50",
        "External transfer attribution hotspot fraction by outcome category",
        fig_dir / "external_hotspot_fraction_by_category.png",
    )
    figure_center_vs_border(attr_df, fig_dir / "external_center_vs_border_attribution.png")

    pages_df = make_gallery_pages(
        gallery_df,
        gallery_dir,
        gal_page_dir,
        args.gallery_page_size,
        args.gallery_ncols,
    )
    pages_df.to_csv(table_dir / "external_gallery_pages_manifest.csv", index=False)

    manifest = {
        "phase": "Phase 15B CAMELYON17-WILDS external attribution SI figures and galleries",
        "status": "COMPLETED",
        "attribution_dir": str(attribution_dir),
        "outdir": str(outdir),
        "n_predictions": int(len(pred_df)),
        "n_attribution_statistics": int(len(attr_df)),
        "n_gallery_candidates": int(len(gallery_df)),
        "n_gallery_pages": int(len(pages_df)),
        "outputs": {
            "summary_figures_dir": str(fig_dir),
            "si_gallery_pages_dir": str(gal_page_dir),
            "summary_tables_dir": str(table_dir),
        },
        "scientific_boundary": (
            "Patch-level attribution-distribution summaries only; not segmentation or whole-slide localization."
        ),
    }
    (outdir / "wilds_attribution_gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    status(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

