#!/usr/bin/env python
r"""
Phase 9B v2 — SI statistical figures and gallery panels for the final 11M PathEOM-Net-R analysis.

This script reads Phase 9A v2 outputs and creates:
- category-count figure
- probability distribution figure
- attribution-statistics figures
- flattened summary tables
- curated SI gallery pages from saved gallery candidates

It is designed to operate on all-test-patch attribution statistics while
avoiding the impractical requirement to print or archive tens of thousands of image panels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def figure_counts(counts: pd.DataFrame, out_png: Path):
    fig = plt.figure(figsize=(5.5, 4.0))
    ax = plt.gca()
    ax.bar(counts["category"].astype(str), counts["count"].astype(float))
    ax.set_title("Prediction-outcome category counts")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    for x, y in zip(range(len(counts)), counts["count"]):
        ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def figure_probabilities(pred_df: pd.DataFrame, out_png: Path):
    order = ["TP", "TN", "FP", "FN"]
    data = [pred_df.loc[pred_df["category"] == c, "probability"].astype(float).to_numpy() for c in order]
    fig = plt.figure(figsize=(6.8, 4.4))
    ax = plt.gca()
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title("Predicted probability distribution by category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Predicted probability for positive class")
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def figure_attr_metric(attr_df: pd.DataFrame, metric: str, ylabel: str, title: str, out_png: Path):
    order = ["TP", "TN", "FP", "FN"]
    vals = []
    for c in order:
        sub = attr_df[attr_df["category"] == c]
        vals.append(float(sub[metric].mean()) if len(sub) else np.nan)
    fig = plt.figure(figsize=(6.0, 4.2))
    ax = plt.gca()
    ax.bar(order, vals)
    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def flatten_summary(df: pd.DataFrame, group_col: str, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for cat, sub in df.groupby(group_col):
        row = {group_col: cat, "n_maps_analyzed": int(len(sub))}
        for metric in metrics:
            vals = sub[metric].astype(float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            row[f"{metric}_median"] = float(vals.median())
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary_json(pred_df: pd.DataFrame, attr_df: pd.DataFrame, out_json: Path):
    summary = {}
    for c in ["TP", "TN", "FP", "FN"]:
        ps = pred_df[pred_df["category"] == c]
        ats = attr_df[attr_df["category"] == c]
        summary[c] = {
            "n_predictions_total": int(len(ps)),
            "n_maps_analyzed": int(len(ats)),
            "probability_mean": float(ps["probability"].mean()) if len(ps) else None,
            "probability_std": float(ps["probability"].std()) if len(ps) > 1 else None,
            "probability_median": float(ps["probability"].median()) if len(ps) else None,
            "saliency_mean": float(ats["saliency_mean"].mean()) if len(ats) else None,
            "hotspot_frac_gt_075_mean": float(ats["hotspot_frac_gt_075"].mean()) if len(ats) else None,
            "center_saliency_mean": float(ats["center_saliency_mean"].mean()) if len(ats) else None,
            "border_saliency_mean": float(ats["border_saliency_mean"].mean()) if len(ats) else None,
        }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def render_gallery(images: List[Path], titles: List[str], out_png: Path, ncols: int = 4, title: str = ""):
    if not images:
        return
    n = len(images)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.1 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, imgp, ttl in zip(axes.ravel(), images, titles):
        img = imread(imgp)
        ax.imshow(img)
        ax.set_title(ttl, fontsize=8)
        ax.axis("off")
    plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.08)
    if title:
        fig.suptitle(title, fontsize=13, weight="bold")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution-dir", required=True, help="Output directory produced by the PCam attribution analysis.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gallery-page-size", type=int, default=16)
    ap.add_argument("--gallery-ncols", type=int, default=4)
    args = ap.parse_args()

    attribution_dir = Path(args.attribution_dir)
    outdir = Path(args.outdir)
    fig_dir = outdir / "summary_figures"
    gal_dir = outdir / "si_galleries"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gal_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = attribution_dir / "predictions" / "final11m_test_predictions.csv"
    attr_csv = attribution_dir / "summary_stats" / "all_map_statistics.csv"
    gallery_csv = attribution_dir / "summary_stats" / "gallery_candidates.csv"

    if not pred_csv.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {pred_csv}")
    if not attr_csv.exists():
        raise FileNotFoundError(f"Missing all-map statistics CSV: {attr_csv}. Run Phase 9A v2 with --generate-all-map-statistics.")

    pred_df = pd.read_csv(pred_csv)
    attr_df = pd.read_csv(attr_csv)
    counts = pred_df.groupby("category").size().reset_index(name="count")
    prob_summary = pred_df.groupby("category")["probability"].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()

    metrics = [
        "saliency_mean", "saliency_std", "saliency_max",
        "hotspot_frac_gt_050", "hotspot_frac_gt_075",
        "center_saliency_mean", "border_saliency_mean"
    ]
    attr_summary = flatten_summary(attr_df, "category", metrics)

    counts.to_csv(outdir / "si_category_counts.csv", index=False)
    prob_summary.to_csv(outdir / "si_probability_summary_by_category.csv", index=False)
    attr_summary.to_csv(outdir / "si_attribution_summary_by_category.csv", index=False)
    write_summary_json(pred_df, attr_df, outdir / "si_map_statistics_summary.json")

    figure_counts(counts, fig_dir / "figure_si_category_counts.png")
    figure_probabilities(pred_df, fig_dir / "figure_si_probability_boxplot.png")
    figure_attr_metric(attr_df, "saliency_mean", "Mean saliency", "Mean attribution strength by category", fig_dir / "figure_si_attribution_mean_by_category.png")
    figure_attr_metric(attr_df, "hotspot_frac_gt_075", "Mean fraction of pixels with saliency >= 0.75", "High-attribution hotspot fraction by category", fig_dir / "figure_si_hotspot_fraction_by_category.png")
    figure_attr_metric(attr_df, "center_saliency_mean", "Mean center saliency", "Central attribution strength by category", fig_dir / "figure_si_center_saliency_by_category.png")
    figure_attr_metric(attr_df, "border_saliency_mean", "Mean border saliency", "Border attribution strength by category", fig_dir / "figure_si_border_saliency_by_category.png")

    if gallery_csv.exists():
        gallery_df = pd.read_csv(gallery_csv)
        for cat in ["TP", "TN", "FP", "FN"]:
            sub = gallery_df[gallery_df["category"] == cat].copy()
            if len(sub) == 0:
                continue
            images = [Path(p) for p in sub["map_png"].astype(str).tolist() if len(str(p))]
            titles = [f"idx={int(i)} | p={float(p):.3f}" for i, p in zip(sub["index"].tolist(), sub["probability"].tolist())]
            page_num = 1
            for img_chunk, ttl_chunk in zip(chunked(images, args.gallery_page_size), chunked(titles, args.gallery_page_size)):
                out_png = gal_dir / f"gallery_{cat}_page{page_num:03d}.png"
                render_gallery(img_chunk, ttl_chunk, out_png, ncols=args.gallery_ncols, title=f"{cat} curated SI gallery page {page_num}")
                page_num += 1

    manifest = {
        "version": "pcam_attribution_galleries_v1",
        "attribution_dir": str(attribution_dir),
        "gallery_page_size": int(args.gallery_page_size),
        "gallery_ncols": int(args.gallery_ncols),
        "n_prediction_rows": int(len(pred_df)),
        "n_map_stat_rows": int(len(attr_df)),
        "key_outputs": {
            "summary_figures_dir": str(fig_dir),
            "si_galleries_dir": str(gal_dir),
            "si_map_statistics_summary_json": str(outdir / "si_map_statistics_summary.json"),
            "si_attribution_summary_csv": str(outdir / "si_attribution_summary_by_category.csv"),
        },
        "important_note": "Attribution statistics are based on maps analyzed by Phase 9A v2; probability/category counts are based on all test predictions."
    }
    (outdir / "pcam_attribution_gallery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


