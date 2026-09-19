#!/usr/bin/env python
r"""
Phase 15A — CAMELYON17-WILDS full-test external attribution/map audit.

Purpose
-------
Generate an external-dataset attribution audit for the frozen final 11M
PathEOM-Net-R model on the full CAMELYON17-WILDS test split already evaluated
in Phase 13D.

This script is deliberately aligned with the Phase 9 final-PCam attribution audit:
- main TP/TN/FP/FN raw + attribution panel,
- optional all-test attribution statistics,
- category-balanced curated attribution galleries,
- optional all-map PNG export (disabled by default).

Inputs
------
1. Full CAMELYON17-WILDS test HDF5 files produced by Phase 13C:
   <path-to-prepared-CAMELYON17-WILDS-full-test-data>

2. Frozen final-model Phase 13D predictions:
   <repository>/predictions/wilds_full_test/
     test_predictions.csv

3. Final 11M model reconstruction assets:
   - src/model/train_final_model.py
   - best_validation_constraint.pt

Scientific boundary
-------------------
Attribution maps are patch-level local model-sensitivity maps.
They are NOT segmentation masks, NOT pixel-level metastatic-probability maps,
and NOT whole-slide localization outputs.

Default visual scale-bar behavior
---------------------------------
Physical scale bars are disabled by default (--mpp 0), because this script is
written to avoid silently asserting a physical pixel calibration. A scale bar
can be enabled intentionally with a positive --mpp value.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


# ============================================================
# General helpers
# ============================================================
def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def status(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k)) for k in fields})


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if len(arr) else float("nan")


# ============================================================
# CLI and paths
# ============================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="External attribution analysis for the CAMELYON17-WILDS full test split."
    )
    ap.add_argument(
        "--subset-dir",
        required=True,
        help="Directory containing the prepared CAMELYON17-WILDS full-test HDF5 files.",
    )
    ap.add_argument(
        "--x-h5",
        default="",
        help="Optional explicit image HDF5 path. Defaults to subset-dir/camelyon17_wilds_full_test_x.h5.",
    )
    ap.add_argument(
        "--y-h5",
        default="",
        help="Optional explicit label HDF5 path. Defaults to subset-dir/camelyon17_wilds_full_test_y.h5.",
    )
    ap.add_argument(
        "--predictions-csv",
        default=str(Path(__file__).resolve().parents[2] / "predictions" / "wilds_full_test" / "test_predictions.csv"),
        help="Frozen final-model CAMELYON17-WILDS full-test prediction CSV.",
    )
    ap.add_argument(
        "--train-script",
        default=str(Path(__file__).resolve().parents[2] / "src" / "model" / "train_final_model.py"),
        help="Final 10.96M training/model script used to reconstruct the model.",
    )
    ap.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "final_model" / "best_validation_constraint.pt"),
        help="Final 11M model checkpoint.",
    )
    ap.add_argument("--variant", default="p7_r2_sens_reg_11m")
    ap.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "wilds_attribution"),
        help="Output folder for maps, statistics, panels, and manifest.",
    )
    ap.add_argument("--threshold", type=float, default=0.220)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--main-n-each", type=int, default=1)
    ap.add_argument("--main-smoothgrad-n", type=int, default=16)
    ap.add_argument("--main-noise-std", type=float, default=0.03)
    ap.add_argument("--all-attribution-batch-size", type=int, default=16)
    ap.add_argument("--all-noise-std", type=float, default=0.00)
    ap.add_argument("--generate-all-map-statistics", action="store_true")
    ap.add_argument(
        "--max-all-maps",
        type=int,
        default=-1,
        help="Optional debugging cap for all-map statistics. -1 means all external test patches.",
    )
    ap.add_argument(
        "--save-gallery-per-category",
        type=int,
        default=64,
        help="Curated SI gallery candidate maps per TP/TN/FP/FN category.",
    )
    ap.add_argument(
        "--save-all-map-pngs",
        action="store_true",
        help="Storage-heavy option: save every analyzed map as a PNG. Disabled by default.",
    )
    ap.add_argument("--mpp", type=float, default=0.0, help="Microns per pixel. <=0 disables scale bars.")
    ap.add_argument("--scalebar-um", type=float, default=25.0)
    ap.add_argument("--bg-opacity", type=float, default=0.22)
    ap.add_argument("--low-alpha", type=float, default=0.16)
    ap.add_argument("--max-alpha", type=float, default=0.88)
    ap.add_argument("--saliency-floor", type=float, default=0.00)
    ap.add_argument("--alpha-gamma", type=float, default=1.0)
    ap.add_argument("--saliency-cap", type=float, default=0.98)
    ap.add_argument("--show-low-background", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    subset_dir = Path(args.subset_dir)
    x_h5 = Path(args.x_h5) if args.x_h5 else subset_dir / "camelyon17_wilds_full_test_x.h5"
    y_h5 = Path(args.y_h5) if args.y_h5 else subset_dir / "camelyon17_wilds_full_test_y.h5"
    paths = {
        "subset_dir": subset_dir,
        "x_h5": x_h5,
        "y_h5": y_h5,
        "predictions_csv": Path(args.predictions_csv),
        "train_script": Path(args.train_script),
        "checkpoint": Path(args.checkpoint),
        "outdir": Path(args.outdir),
    }
    for key in ["x_h5", "y_h5", "predictions_csv", "train_script", "checkpoint"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Required path not found ({key}): {paths[key]}")
    return paths


def prepare_dirs(outdir: Path, overwrite: bool) -> Dict[str, Path]:
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {outdir}\n"
            "Use a new --outdir or pass --overwrite intentionally."
        )
    pred_dir = outdir / "predictions"
    main_dir = outdir / "main_external_attribution_panel"
    stats_dir = outdir / "summary_stats"
    gallery_dir = outdir / "gallery_candidate_maps"
    all_png_dir = outdir / "all_map_pngs"
    for d in [outdir, pred_dir, main_dir, stats_dir, gallery_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for cat in ["TP", "TN", "FP", "FN"]:
        (gallery_dir / cat).mkdir(parents=True, exist_ok=True)
        (all_png_dir / cat).mkdir(parents=True, exist_ok=True)
    return {
        "outdir": outdir,
        "pred_dir": pred_dir,
        "main_dir": main_dir,
        "stats_dir": stats_dir,
        "gallery_dir": gallery_dir,
        "all_png_dir": all_png_dir,
    }


# ============================================================
# CAMELYON17-WILDS HDF5 reader
# ============================================================
def first_dataset_key(h5: h5py.File, preferred: Sequence[str]) -> str:
    for k in preferred:
        if k in h5:
            return k
    keys = list(h5.keys())
    if not keys:
        raise ValueError("HDF5 file has no dataset keys.")
    return keys[0]


class Camelyon17WildsFullReader:
    def __init__(self, x_h5: Path, y_h5: Path):
        self.x_path = x_h5
        self.y_path = y_h5
        self.fx = h5py.File(self.x_path, "r")
        self.fy = h5py.File(self.y_path, "r")
        self.x_key = first_dataset_key(self.fx, ["x", "X", "images", "data"])
        self.y_key = first_dataset_key(self.fy, ["y", "Y", "labels"])
        self.x_ds = self.fx[self.x_key]
        self.y_ds = self.fy[self.y_key]
        self.n = int(self.x_ds.shape[0])
        if int(self.y_ds.shape[0]) != self.n:
            raise ValueError(f"x/y row count mismatch: x={self.n}, y={self.y_ds.shape[0]}")

    def __len__(self) -> int:
        return self.n

    def labels_all(self) -> np.ndarray:
        return np.asarray(self.y_ds).reshape(-1).astype(np.int64)

    def get(self, idx: int) -> Tuple[np.ndarray, int]:
        x = np.asarray(self.x_ds[int(idx)])
        y = int(np.asarray(self.y_ds[int(idx)]).reshape(-1)[0])
        if x.ndim == 3 and x.shape[-1] == 3:
            x = np.transpose(x, (2, 0, 1))
        x = np.ascontiguousarray(np.clip(x.astype(np.float32) / 255.0, 0.0, 1.0))
        return x, y

    def get_batch(self, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for idx in indices:
            x, y = self.get(int(idx))
            xs.append(x)
            ys.append(y)
        return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)

    def close(self) -> None:
        try:
            self.fx.close()
        except Exception:
            pass
        try:
            self.fy.close()
        except Exception:
            pass


# ============================================================
# Prediction-table loading and validation
# ============================================================
def label_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    conds = [
        (out.y_true == 1) & (out.pred_selected == 1),
        (out.y_true == 0) & (out.pred_selected == 0),
        (out.y_true == 0) & (out.pred_selected == 1),
        (out.y_true == 1) & (out.pred_selected == 0),
    ]
    out["category"] = np.select(conds, ["TP", "TN", "FP", "FN"], default="OTHER")
    return out


def load_external_predictions(path: Path, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    if "subset_index" in df.columns and "index" not in df.columns:
        rename["subset_index"] = "index"
    if "idx" in df.columns and "index" not in df.columns:
        rename["idx"] = "index"
    if "label" in df.columns and "y_true" not in df.columns:
        rename["label"] = "y_true"
    if "prob" in df.columns and "probability" not in df.columns:
        rename["prob"] = "probability"
    if "y_prob" in df.columns and "probability" not in df.columns:
        rename["y_prob"] = "probability"
    if "pred" in df.columns and "pred_selected" not in df.columns:
        rename["pred"] = "pred_selected"
    if "predicted_label" in df.columns and "pred_selected" not in df.columns:
        rename["predicted_label"] = "pred_selected"
    df = df.rename(columns=rename)

    required = ["index", "y_true", "probability"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Prediction CSV missing columns {missing}; available: {list(df.columns)}")

    if "pred_selected" not in df.columns:
        use_threshold = float(df["selected_threshold"].iloc[0]) if "selected_threshold" in df.columns else float(threshold)
        df["pred_selected"] = (df["probability"].astype(float) >= use_threshold).astype(int)

    df["index"] = df["index"].astype(int)
    df["y_true"] = df["y_true"].astype(int)
    df["probability"] = df["probability"].astype(float)
    df["pred_selected"] = df["pred_selected"].astype(int)
    if "selected_threshold" not in df.columns:
        df["selected_threshold"] = float(threshold)
    if "category" not in df.columns:
        df = label_categories(df)
    else:
        df["category"] = df["category"].astype(str)

    if df["index"].duplicated().any():
        raise ValueError("Prediction CSV contains duplicate subset indices.")
    if not set(df["y_true"].unique()).issubset({0, 1}):
        raise ValueError("Prediction labels are not binary.")
    if not set(df["pred_selected"].unique()).issubset({0, 1}):
        raise ValueError("Prediction decisions are not binary.")
    if not np.isfinite(df["probability"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction probabilities contain non-finite values.")

    return df.sort_values("index").reset_index(drop=True)


def validate_prediction_alignment(df: pd.DataFrame, reader: Camelyon17WildsFullReader) -> Dict[str, Any]:
    n = len(reader)
    if len(df) != n:
        raise ValueError(f"Prediction row count mismatch: predictions={len(df)}, HDF5={n}")
    expected_idx = np.arange(n, dtype=np.int64)
    idx = df["index"].to_numpy(dtype=np.int64)
    if not np.array_equal(idx, expected_idx):
        raise ValueError("Prediction indices are not exactly 0..N-1 aligned to the HDF5 external-test rows.")
    y_h5 = reader.labels_all()
    y_csv = df["y_true"].to_numpy(dtype=np.int64)
    mismatch = int(np.sum(y_h5 != y_csv))
    if mismatch:
        raise ValueError(f"Prediction/HDF5 label mismatch on {mismatch} rows.")
    return {
        "n_rows": int(n),
        "index_alignment": "PASS",
        "label_alignment": "PASS",
        "n_negative": int(np.sum(y_csv == 0)),
        "n_positive": int(np.sum(y_csv == 1)),
        "category_counts": df.groupby("category").size().to_dict(),
    }


# ============================================================
# Final model reconstruction
# ============================================================
def import_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("final_model_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def try_build_model(mod, variant: str):
    for name in ["build_model_from_variant", "build_model", "make_model", "create_model", "get_model"]:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                sig = inspect.signature(fn)
                if "variant" in sig.parameters:
                    return fn(variant=variant)
                if len(sig.parameters) == 1:
                    return fn(variant)
                if len(sig.parameters) == 0:
                    return fn()
            except Exception:
                pass

    for name in [
        "MetaPathModel", "PathEOMNetR", "PathEOMNet", "MetaPathNet",
        "RegularizedRedesignModel", "PCamModel",
    ]:
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        try:
            sig = inspect.signature(cls)
            if "variant" in sig.parameters:
                return cls(variant=variant)
            if len(sig.parameters) == 1:
                return cls(variant)
            if len(sig.parameters) == 0:
                return cls()
        except Exception:
            pass

    raise RuntimeError(
        "Could not reconstruct the final 11M model from the training script. "
        "Use src/model/train_final_model.py for the final p7_r2_sens_reg_11m configuration."
    )


def load_model(train_script: Path, checkpoint: Path, variant: str, device: torch.device):
    mod = import_module_from_path(train_script)
    model = try_build_model(mod, variant)
    model.to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def model_logit(model, xt: torch.Tensor) -> torch.Tensor:
    out = model(xt)
    if isinstance(out, dict):
        out = out.get("logits", out)
    if out.ndim > 1:
        out = out[:, 0]
    return out


# ============================================================
# Attribution computation
# ============================================================
def make_attribution_cmap():
    return LinearSegmentedColormap.from_list(
        "attrib_bgyr",
        [
            (0.00, "#2747d9"),
            (0.35, "#28b463"),
            (0.68, "#f4d03f"),
            (0.83, "#eb984e"),
            (1.00, "#c0392b"),
        ],
    )


def normalize_saliency(sal: np.ndarray) -> np.ndarray:
    lo = np.percentile(sal, 5)
    hi = np.percentile(sal, 99.5)
    sal = (sal - lo) / (hi - lo + 1e-12)
    return np.clip(sal, 0.0, 1.0)


def compute_smoothgrad_single(
    model,
    x_chw: np.ndarray,
    device: torch.device,
    n_smooth: int = 16,
    noise_std: float = 0.03,
) -> Tuple[float, np.ndarray]:
    x0 = torch.tensor(x_chw[None], dtype=torch.float32, device=device)
    grads = []
    prob_value = None
    n_smooth = max(1, int(n_smooth))
    for _ in range(n_smooth):
        xt = (
            torch.clamp(x0 + torch.randn_like(x0) * noise_std, 0.0, 1.0).detach()
            if n_smooth > 1
            else x0.detach()
        )
        xt.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logit = model_logit(model, xt)[0]
        prob = torch.sigmoid(logit)
        if prob_value is None:
            prob_value = float(prob.detach().cpu().item())
        logit.backward()
        grad = xt.grad.detach().cpu().numpy()[0]
        grads.append(np.abs(grad))
    g = np.mean(np.stack(grads, axis=0), axis=0)
    sal = np.max(g, axis=0)
    return float(prob_value), normalize_saliency(sal)


def compute_batch_singlepass_saliency(
    model,
    xb: np.ndarray,
    device: torch.device,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast full-external-set attribution mode.
    Computes one gradient saliency map per sample in a batch.
    Summing logits preserves d logit_i / d x_i per-sample independence.
    """
    x0 = torch.tensor(xb, dtype=torch.float32, device=device)
    if noise_std > 0:
        x0 = torch.clamp(x0 + torch.randn_like(x0) * noise_std, 0.0, 1.0)
    xt = x0.detach()
    xt.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model_logit(model, xt)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    logits.sum().backward()
    grad = xt.grad.detach().cpu().numpy()
    g = np.abs(grad)
    sal = np.max(g, axis=1)
    sal_norm = np.stack([normalize_saliency(s) for s in sal], axis=0)
    return probs.astype(np.float64), sal_norm.astype(np.float32)


def attribution_stats(sal: np.ndarray) -> Dict[str, float]:
    h, w = sal.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    center_mask = rr <= 0.35 * min(h, w)
    border_mask = rr >= 0.45 * min(h, w)
    return {
        "saliency_mean": float(np.mean(sal)),
        "saliency_std": float(np.std(sal)),
        "saliency_max": float(np.max(sal)),
        "hotspot_frac_gt_050": float(np.mean(sal >= 0.50)),
        "hotspot_frac_gt_075": float(np.mean(sal >= 0.75)),
        "center_saliency_mean": float(np.mean(sal[center_mask])) if center_mask.any() else float("nan"),
        "border_saliency_mean": float(np.mean(sal[border_mask])) if border_mask.any() else float("nan"),
    }


# ============================================================
# Visualization
# ============================================================
def chw_to_rgb(x_chw: np.ndarray) -> np.ndarray:
    return np.transpose(x_chw, (1, 2, 0))


def add_scale_bar(
    ax,
    image_shape_hw: Tuple[int, int],
    mpp: float,
    scalebar_um: float,
    bar_color: str = "black",
    text_color: str = "black",
) -> None:
    if mpp is None or not np.isfinite(mpp) or mpp <= 0:
        return
    h, w = image_shape_hw
    bar_px = int(round(scalebar_um / mpp))
    bar_px = max(5, min(bar_px, w // 2))
    x0 = w - bar_px - 8
    y0 = h - 9
    outline = "white" if bar_color.lower() == "black" else "black"
    ax.add_patch(Rectangle((x0 - 1, y0 - 1), bar_px + 2, 4, color=outline, alpha=0.55, linewidth=0))
    ax.add_patch(Rectangle((x0, y0), bar_px, 2, color=bar_color, linewidth=0))
    ax.text(
        x0 + bar_px / 2,
        y0 - 4,
        f"{int(scalebar_um)} µm",
        color=text_color,
        ha="center",
        va="bottom",
        fontsize=8,
        weight="bold",
    )


def overlay_saliency(
    raw_rgb: np.ndarray,
    sal: np.ndarray,
    bg_opacity: float,
    low_alpha: float,
    max_alpha: float,
    saliency_floor: float,
    gamma: float,
    saliency_cap: float,
    show_low_background: bool,
) -> np.ndarray:
    raw_rgb = np.clip(raw_rgb, 0.0, 1.0)
    sal = np.clip(sal, 0.0, 1.0)
    if saliency_cap < 1.0:
        sal = np.clip(sal / max(saliency_cap, 1e-6), 0.0, 1.0)
    heat = make_attribution_cmap()(sal)[..., :3]
    alpha_map = np.clip((sal - saliency_floor) / max(1.0 - saliency_floor, 1e-6), 0.0, 1.0)
    alpha_map = alpha_map ** gamma
    alpha_map = low_alpha + (max_alpha - low_alpha) * alpha_map if show_low_background else max_alpha * alpha_map
    raw_faded = bg_opacity * raw_rgb + (1.0 - bg_opacity) * 1.0
    out = (1.0 - alpha_map[..., None]) * raw_faded + alpha_map[..., None] * heat
    return np.clip(out, 0.0, 1.0)


def render_single(
    raw_rgb: np.ndarray,
    sal: Optional[np.ndarray],
    out_png: Path,
    title: str,
    args: argparse.Namespace,
    overlay: bool,
) -> None:
    fig = plt.figure(figsize=(3.1, 3.25))
    ax = plt.gca()
    img = (
        overlay_saliency(
            raw_rgb,
            sal,
            args.bg_opacity,
            args.low_alpha,
            args.max_alpha,
            args.saliency_floor,
            args.alpha_gamma,
            args.saliency_cap,
            args.show_low_background,
        )
        if overlay and sal is not None
        else raw_rgb
    )
    ax.imshow(img, interpolation="nearest")
    add_scale_bar(ax, raw_rgb.shape[:2], args.mpp, args.scalebar_um)
    ax.set_title(title, fontsize=9, weight="bold")
    ax.axis("off")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_main_panel(examples: List[Dict[str, Any]], out_png: Path, args: argparse.Namespace) -> None:
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(8.8, 3.35 * n))
    if n == 1:
        axes = np.array([axes])
    letter_ord = ord("A")
    for i, ex in enumerate(examples):
        raw_rgb = ex["raw_rgb"]
        sal = ex["saliency"]
        ov = overlay_saliency(
            raw_rgb,
            sal,
            args.bg_opacity,
            args.low_alpha,
            args.max_alpha,
            args.saliency_floor,
            args.alpha_gamma,
            args.saliency_cap,
            args.show_low_background,
        )
        cat = ex["category"]
        y = ex["y_true"]
        pred = ex["pred_selected"]
        p = ex["probability"]
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.imshow(raw_rgb, interpolation="nearest")
        ax1.imshow(ov, interpolation="nearest")
        add_scale_bar(ax0, raw_rgb.shape[:2], args.mpp, args.scalebar_um)
        add_scale_bar(ax1, raw_rgb.shape[:2], args.mpp, args.scalebar_um)
        ax0.set_title(f"{cat} raw\ntrue={y}, pred={pred}, p={p:.3f}", fontsize=10, weight="bold")
        ax1.set_title(f"{cat} attribution\ntrue={y}, pred={pred}, p={p:.3f}", fontsize=10, weight="bold")
        ax0.axis("off")
        ax1.axis("off")
        ax0.text(-0.13, 1.02, chr(letter_ord), transform=ax0.transAxes, fontsize=15, weight="bold", va="bottom", ha="right")
        ax1.text(-0.13, 1.02, chr(letter_ord + 1), transform=ax1.transAxes, fontsize=15, weight="bold", va="bottom", ha="right")
        letter_ord += 2
    plt.subplots_adjust(top=0.90, bottom=0.075, hspace=0.38, wspace=0.10)
    fig.suptitle("CAMELYON17-WILDS external patch-level attribution maps", y=0.985, fontsize=14, weight="bold")
    cax = fig.add_axes([0.18, 0.035, 0.64, 0.016])
    gradient = np.linspace(0, 1, 512)[None, :]
    cax.imshow(gradient, aspect="auto", cmap=make_attribution_cmap())
    cax.set_xticks([])
    cax.set_yticks([])
    fig.text(0.13, 0.042, "Blue: low", ha="left", va="center", fontsize=8.5)
    fig.text(0.35, 0.042, "Green: moderate", ha="center", va="center", fontsize=8.5)
    fig.text(0.62, 0.042, "Yellow/orange: high", ha="center", va="center", fontsize=8.5)
    fig.text(0.83, 0.042, "Red: very high", ha="left", va="center", fontsize=8.5)
    fig.text(
        0.50,
        0.016,
        "Attribution strength / local model sensitivity (not pixel-level metastasis probability)",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Selection and summaries
# ============================================================
def pick_main_examples(df: pd.DataFrame, n_each: int) -> pd.DataFrame:
    chosen = []
    for cat in ["TP", "TN", "FP", "FN"]:
        sub = df[df["category"] == cat].copy()
        if len(sub) == 0:
            continue
        if cat == "TP":
            sub = sub.sort_values("probability", ascending=False)
            lo = int(0.10 * len(sub))
            picked = sub.iloc[lo:lo+n_each] if len(sub) > lo else sub.head(n_each)
        elif cat == "TN":
            sub = sub.sort_values("probability", ascending=True)
            lo = int(0.10 * len(sub))
            picked = sub.iloc[lo:lo+n_each] if len(sub) > lo else sub.head(n_each)
        elif cat == "FP":
            picked = sub.sort_values("probability", ascending=False).head(n_each)
        else:
            picked = sub.sort_values("probability", ascending=True).head(n_each)
        chosen.append(picked)
    if not chosen:
        raise RuntimeError("No TP/TN/FP/FN examples could be selected.")
    return pd.concat(chosen, axis=0).reset_index(drop=True)


def flatten_group_summary(df: pd.DataFrame, group_col: str, metrics: List[str]) -> pd.DataFrame:
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


def select_gallery_candidates(stats_df: pd.DataFrame, per_category: int) -> pd.DataFrame:
    pieces = []
    for cat in ["TP", "TN", "FP", "FN"]:
        sub = stats_df[stats_df["category"] == cat].copy()
        if len(sub) == 0:
            continue
        n_take = min(int(per_category), len(sub))
        if n_take <= 0:
            continue
        sub = sub.sort_values("probability")
        positions = np.linspace(0, len(sub) - 1, n_take).round().astype(int)
        pieces.append(sub.iloc[positions].copy())
    return pd.concat(pieces, axis=0).reset_index(drop=True) if pieces else pd.DataFrame()


# ============================================================
# Main workflow
# ============================================================
def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    dirs = prepare_dirs(paths["outdir"], args.overwrite)
    device = resolve_device(args.device)

    status("Opening full CAMELYON17-WILDS test HDF5 files.")
    reader = Camelyon17WildsFullReader(paths["x_h5"], paths["y_h5"])
    status(f"External test rows: {len(reader):,}")

    status("Loading Phase 13D frozen prediction table.")
    pred_df = load_external_predictions(paths["predictions_csv"], args.threshold)
    alignment = validate_prediction_alignment(pred_df, reader)
    pred_copy = dirs["pred_dir"] / "wilds_full_test_predictions_aligned.csv"
    pred_df.to_csv(pred_copy, index=False)
    write_json(dirs["stats_dir"] / "prediction_alignment_verification.json", alignment)

    status("Loading frozen final 11M PathEOM-Net-R model.")
    model, load_info = load_model(paths["train_script"], paths["checkpoint"], args.variant, device)

    # Main panel.
    main_selected = pick_main_examples(pred_df, args.main_n_each)
    main_selected_path = dirs["main_dir"] / "selected_external_examples.csv"
    main_selected.to_csv(main_selected_path, index=False)
    main_examples: List[Dict[str, Any]] = []

    status("Generating main external attribution panel examples.")
    for _, row in main_selected.iterrows():
        idx = int(row["index"])
        x_chw, y = reader.get(idx)
        raw_rgb = chw_to_rgb(x_chw)
        prob_recomputed, sal = compute_smoothgrad_single(
            model,
            x_chw,
            device,
            n_smooth=args.main_smoothgrad_n,
            noise_std=args.main_noise_std,
        )
        prob_csv = float(row["probability"])
        pred = int(row["pred_selected"])
        cat = str(row["category"])
        base = f"{cat}_idx{idx}_true{y}_pred{pred}_p{prob_csv:.3f}"
        render_single(
            raw_rgb,
            None,
            dirs["main_dir"] / f"{base}_raw.png",
            f"{cat} raw | true={y}, pred={pred}, p={prob_csv:.3f}",
            args,
            overlay=False,
        )
        render_single(
            raw_rgb,
            sal,
            dirs["main_dir"] / f"{base}_attribution.png",
            f"{cat} attribution | true={y}, pred={pred}, p={prob_csv:.3f}",
            args,
            overlay=True,
        )
        main_examples.append(
            {
                "category": cat,
                "index": idx,
                "y_true": y,
                "pred_selected": pred,
                "probability": prob_csv,
                "probability_recomputed_for_saliency": float(prob_recomputed),
                "raw_rgb": raw_rgb,
                "saliency": sal,
            }
        )

    main_panel = dirs["main_dir"] / "wilds_external_attribution_panel.png"
    render_main_panel(main_examples, main_panel, args)

    all_stats_df = pd.DataFrame()
    gallery_df = pd.DataFrame()
    if args.generate_all_map_statistics:
        work_df = pred_df.copy()
        if args.max_all_maps > 0:
            work_df = work_df.iloc[: args.max_all_maps].copy()
        status(f"Generating attribution statistics for {len(work_df):,} external test patches.")
        records: List[Dict[str, Any]] = []
        indices = work_df["index"].astype(int).tolist()
        chunk = max(1, int(args.all_attribution_batch_size))
        pred_lookup = work_df.set_index("index").to_dict(orient="index")

        for start in range(0, len(indices), chunk):
            idxs = indices[start : start + chunk]
            xb, _yb = reader.get_batch(idxs)
            probs_recomputed, sal_batch = compute_batch_singlepass_saliency(
                model,
                xb,
                device,
                noise_std=args.all_noise_std,
            )
            for local_i, idx in enumerate(idxs):
                row = pred_lookup[int(idx)]
                sal = sal_batch[local_i]
                st = attribution_stats(sal)
                p_csv = float(row["probability"])
                p_recomp = float(probs_recomputed[local_i])
                records.append(
                    {
                        "index": int(idx),
                        "category": str(row["category"]),
                        "y_true": int(row["y_true"]),
                        "pred_selected": int(row["pred_selected"]),
                        "probability": p_csv,
                        "probability_recomputed_for_saliency": p_recomp,
                        "abs_probability_difference": abs(p_csv - p_recomp),
                        **st,
                    }
                )
            if (start // chunk) % 50 == 0 or start + chunk >= len(indices):
                status(f"Attribution stats {min(start + chunk, len(indices)):,}/{len(indices):,}")

        all_stats_df = pd.DataFrame(records)
        all_stats_path = dirs["stats_dir"] / "all_external_map_statistics.csv"
        all_stats_df.to_csv(all_stats_path, index=False)

        summary_metrics = [
            "saliency_mean",
            "saliency_std",
            "saliency_max",
            "hotspot_frac_gt_050",
            "hotspot_frac_gt_075",
            "center_saliency_mean",
            "border_saliency_mean",
            "abs_probability_difference",
        ]
        flat_summary = flatten_group_summary(all_stats_df, "category", summary_metrics)
        flat_summary.to_csv(
            dirs["stats_dir"] / "external_attribution_summary_by_category.csv",
            index=False,
        )

        probability_recompute_summary = {
            "n_maps": int(len(all_stats_df)),
            "mean_abs_probability_difference": float(all_stats_df["abs_probability_difference"].mean()),
            "max_abs_probability_difference": float(all_stats_df["abs_probability_difference"].max()),
            "median_abs_probability_difference": float(all_stats_df["abs_probability_difference"].median()),
        }
        write_json(
            dirs["stats_dir"] / "probability_recompute_consistency.json",
            probability_recompute_summary,
        )

        gallery_df = select_gallery_candidates(all_stats_df, args.save_gallery_per_category)
        gallery_rows: List[Dict[str, Any]] = []
        if len(gallery_df):
            status(f"Rendering {len(gallery_df):,} curated external attribution gallery maps.")
            for _, row in gallery_df.iterrows():
                idx = int(row["index"])
                x_chw, y = reader.get(idx)
                raw_rgb = chw_to_rgb(x_chw)
                _, sal = compute_smoothgrad_single(model, x_chw, device, n_smooth=1, noise_std=args.all_noise_std)
                cat = str(row["category"])
                pred = int(row["pred_selected"])
                prob = float(row["probability"])
                base = f"{cat}_idx{idx}_true{y}_pred{pred}_p{prob:.3f}"
                out_png = dirs["gallery_dir"] / cat / f"{base}_attribution.png"
                render_single(
                    raw_rgb,
                    sal,
                    out_png,
                    f"{cat} attribution | true={y}, pred={pred}, p={prob:.3f}",
                    args,
                    overlay=True,
                )
                gallery_rows.append({**row.to_dict(), "map_png": str(out_png)})
            pd.DataFrame(gallery_rows).to_csv(
                dirs["stats_dir"] / "external_gallery_candidates.csv",
                index=False,
            )

        if args.save_all_map_pngs:
            status("Saving all external map PNGs. This option is storage-heavy.")
            for _, row in all_stats_df.iterrows():
                idx = int(row["index"])
                x_chw, y = reader.get(idx)
                raw_rgb = chw_to_rgb(x_chw)
                _, sal = compute_smoothgrad_single(model, x_chw, device, n_smooth=1, noise_std=args.all_noise_std)
                cat = str(row["category"])
                pred = int(row["pred_selected"])
                prob = float(row["probability"])
                base = f"{cat}_idx{idx}_true{y}_pred{pred}_p{prob:.3f}"
                out_png = dirs["all_png_dir"] / cat / f"{base}_attribution.png"
                render_single(
                    raw_rgb,
                    sal,
                    out_png,
                    f"{cat} attribution | true={y}, pred={pred}, p={prob:.3f}",
                    args,
                    overlay=True,
                )

    manifest = {
        "phase": "Phase 15A CAMELYON17-WILDS full-test external attribution/map audit",
        "status": "COMPLETED",
        "scientific_boundary": (
            "Patch-level local model-sensitivity maps only; not segmentation, not pixel-level "
            "metastatic probability, and not whole-slide localization."
        ),
        "variant": args.variant,
        "threshold": float(args.threshold),
        "device": str(device),
        "input_paths": {k: str(v) for k, v in paths.items()},
        "output_paths": {k: str(v) for k, v in dirs.items()},
        "alignment_verification": alignment,
        "model_load_info": load_info,
        "main_panel_png": str(main_panel),
        "main_selected_examples_csv": str(main_selected_path),
        "generate_all_map_statistics": bool(args.generate_all_map_statistics),
        "max_all_maps": int(args.max_all_maps),
        "save_gallery_per_category": int(args.save_gallery_per_category),
        "save_all_map_pngs": bool(args.save_all_map_pngs),
        "mpp": float(args.mpp),
        "scalebar_um": float(args.scalebar_um),
        "all_map_statistics_csv": (
            str(dirs["stats_dir"] / "all_external_map_statistics.csv")
            if args.generate_all_map_statistics
            else ""
        ),
        "attribution_summary_by_category_csv": (
            str(dirs["stats_dir"] / "external_attribution_summary_by_category.csv")
            if args.generate_all_map_statistics
            else ""
        ),
        "gallery_candidates_csv": (
            str(dirs["stats_dir"] / "external_gallery_candidates.csv")
            if args.generate_all_map_statistics and len(gallery_df)
            else ""
        ),
    }
    write_json(dirs["outdir"] / "wilds_attribution_manifest.json", manifest)
    reader.close()
    status("Phase 15A external attribution/map audit completed.")
    print(json.dumps(jsonable(manifest), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        raise




