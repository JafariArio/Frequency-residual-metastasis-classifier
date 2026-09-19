#!/usr/bin/env python
r"""
Phase 9A v3 — final 11M PathEOM-Net-R:
- full test-set predictions
- final main-manuscript attribution panel
- all-test-patch attribution statistics
- optional curated gallery images
- optional all-map PNG export (not recommended by default)

This script extends the Phase 9A v2 workflow. Additions:
1) ECE now matches the Phase 7 training definition:
   ECE = sum_bin weight * |mean(y_true_bin) - mean(prob_bin)|.
2) ROC-AUC / PR-AUC are computed when scikit-learn is available.
3) The main attribution panel layout is corrected to avoid title overlap.
4) Bulk map processing supports batched single-pass gradients for faster all-test statistics.
5) The default full analysis computes all-map statistics without saving 65k PNGs.
   A curated gallery sample is saved for SI; all PNGs remain optional.

Scientific boundary:
Attribution maps show patch-level local model sensitivity.
They are not tumor segmentation masks and not pixel-level metastatic probabilities.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score,
        balanced_accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, matthews_corrcoef, cohen_kappa_score,
        log_loss, roc_curve, precision_recall_curve
    )
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ============================================================
# Data helpers
# ============================================================
def find_file(root: Path, split: str, kind: str) -> Path:
    patterns = [
        f"*split_{split}_{kind}.h5",
        f"*{split}_{kind}.h5",
        f"split_{split}_{kind}.h5",
        f"{split}_{kind}.h5",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(root.glob(pat))
        hits.extend(root.glob(f"**/{pat}"))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(f"Could not find PCam {split}_{kind}.h5 under {root}")
    return hits[0]


def first_dataset_key(h5: h5py.File) -> str:
    keys = list(h5.keys())
    for k in ["x", "X", "data", "images", "y", "Y", "labels"]:
        if k in h5:
            return k
    if not keys:
        raise ValueError("H5 file has no dataset keys.")
    return keys[0]


class PCamReader:
    def __init__(self, root: str, split: str = "test"):
        self.root = Path(root)
        self.split = split
        self.x_path = find_file(self.root, split, "x")
        self.y_path = find_file(self.root, split, "y")
        self.fx = h5py.File(self.x_path, "r")
        self.fy = h5py.File(self.y_path, "r")
        self.x_key = first_dataset_key(self.fx)
        self.y_key = first_dataset_key(self.fy)
        self.n = int(self.fx[self.x_key].shape[0])

    def __len__(self) -> int:
        return self.n

    def get(self, idx: int) -> Tuple[np.ndarray, int]:
        x = np.asarray(self.fx[self.x_key][idx])
        y = int(np.asarray(self.fy[self.y_key][idx]).reshape(-1)[0])
        if x.ndim == 3 and x.shape[-1] == 3:
            x = np.transpose(x, (2, 0, 1))
        x = np.ascontiguousarray(np.clip(x.astype(np.float32) / 255.0, 0.0, 1.0))
        return x, y

    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for idx in indices:
            x, y = self.get(int(idx))
            xs.append(x)
            ys.append(y)
        return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


# ============================================================
# Model helpers
# ============================================================
def import_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("final_model_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import script: {path}")
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
        "RegularizedRedesignModel", "PCamModel"
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
        "Could not build the model from the training script. "
        "Use src/model/train_final_model.py for the final p7_r2_sens_reg_11m configuration."
    )


def load_model(train_script: str, checkpoint: str, variant: str, device: torch.device):
    mod = import_module_from_path(train_script)
    model = try_build_model(mod, variant)
    model.to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    info = {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    return model, info


def model_logit(model, xt: torch.Tensor) -> torch.Tensor:
    out = model(xt)
    if out.ndim > 1:
        out = out[:, 0]
    return out


# ============================================================
# Prediction + metric helpers
# ============================================================
def run_predictions(model, reader: PCamReader, device: torch.device, batch_size: int, threshold: float) -> pd.DataFrame:
    rows = []
    n = len(reader)
    batch_size = max(1, int(batch_size))
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        idxs = list(range(start, end))
        xb, yb = reader.get_batch(idxs)
        xt = torch.tensor(xb, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = torch.sigmoid(model_logit(model, xt)).detach().cpu().numpy()
        preds = (probs >= threshold).astype(np.int64)
        for idx, y, p, pred in zip(idxs, yb.tolist(), probs.tolist(), preds.tolist()):
            rows.append({
                "index": int(idx),
                "y_true": int(y),
                "probability": float(p),
                "selected_threshold": float(threshold),
                "pred_selected": int(pred),
            })
        if (start // batch_size) % 20 == 0 or end == n:
            print(f"Predictions: {end}/{n}")
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, prob, bins: int = 15) -> float:
    """Match the Phase 7 training-script ECE definition."""
    y_true = np.asarray(y_true).astype(np.float32)
    prob = np.asarray(prob).astype(np.float32)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(y_true[mask].mean()) - float(prob[mask].mean()))
    return float(ece)


def metric_safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def confusion_matrix_rows(tp: int, tn: int, fp: int, fn: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"actual_class": "Negative", "pred_negative": tn, "pred_positive": fp, "row_total": tn + fp},
        {"actual_class": "Positive", "pred_negative": fn, "pred_positive": tp, "row_total": fn + tp},
        {"actual_class": "column_total", "pred_negative": tn + fn, "pred_positive": fp + tp, "row_total": tn + fp + fn + tp},
    ])


def confusion_matrix_normalized_rows(tp: int, tn: int, fp: int, fn: int) -> pd.DataFrame:
    neg_total = tn + fp
    pos_total = fn + tp
    pred_neg_total = tn + fn
    pred_pos_total = fp + tp
    total = tn + fp + fn + tp
    return pd.DataFrame([
        {
            "normalization": "by_actual_class",
            "actual_class": "Negative",
            "pred_negative": metric_safe_div(tn, neg_total),
            "pred_positive": metric_safe_div(fp, neg_total),
        },
        {
            "normalization": "by_actual_class",
            "actual_class": "Positive",
            "pred_negative": metric_safe_div(fn, pos_total),
            "pred_positive": metric_safe_div(tp, pos_total),
        },
        {
            "normalization": "by_predicted_class",
            "actual_class": "Negative",
            "pred_negative": metric_safe_div(tn, pred_neg_total),
            "pred_positive": metric_safe_div(fp, pred_pos_total),
        },
        {
            "normalization": "by_predicted_class",
            "actual_class": "Positive",
            "pred_negative": metric_safe_div(fn, pred_neg_total),
            "pred_positive": metric_safe_div(tp, pred_pos_total),
        },
        {
            "normalization": "by_total",
            "actual_class": "Negative",
            "pred_negative": metric_safe_div(tn, total),
            "pred_positive": metric_safe_div(fp, total),
        },
        {
            "normalization": "by_total",
            "actual_class": "Positive",
            "pred_negative": metric_safe_div(fn, total),
            "pred_positive": metric_safe_div(tp, total),
        },
    ])


def per_class_metrics_rows(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    if SKLEARN_OK:
        precision = precision_score(y, pred, average=None, labels=[0, 1], zero_division=0)
        recall = recall_score(y, pred, average=None, labels=[0, 1], zero_division=0)
        f1 = f1_score(y, pred, average=None, labels=[0, 1], zero_division=0)
        support = np.array([(y == 0).sum(), (y == 1).sum()], dtype=int)
        return pd.DataFrame([
            {"class_label": 0, "class_name": "Negative", "precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            {"class_label": 1, "class_name": "Positive", "precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
        ])
    return pd.DataFrame()


def compute_binary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = df["y_true"].astype(int).to_numpy()
    p = df["probability"].astype(float).to_numpy()
    pred = df["pred_selected"].astype(int).to_numpy()

    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    sens = metric_safe_div(tp, tp + fn)
    spec = metric_safe_div(tn, tn + fp)
    precision_pos = metric_safe_div(tp, tp + fp)
    npv = metric_safe_div(tn, tn + fn)
    fpr = metric_safe_div(fp, fp + tn)
    fnr = metric_safe_div(fn, fn + tp)
    fdr = metric_safe_div(fp, fp + tp)
    forate = metric_safe_div(fn, fn + tn)
    acc = metric_safe_div(tp + tn, len(df))
    bacc = 0.5 * (sens + spec)
    f1_pos = metric_safe_div(2 * precision_pos * sens, precision_pos + sens)
    gmean = float(np.sqrt(sens * spec)) if np.isfinite(sens) and np.isfinite(spec) else float("nan")
    prevalence = float(np.mean(y == 1))
    predicted_positive_rate = float(np.mean(pred == 1))
    brier = float(np.mean((p - y) ** 2))
    ece = expected_calibration_error(y, p, bins=15)

    out: Dict[str, float] = {
        "n_total": int(len(df)),
        "n_negative": int((y == 0).sum()),
        "n_positive": int((y == 1).sum()),
        "prevalence_positive": prevalence,
        "predicted_positive_rate": predicted_positive_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "sensitivity_recall_positive": float(sens),
        "specificity_recall_negative": float(spec),
        "precision_ppv_positive": float(precision_pos),
        "npv_negative_predictive_value": float(npv),
        "positive_f1": float(f1_pos),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "false_discovery_rate": float(fdr),
        "false_omission_rate": float(forate),
        "g_mean_sens_spec": float(gmean),
        "brier": float(brier),
        "ece": float(ece),
        "ece_definition": "15-bin positive-rate-vs-probability ECE matching Phase 7 training script",
    }

    if SKLEARN_OK and len(np.unique(y)) == 2:
        out.update({
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc_average_precision": float(average_precision_score(y, p)),
            "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_precision": float(precision_score(y, pred, average="weighted", zero_division=0)),
            "weighted_recall": float(recall_score(y, pred, average="weighted", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "matthews_corrcoef": float(matthews_corrcoef(y, pred)),
            "cohen_kappa": float(cohen_kappa_score(y, pred)),
            "nll_log_loss": float(log_loss(y, p, labels=[0, 1])),
            "sklearn_roc_pr_available": True,
        })
    else:
        out.update({
            "roc_auc": float("nan"),
            "pr_auc_average_precision": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "weighted_precision": float("nan"),
            "weighted_recall": float("nan"),
            "weighted_f1": float("nan"),
            "matthews_corrcoef": float("nan"),
            "cohen_kappa": float("nan"),
            "nll_log_loss": float("nan"),
            "sklearn_roc_pr_available": False,
        })
    return out


def label_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    conds = [
        (df.y_true == 1) & (df.pred_selected == 1),
        (df.y_true == 0) & (df.pred_selected == 0),
        (df.y_true == 0) & (df.pred_selected == 1),
        (df.y_true == 1) & (df.pred_selected == 0),
    ]
    cats = ["TP", "TN", "FP", "FN"]
    df["category"] = np.select(conds, cats, default="OTHER")
    return df


def pick_main_examples(df: pd.DataFrame, n_each: int = 1) -> pd.DataFrame:
    """Representative paper panel: strong correct calls + clear error calls."""
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
        else:  # FN
            picked = sub.sort_values("probability", ascending=True).head(n_each)
        chosen.append(picked)
    if not chosen:
        raise RuntimeError("No TP/TN/FP/FN examples could be selected.")
    return pd.concat(chosen, axis=0).reset_index(drop=True)


# ============================================================
# Attribution helpers
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
        xt = torch.clamp(x0 + torch.randn_like(x0) * noise_std, 0.0, 1.0).detach() if n_smooth > 1 else x0.detach()
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
    Fast all-test attribution mode.
    Computes one gradient saliency map per sample in a batch.
    For each sample, the gradient is d logit_i / d x_i; summing logits preserves per-sample independence.
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


def chw_to_rgb(x_chw: np.ndarray) -> np.ndarray:
    return np.transpose(x_chw, (1, 2, 0))


def add_scale_bar(
    ax,
    image_shape_hw: Tuple[int, int],
    mpp: float,
    scalebar_um: float = 25.0,
    bar_color: str = "black",
    text_color: str = "black",
):
    h, w = image_shape_hw
    bar_px = int(round(scalebar_um / mpp))
    bar_px = max(5, min(bar_px, w // 2))
    x0 = w - bar_px - 8
    y0 = h - 9
    outline = "white" if bar_color.lower() == "black" else "black"
    ax.add_patch(Rectangle((x0 - 1, y0 - 1), bar_px + 2, 4, color=outline, alpha=0.55, linewidth=0))
    ax.add_patch(Rectangle((x0, y0), bar_px, 2, color=bar_color, linewidth=0))
    ax.text(
        x0 + bar_px / 2, y0 - 4, f"{int(scalebar_um)} µm",
        color=text_color, ha="center", va="bottom", fontsize=8, weight="bold"
    )


def overlay_saliency(
    raw_rgb: np.ndarray,
    sal: np.ndarray,
    bg_opacity: float = 0.22,
    low_alpha: float = 0.16,
    max_alpha: float = 0.88,
    saliency_floor: float = 0.00,
    gamma: float = 1.0,
    saliency_cap: float = 0.98,
    show_low_background: bool = True,
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
    mpp: float,
    scalebar_um: float,
    bg_opacity: float,
    low_alpha: float,
    max_alpha: float,
    saliency_floor: float,
    gamma: float,
    saliency_cap: float,
    show_low_background: bool,
    overlay: bool = False,
):
    fig = plt.figure(figsize=(3.1, 3.25))
    ax = plt.gca()
    img = overlay_saliency(
        raw_rgb, sal, bg_opacity, low_alpha, max_alpha,
        saliency_floor, gamma, saliency_cap, show_low_background
    ) if overlay and sal is not None else raw_rgb
    ax.imshow(img, interpolation="nearest")
    add_scale_bar(ax, raw_rgb.shape[:2], mpp, scalebar_um, "black", "black")
    ax.set_title(title, fontsize=9, weight="bold")
    ax.axis("off")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_main_panel(
    examples: List[Dict],
    out_png: Path,
    mpp: float,
    scalebar_um: float,
    bg_opacity: float,
    low_alpha: float,
    max_alpha: float,
    saliency_floor: float,
    gamma: float,
    saliency_cap: float,
    show_low_background: bool,
):
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(8.8, 3.35 * n))
    if n == 1:
        axes = np.array([axes])
    letter_ord = ord("A")
    for i, ex in enumerate(examples):
        raw_rgb = ex["raw_rgb"]
        sal = ex["saliency"]
        ov = overlay_saliency(
            raw_rgb, sal, bg_opacity, low_alpha, max_alpha,
            saliency_floor, gamma, saliency_cap, show_low_background
        )
        cat, y, pred, p = ex["category"], ex["y_true"], ex["pred_selected"], ex["probability"]
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.imshow(raw_rgb, interpolation="nearest")
        ax1.imshow(ov, interpolation="nearest")
        add_scale_bar(ax0, raw_rgb.shape[:2], mpp, scalebar_um, "black", "black")
        add_scale_bar(ax1, raw_rgb.shape[:2], mpp, scalebar_um, "black", "black")
        ax0.set_title(f"{cat} raw\ntrue={y}, pred={pred}, p={p:.3f}", fontsize=10, weight="bold")
        ax1.set_title(f"{cat} attribution\ntrue={y}, pred={pred}, p={p:.3f}", fontsize=10, weight="bold")
        ax0.axis("off")
        ax1.axis("off")
        ax0.text(-0.13, 1.02, chr(letter_ord), transform=ax0.transAxes, fontsize=15, weight="bold", va="bottom", ha="right")
        ax1.text(-0.13, 1.02, chr(letter_ord + 1), transform=ax1.transAxes, fontsize=15, weight="bold", va="bottom", ha="right")
        letter_ord += 2
    plt.subplots_adjust(top=0.90, bottom=0.075, hspace=0.38, wspace=0.10)
    fig.suptitle("Representative final 11M PathEOM-Net-R patch-level attribution maps", y=0.985, fontsize=14, weight="bold")
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
        0.50, 0.016,
        "Attribution strength / local model sensitivity (not pixel-level metastasis probability)",
        ha="center", va="center", fontsize=8.5
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Summary helpers
# ============================================================
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


def select_gallery_candidates(stats_df: pd.DataFrame, per_category: int, seed: int = 13) -> pd.DataFrame:
    """
    Balanced curated gallery sample from all maps, not necessarily the most extreme only.
    Stratifies by probability quantiles where possible.
    """
    rng = np.random.default_rng(seed)
    pieces = []
    for cat in ["TP", "TN", "FP", "FN"]:
        sub = stats_df[stats_df["category"] == cat].copy()
        if len(sub) == 0:
            continue
        n_take = min(int(per_category), len(sub))
        if n_take <= 0:
            continue
        sub = sub.sort_values("probability")
        # evenly spaced over the sorted probability range
        positions = np.linspace(0, len(sub) - 1, n_take).round().astype(int)
        picked = sub.iloc[positions].copy()
        pieces.append(picked)
    return pd.concat(pieces, axis=0).reset_index(drop=True) if pieces else pd.DataFrame()


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--train-script", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", default="p7_r2_sens_reg_11m")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prediction-batch-size", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.220)
    ap.add_argument("--main-n-each", type=int, default=1)
    ap.add_argument("--main-smoothgrad-n", type=int, default=16)
    ap.add_argument("--main-noise-std", type=float, default=0.03)
    ap.add_argument("--all-attribution-batch-size", type=int, default=16)
    ap.add_argument("--all-noise-std", type=float, default=0.00)
    ap.add_argument("--mpp", type=float, default=0.972)
    ap.add_argument("--scalebar-um", type=float, default=25.0)
    ap.add_argument("--bg-opacity", type=float, default=0.22)
    ap.add_argument("--low-alpha", type=float, default=0.16)
    ap.add_argument("--max-alpha", type=float, default=0.88)
    ap.add_argument("--saliency-floor", type=float, default=0.00)
    ap.add_argument("--alpha-gamma", type=float, default=1.0)
    ap.add_argument("--saliency-cap", type=float, default=0.98)
    ap.add_argument("--show-low-background", action="store_true")
    ap.add_argument("--generate-all-map-statistics", action="store_true")
    ap.add_argument("--max-all-maps", type=int, default=-1,
                    help="Positive number for a smoke test; -1 means the full test split.")
    ap.add_argument("--save-all-map-pngs", action="store_true",
                    help="Optional and storage-heavy. Not recommended for the paper workflow.")
    ap.add_argument("--save-gallery-per-category", type=int, default=64,
                    help="Curated SI gallery candidate maps saved from the all-map analysis.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    pred_dir = outdir / "predictions"
    main_dir = outdir / "main_manuscript_maps"
    stats_dir = outdir / "summary_stats"
    gallery_dir = outdir / "gallery_candidate_maps"
    all_png_dir = outdir / "all_map_pngs"
    for d in [pred_dir, main_dir, stats_dir, gallery_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for cat in ["TP", "TN", "FP", "FN"]:
        (gallery_dir / cat).mkdir(parents=True, exist_ok=True)
        if args.save_all_map_pngs:
            (all_png_dir / cat).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    reader = PCamReader(args.root, split=args.split)
    model, load_info = load_model(args.train_script, args.checkpoint, args.variant, device)

    # Full test predictions
    pred_df = run_predictions(model, reader, device, args.prediction_batch_size, args.threshold)
    pred_df = label_categories(pred_df)
    pred_csv = pred_dir / "final11m_test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    metrics = compute_binary_metrics(pred_df)
    metrics.update({
        "threshold": float(args.threshold),
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "n_test_examples": int(len(pred_df)),
        "sklearn_roc_pr_available": bool(SKLEARN_OK),
    })
    (stats_dir / "final11m_performance_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(stats_dir / "final11m_performance_summary.csv", index=False)

    # Detailed manuscript/SI metric exports
    y_metric = pred_df["y_true"].astype(int).to_numpy()
    pred_metric = pred_df["pred_selected"].astype(int).to_numpy()
    conf_df = confusion_matrix_rows(metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"])
    conf_norm_df = confusion_matrix_normalized_rows(metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"])
    class_metric_df = per_class_metrics_rows(y_metric, pred_metric)
    conf_df.to_csv(stats_dir / "final11m_confusion_matrix_counts.csv", index=False)
    conf_norm_df.to_csv(stats_dir / "final11m_confusion_matrix_normalized.csv", index=False)
    class_metric_df.to_csv(stats_dir / "final11m_per_class_metrics.csv", index=False)

    if SKLEARN_OK and len(np.unique(y_metric)) == 2:
        p_metric = pred_df["probability"].astype(float).to_numpy()
        roc_fpr, roc_tpr, roc_thr = roc_curve(y_metric, p_metric)
        pr_prec, pr_rec, pr_thr = precision_recall_curve(y_metric, p_metric)
        roc_curve_df = pd.DataFrame({"fpr": roc_fpr, "tpr": roc_tpr, "threshold": roc_thr})
        # PR thresholds have length n-1; pad with NaN so the table is rectangular.
        pr_thr_pad = np.concatenate([pr_thr, np.array([np.nan])])
        pr_curve_df = pd.DataFrame({"recall": pr_rec, "precision": pr_prec, "threshold": pr_thr_pad})
        roc_curve_df.to_csv(stats_dir / "final11m_roc_curve_points.csv", index=False)
        pr_curve_df.to_csv(stats_dir / "final11m_pr_curve_points.csv", index=False)

    cat_counts = pred_df.groupby("category").size().reset_index(name="count")
    cat_counts.to_csv(stats_dir / "category_counts.csv", index=False)
    pred_df.groupby("category")["probability"].agg(["count", "mean", "std", "median", "min", "max"]).reset_index().to_csv(
        stats_dir / "probability_summary_by_category.csv", index=False
    )

    # Main manuscript panel
    main_selected = pick_main_examples(pred_df, n_each=args.main_n_each)
    main_selected.to_csv(main_dir / "main_selected_examples.csv", index=False)
    main_examples = []
    for _, r in main_selected.iterrows():
        idx = int(r["index"])
        x_chw, y = reader.get(idx)
        raw_rgb = chw_to_rgb(x_chw)
        _, sal = compute_smoothgrad_single(
            model, x_chw, device,
            n_smooth=args.main_smoothgrad_n,
            noise_std=args.main_noise_std,
        )
        prob = float(r["probability"])
        pred = int(r["pred_selected"])
        cat = str(r["category"])
        base = f"{cat}_idx{idx}_true{y}_pred{pred}_p{prob:.3f}"
        render_single(
            raw_rgb, None, main_dir / f"{base}_raw.png",
            f"{cat} raw | true={y}, pred={pred}, p={prob:.3f}",
            args.mpp, args.scalebar_um, args.bg_opacity, args.low_alpha, args.max_alpha,
            args.saliency_floor, args.alpha_gamma, args.saliency_cap, args.show_low_background, overlay=False
        )
        render_single(
            raw_rgb, sal, main_dir / f"{base}_attribution.png",
            f"{cat} attribution | true={y}, pred={pred}, p={prob:.3f}",
            args.mpp, args.scalebar_um, args.bg_opacity, args.low_alpha, args.max_alpha,
            args.saliency_floor, args.alpha_gamma, args.saliency_cap, args.show_low_background, overlay=True
        )
        main_examples.append({
            "category": cat, "index": idx, "y_true": y, "pred_selected": pred,
            "probability": prob, "raw_rgb": raw_rgb, "saliency": sal
        })
    main_panel = main_dir / "figure_main_final11m_attribution_panel.png"
    render_main_panel(
        main_examples, main_panel, args.mpp, args.scalebar_um,
        args.bg_opacity, args.low_alpha, args.max_alpha, args.saliency_floor,
        args.alpha_gamma, args.saliency_cap, args.show_low_background
    )

    all_stats_df = pd.DataFrame()
    gallery_df = pd.DataFrame()
    if args.generate_all_map_statistics:
        work_df = pred_df.copy()
        if args.max_all_maps > 0:
            work_df = work_df.iloc[:args.max_all_maps].copy()
        print(f"Generating attribution statistics for {len(work_df)} patches...")
        records = []
        indices = work_df["index"].astype(int).tolist()
        chunk = max(1, int(args.all_attribution_batch_size))
        pred_lookup = work_df.set_index("index").to_dict(orient="index")
        for start in range(0, len(indices), chunk):
            idxs = indices[start:start+chunk]
            xb, yb = reader.get_batch(idxs)
            _, sal_batch = compute_batch_singlepass_saliency(
                model, xb, device, noise_std=args.all_noise_std
            )
            for local_i, idx in enumerate(idxs):
                r = pred_lookup[int(idx)]
                sal = sal_batch[local_i]
                st = attribution_stats(sal)
                records.append({
                    "index": int(idx),
                    "category": str(r["category"]),
                    "y_true": int(r["y_true"]),
                    "pred_selected": int(r["pred_selected"]),
                    "probability": float(r["probability"]),
                    **st,
                })
            if (start // chunk) % 50 == 0 or start + chunk >= len(indices):
                print(f"Attribution stats: {min(start+chunk, len(indices))}/{len(indices)}")
        all_stats_df = pd.DataFrame(records)
        all_stats_path = stats_dir / "all_map_statistics.csv"
        all_stats_df.to_csv(all_stats_path, index=False)

        flat_summary = flatten_group_summary(
            all_stats_df, "category",
            [
                "saliency_mean", "saliency_std", "saliency_max",
                "hotspot_frac_gt_050", "hotspot_frac_gt_075",
                "center_saliency_mean", "border_saliency_mean"
            ]
        )
        flat_summary.to_csv(stats_dir / "attribution_summary_by_category.csv", index=False)

        # Curated gallery candidates from maps actually analyzed
        gallery_df = select_gallery_candidates(all_stats_df, args.save_gallery_per_category)
        gallery_rows = []
        if len(gallery_df):
            for _, r in gallery_df.iterrows():
                idx = int(r["index"])
                x_chw, y = reader.get(idx)
                raw_rgb = chw_to_rgb(x_chw)
                # One-pass saliency for saved gallery image. This preserves same all-map mode.
                _, sal = compute_smoothgrad_single(model, x_chw, device, n_smooth=1, noise_std=args.all_noise_std)
                cat, pred, prob = str(r["category"]), int(r["pred_selected"]), float(r["probability"])
                base = f"{cat}_idx{idx}_true{y}_pred{pred}_p{prob:.3f}"
                out_png = gallery_dir / cat / f"{base}_attribution.png"
                render_single(
                    raw_rgb, sal, out_png,
                    f"{cat} attribution | true={y}, pred={pred}, p={prob:.3f}",
                    args.mpp, args.scalebar_um, args.bg_opacity, args.low_alpha, args.max_alpha,
                    args.saliency_floor, args.alpha_gamma, args.saliency_cap, args.show_low_background, overlay=True
                )
                gallery_rows.append({**r.to_dict(), "map_png": str(out_png)})
            pd.DataFrame(gallery_rows).to_csv(stats_dir / "gallery_candidates.csv", index=False)

        # Optional all map PNGs — storage heavy and not needed for paper.
        if args.save_all_map_pngs:
            print("Saving all map PNGs. This is storage-heavy...")
            for _, r in all_stats_df.iterrows():
                idx = int(r["index"])
                x_chw, y = reader.get(idx)
                raw_rgb = chw_to_rgb(x_chw)
                _, sal = compute_smoothgrad_single(model, x_chw, device, n_smooth=1, noise_std=args.all_noise_std)
                cat, pred, prob = str(r["category"]), int(r["pred_selected"]), float(r["probability"])
                base = f"{cat}_idx{idx}_true{y}_pred{pred}_p{prob:.3f}"
                out_png = all_png_dir / cat / f"{base}_attribution.png"
                render_single(
                    raw_rgb, sal, out_png,
                    f"{cat} attribution | true={y}, pred={pred}, p={prob:.3f}",
                    args.mpp, args.scalebar_um, args.bg_opacity, args.low_alpha, args.max_alpha,
                    args.saliency_floor, args.alpha_gamma, args.saliency_cap, args.show_low_background, overlay=True
                )

    manifest = {
        "version": "pcam_attribution_metrics_v1",
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "train_script": args.train_script,
        "threshold": args.threshold,
        "device": str(device),
        "split": args.split,
        "n_test_examples": int(len(pred_df)),
        "generate_all_map_statistics": bool(args.generate_all_map_statistics),
        "max_all_maps": int(args.max_all_maps),
        "save_all_map_pngs": bool(args.save_all_map_pngs),
        "save_gallery_per_category": int(args.save_gallery_per_category),
        "main_smoothgrad_n": int(args.main_smoothgrad_n),
        "all_attribution_batch_size": int(args.all_attribution_batch_size),
        "all_noise_std": float(args.all_noise_std),
        "mpp": float(args.mpp),
        "scalebar_um": float(args.scalebar_um),
        "display": {
            "bg_opacity": float(args.bg_opacity),
            "low_alpha": float(args.low_alpha),
            "max_alpha": float(args.max_alpha),
            "saliency_floor": float(args.saliency_floor),
            "alpha_gamma": float(args.alpha_gamma),
            "saliency_cap": float(args.saliency_cap),
            "show_low_background": bool(args.show_low_background),
        },
        "model_load_info": load_info,
        "key_outputs": {
            "predictions_csv": str(pred_csv),
            "main_selected_examples_csv": str(main_dir / "main_selected_examples.csv"),
            "main_panel_png": str(main_panel),
            "performance_summary_json": str(stats_dir / "final11m_performance_summary.json"),
            "confusion_matrix_counts_csv": str(stats_dir / "final11m_confusion_matrix_counts.csv"),
            "confusion_matrix_normalized_csv": str(stats_dir / "final11m_confusion_matrix_normalized.csv"),
            "per_class_metrics_csv": str(stats_dir / "final11m_per_class_metrics.csv"),
            "roc_curve_points_csv": str(stats_dir / "final11m_roc_curve_points.csv") if SKLEARN_OK else "",
            "pr_curve_points_csv": str(stats_dir / "final11m_pr_curve_points.csv") if SKLEARN_OK else "",
            "all_map_statistics_csv": str(stats_dir / "all_map_statistics.csv") if args.generate_all_map_statistics else "",
            "attribution_summary_by_category_csv": str(stats_dir / "attribution_summary_by_category.csv") if args.generate_all_map_statistics else "",
            "gallery_candidates_csv": str(stats_dir / "gallery_candidates.csv") if len(gallery_df) else "",
        },
        "important_note": "Patch-level attribution only; not whole-slide localization and not segmentation.",
    }
    (outdir / "pcam_attribution_metrics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


