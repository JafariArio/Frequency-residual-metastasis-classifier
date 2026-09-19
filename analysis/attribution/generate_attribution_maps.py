#!/usr/bin/env python
r'''
Phase 9A — final 11M PathEOM-Net-R main-manuscript + all-map generator.

Purpose
-------
1) Run the final 11M model on PCam test patches.
2) Export per-sample predictions and performance summaries.
3) Select one representative set of TP/TN/FP/FN examples for the main manuscript.
4) Generate a polished main attribution panel.
5) Optionally generate raw + attribution maps for the full test split (or a capped subset).
6) Export per-sample attribution summary statistics for all generated maps.

Scientific intent
-----------------
- Main manuscript: one clean representative set from the final 11M model.
- Supplementary Information: bulk map generation + statistical summaries from the same model.
- Patch-level attribution only. These are not segmentation masks and not whole-slide localization maps.
'''
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# -------------------------
# Data helpers
# -------------------------
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
    for k in ['x', 'X', 'data', 'images', 'y', 'Y', 'labels']:
        if k in h5:
            return k
    if not keys:
        raise ValueError('H5 file has no dataset keys.')
    return keys[0]


class PCamReader:
    def __init__(self, root: str, split: str = 'test'):
        self.root = Path(root)
        self.split = split
        self.x_path = find_file(self.root, split, 'x')
        self.y_path = find_file(self.root, split, 'y')
        self.fx = h5py.File(self.x_path, 'r')
        self.fy = h5py.File(self.y_path, 'r')
        self.x_key = first_dataset_key(self.fx)
        self.y_key = first_dataset_key(self.fy)
        self.n = int(self.fx[self.x_key].shape[0])

    def __len__(self):
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
            x, y = self.get(idx)
            xs.append(x)
            ys.append(y)
        return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


# -------------------------
# Model helpers
# -------------------------
def import_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location('final_model_module', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not import script: {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def try_build_model(mod, variant: str):
    for name in ['build_model_from_variant', 'build_model', 'make_model', 'create_model', 'get_model']:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                sig = inspect.signature(fn)
                if 'variant' in sig.parameters:
                    return fn(variant=variant)
                if len(sig.parameters) == 1:
                    return fn(variant)
                if len(sig.parameters) == 0:
                    return fn()
            except Exception:
                pass
    for name in ['MetaPathModel', 'PathEOMNetR', 'PathEOMNet', 'MetaPathNet',
                 'RegularizedRedesignModel', 'PCamModel']:
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        try:
            sig = inspect.signature(cls)
            if 'variant' in sig.parameters:
                return cls(variant=variant)
            if len(sig.parameters) == 1:
                return cls(variant)
            if len(sig.parameters) == 0:
                return cls()
        except Exception:
            pass
    raise RuntimeError(
        'Could not build the model from the training script. '
        'Send the error together with the training script if patching is needed.'
    )


def load_model(train_script: str, checkpoint: str, variant: str, device: torch.device):
    mod = import_module_from_path(train_script)
    model = try_build_model(mod, variant)
    model.to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    info = {'missing_keys': list(missing), 'unexpected_keys': list(unexpected)}
    return model, info


def model_logit(model, xt: torch.Tensor) -> torch.Tensor:
    out = model(xt)
    if out.ndim > 1:
        out = out[:, 0]
    return out


# -------------------------
# Prediction / metrics helpers
# -------------------------
def run_predictions(model, reader: PCamReader, device: torch.device, batch_size: int, threshold: float) -> pd.DataFrame:
    rows = []
    n = len(reader)
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
                'index': int(idx),
                'y_true': int(y),
                'probability': float(p),
                'selected_threshold': float(threshold),
                'pred_selected': int(pred),
            })
        if (start // batch_size) % 20 == 0:
            print(f'Predictions: {end}/{n}')
    return pd.DataFrame(rows)


def compute_binary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = df['y_true'].astype(int).to_numpy()
    p = df['probability'].astype(float).to_numpy()
    pred = df['pred_selected'].astype(int).to_numpy()
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    acc = (tp + tn) / max(len(df), 1)
    bacc = 0.5 * (sens + spec)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)

    # Brier
    brier = float(np.mean((p - y) ** 2))

    # ECE
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        if i < len(bins)-2:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        if mask.any():
            conf = float(np.mean(p[mask]))
            acc_bin = float(np.mean(y[mask] == pred[mask]))
            ece += abs(conf - acc_bin) * mask.mean()

    # ROC-AUC / PR-AUC via rank-based / sklearn-free approximations not reliable.
    # If user already has summary values, use those separately. Here we keep thresholded metrics robust.
    return {
        'n_total': int(len(df)), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': float(acc), 'balanced_accuracy': float(bacc),
        'sensitivity': float(sens), 'specificity': float(spec),
        'precision': float(prec), 'positive_f1': float(f1),
        'brier': float(brier), 'ece_10bin': float(ece),
    }


def label_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    conds = [
        (df.y_true == 1) & (df.pred_selected == 1),
        (df.y_true == 0) & (df.pred_selected == 0),
        (df.y_true == 0) & (df.pred_selected == 1),
        (df.y_true == 1) & (df.pred_selected == 0),
    ]
    cats = ['TP', 'TN', 'FP', 'FN']
    df['category'] = np.select(conds, cats, default='OTHER')
    return df


def pick_main_examples(df: pd.DataFrame, n_each: int = 1) -> pd.DataFrame:
    """Pick one clean representative set for the main manuscript.
    TP/TN: representative but strong examples.
    FP/FN: clearer error examples.
    """
    chosen = []
    for cat in ['TP', 'TN', 'FP', 'FN']:
        sub = df[df['category'] == cat].copy()
        if len(sub) == 0:
            continue
        if cat == 'TP':
            sub = sub.sort_values('probability', ascending=False)
            lo = int(0.10 * len(sub)); hi = max(lo + n_each, int(0.35 * len(sub)))
            block = sub.iloc[lo:hi] if hi > lo else sub.head(n_each)
            picked = block.head(n_each)
        elif cat == 'TN':
            sub = sub.sort_values('probability', ascending=True)
            lo = int(0.10 * len(sub)); hi = max(lo + n_each, int(0.35 * len(sub)))
            block = sub.iloc[lo:hi] if hi > lo else sub.head(n_each)
            picked = block.head(n_each)
        elif cat == 'FP':
            picked = sub.sort_values('probability', ascending=False).head(n_each)
        else:  # FN
            picked = sub.sort_values('probability', ascending=True).head(n_each)
        chosen.append(picked)
    if not chosen:
        raise RuntimeError('No TP/TN/FP/FN examples could be selected.')
    return pd.concat(chosen, axis=0).reset_index(drop=True)


# -------------------------
# Attribution helpers
# -------------------------
def make_attribution_cmap():
    return LinearSegmentedColormap.from_list(
        'attrib_bgyr',
        [
            (0.00, '#2747d9'),   # blue
            (0.35, '#28b463'),   # green
            (0.68, '#f4d03f'),   # yellow
            (0.83, '#eb984e'),   # orange
            (1.00, '#c0392b'),   # red
        ]
    )


def compute_smoothgrad_saliency(model, x_chw: np.ndarray, device: torch.device,
                                n_smooth: int = 16, noise_std: float = 0.03) -> Tuple[float, np.ndarray]:
    x0 = torch.tensor(x_chw[None], dtype=torch.float32, device=device)
    grads = []
    prob_value = None
    n_smooth = max(1, int(n_smooth))
    for _ in range(n_smooth):
        if n_smooth > 1:
            xt = torch.clamp(x0 + torch.randn_like(x0) * noise_std, 0.0, 1.0).detach()
        else:
            xt = x0.detach()
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
    lo = np.percentile(sal, 5)
    hi = np.percentile(sal, 99.5)
    sal = (sal - lo) / (hi - lo + 1e-12)
    sal = np.clip(sal, 0.0, 1.0)
    return float(prob_value), sal


def attribution_stats(sal: np.ndarray) -> Dict[str, float]:
    h, w = sal.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    center_mask = rr <= 0.35 * min(h, w)
    border_mask = rr >= 0.45 * min(h, w)
    return {
        'saliency_mean': float(np.mean(sal)),
        'saliency_std': float(np.std(sal)),
        'saliency_max': float(np.max(sal)),
        'hotspot_frac_gt_050': float(np.mean(sal >= 0.50)),
        'hotspot_frac_gt_075': float(np.mean(sal >= 0.75)),
        'center_saliency_mean': float(np.mean(sal[center_mask])) if center_mask.any() else float('nan'),
        'border_saliency_mean': float(np.mean(sal[border_mask])) if border_mask.any() else float('nan'),
    }


def chw_to_rgb(x_chw: np.ndarray) -> np.ndarray:
    return np.transpose(x_chw, (1, 2, 0))


def add_scale_bar(ax, image_shape_hw: Tuple[int, int], mpp: float,
                  scalebar_um: float = 25.0, bar_color: str = 'black', text_color: str = 'black'):
    h, w = image_shape_hw
    bar_px = int(round(scalebar_um / mpp))
    bar_px = max(5, min(bar_px, w // 2))
    x0 = w - bar_px - 8
    y0 = h - 9
    outline = 'white' if bar_color.lower() == 'black' else 'black'
    ax.add_patch(Rectangle((x0 - 1, y0 - 1), bar_px + 2, 4, color=outline, alpha=0.55, linewidth=0))
    ax.add_patch(Rectangle((x0, y0), bar_px, 2, color=bar_color, linewidth=0))
    ax.text(x0 + bar_px / 2, y0 - 4, f'{int(scalebar_um)} µm',
            color=text_color, ha='center', va='bottom', fontsize=8, weight='bold')


def overlay_saliency(raw_rgb: np.ndarray, sal: np.ndarray, bg_opacity: float = 0.22,
                     low_alpha: float = 0.16, max_alpha: float = 0.88,
                     saliency_floor: float = 0.00, gamma: float = 1.0,
                     saliency_cap: float = 0.98, show_low_background: bool = True) -> np.ndarray:
    raw_rgb = np.clip(raw_rgb, 0.0, 1.0)
    sal = np.clip(sal, 0.0, 1.0)
    if saliency_cap < 1.0:
        sal = np.clip(sal / max(saliency_cap, 1e-6), 0.0, 1.0)
    cmap = make_attribution_cmap()
    heat = cmap(sal)[..., :3]
    alpha_map = np.clip((sal - saliency_floor) / max(1.0 - saliency_floor, 1e-6), 0.0, 1.0)
    alpha_map = alpha_map ** gamma
    if show_low_background:
        alpha_map = low_alpha + (max_alpha - low_alpha) * alpha_map
    else:
        alpha_map = max_alpha * alpha_map
    raw_faded = bg_opacity * raw_rgb + (1.0 - bg_opacity) * 1.0
    out = (1.0 - alpha_map[..., None]) * raw_faded + alpha_map[..., None] * heat
    return np.clip(out, 0.0, 1.0)


def render_single(raw_rgb: np.ndarray, sal: Optional[np.ndarray], out_png: Path,
                  title: str, mpp: float, scalebar_um: float,
                  bg_opacity: float, low_alpha: float, max_alpha: float,
                  saliency_floor: float, gamma: float, saliency_cap: float,
                  show_low_background: bool, scale_bar_color: str = 'black',
                  scale_text_color: str = 'black', overlay: bool = False):
    fig = plt.figure(figsize=(3.0, 3.2))
    ax = plt.gca()
    if overlay and sal is not None:
        img = overlay_saliency(raw_rgb, sal, bg_opacity, low_alpha, max_alpha,
                               saliency_floor, gamma, saliency_cap, show_low_background)
        ax.imshow(img, interpolation='nearest')
    else:
        ax.imshow(raw_rgb, interpolation='nearest')
    add_scale_bar(ax, raw_rgb.shape[:2], mpp, scalebar_um, scale_bar_color, scale_text_color)
    ax.set_title(title, fontsize=9, weight='bold')
    ax.axis('off')
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def render_main_panel(examples: List[Dict], out_png: Path,
                      mpp: float, scalebar_um: float,
                      bg_opacity: float, low_alpha: float, max_alpha: float,
                      saliency_floor: float, gamma: float, saliency_cap: float,
                      show_low_background: bool):
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(8.6, 3.1 * n))
    if n == 1:
        axes = np.array([axes])
    letter_ord = ord('a')
    for i, ex in enumerate(examples):
        raw_rgb = ex['raw_rgb']
        sal = ex['saliency']
        overlay = overlay_saliency(raw_rgb, sal, bg_opacity, low_alpha, max_alpha,
                                   saliency_floor, gamma, saliency_cap, show_low_background)
        cat = ex['category']; y = ex['y_true']; pred = ex['pred_selected']; p = ex['probability']
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.imshow(raw_rgb, interpolation='nearest')
        ax1.imshow(overlay, interpolation='nearest')
        add_scale_bar(ax0, raw_rgb.shape[:2], mpp, scalebar_um, 'black', 'black')
        add_scale_bar(ax1, raw_rgb.shape[:2], mpp, scalebar_um, 'black', 'black')
        ax0.set_title(f'{cat} raw\ntrue={y}, pred={pred}, p={p:.3f}', fontsize=10, weight='bold')
        ax1.set_title(f'{cat} attribution\ntrue={y}, pred={pred}, p={p:.3f}', fontsize=10, weight='bold')
        ax0.axis('off'); ax1.axis('off')
        ax0.text(-0.12, 1.02, chr(letter_ord), transform=ax0.transAxes, fontsize=14, weight='bold')
        ax1.text(-0.12, 1.02, chr(letter_ord + 1), transform=ax1.transAxes, fontsize=14, weight='bold')
        letter_ord += 2
    plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.34, wspace=0.08)
    fig.suptitle('Representative final 11M PathEOM-Net-R patch-level attribution maps', fontsize=14, weight='bold')
    cax = fig.add_axes([0.18, 0.035, 0.64, 0.017])
    gradient = np.linspace(0, 1, 512)[None, :]
    cax.imshow(gradient, aspect='auto', cmap=make_attribution_cmap())
    cax.set_xticks([]); cax.set_yticks([])
    fig.text(0.13, 0.043, 'Blue: low', ha='left', va='center', fontsize=8.5)
    fig.text(0.35, 0.043, 'Green: moderate', ha='center', va='center', fontsize=8.5)
    fig.text(0.62, 0.043, 'Yellow/orange: high', ha='center', va='center', fontsize=8.5)
    fig.text(0.83, 0.043, 'Red: very high', ha='left', va='center', fontsize=8.5)
    fig.text(0.50, 0.018, 'Attribution strength / local model sensitivity (not pixel-level metastasis probability)',
             ha='center', va='center', fontsize=8.5)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--train-script', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--variant', default='p7_r2_sens_reg_11m')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--threshold', type=float, default=0.220,
                    help='Final selected threshold for the 11M model.')
    ap.add_argument('--main-n-each', type=int, default=1)
    ap.add_argument('--main-smoothgrad-n', type=int, default=16)
    ap.add_argument('--all-smoothgrad-n', type=int, default=1,
                    help='Use 1 for faster bulk generation on CPU.')
    ap.add_argument('--noise-std', type=float, default=0.03)
    ap.add_argument('--mpp', type=float, default=0.972)
    ap.add_argument('--scalebar-um', type=float, default=25.0)
    ap.add_argument('--bg-opacity', type=float, default=0.22)
    ap.add_argument('--low-alpha', type=float, default=0.16)
    ap.add_argument('--max-alpha', type=float, default=0.88)
    ap.add_argument('--saliency-floor', type=float, default=0.00)
    ap.add_argument('--alpha-gamma', type=float, default=1.0)
    ap.add_argument('--saliency-cap', type=float, default=0.98)
    ap.add_argument('--show-low-background', action='store_true')
    ap.add_argument('--generate-all-maps', action='store_true')
    ap.add_argument('--max-all-maps', type=int, default=-1,
                    help='Set to a positive number for a capped dry run; -1 means full test split.')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    main_dir = outdir / 'main_manuscript_maps'
    all_dir = outdir / 'all_test_maps'
    stats_dir = outdir / 'summary_stats'
    pred_dir = outdir / 'predictions'
    for d in [main_dir, all_dir, stats_dir, pred_dir]:
        d.mkdir(parents=True, exist_ok=True)
    (all_dir / 'raw').mkdir(exist_ok=True)
    (all_dir / 'maps').mkdir(exist_ok=True)
    for cat in ['TP', 'TN', 'FP', 'FN']:
        (all_dir / 'raw' / cat).mkdir(parents=True, exist_ok=True)
        (all_dir / 'maps' / cat).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    reader = PCamReader(args.root, split=args.split)
    model, load_info = load_model(args.train_script, args.checkpoint, args.variant, device)

    # 1) Predictions for the full split
    pred_df = run_predictions(model, reader, device, args.batch_size, args.threshold)
    pred_df = label_categories(pred_df)
    pred_csv = pred_dir / 'final11m_test_predictions.csv'
    pred_df.to_csv(pred_csv, index=False)

    # 2) Performance summaries
    metrics = compute_binary_metrics(pred_df)
    metrics['threshold'] = float(args.threshold)
    metrics['variant'] = args.variant
    metrics['checkpoint'] = str(args.checkpoint)
    metrics['n_test_examples'] = int(len(pred_df))
    (stats_dir / 'final11m_performance_summary.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    pd.DataFrame([metrics]).to_csv(stats_dir / 'final11m_performance_summary.csv', index=False)

    cat_counts = pred_df.groupby('category').size().reset_index(name='count')
    cat_counts.to_csv(stats_dir / 'category_counts.csv', index=False)
    prob_summary = pred_df.groupby('category')['probability'].agg(['count', 'mean', 'std', 'median', 'min', 'max']).reset_index()
    prob_summary.to_csv(stats_dir / 'probability_summary_by_category.csv', index=False)

    # 3) Main manuscript example selection
    main_selected = pick_main_examples(pred_df, n_each=args.main_n_each)
    main_selected.to_csv(main_dir / 'main_selected_examples.csv', index=False)

    main_examples = []
    for _, r in main_selected.iterrows():
        idx = int(r['index'])
        x_chw, y = reader.get(idx)
        raw_rgb = chw_to_rgb(x_chw)
        prob, sal = compute_smoothgrad_saliency(model, x_chw, device,
                                                n_smooth=args.main_smoothgrad_n,
                                                noise_std=args.noise_std)
        prob = float(r['probability'])
        pred = int(r['pred_selected'])
        cat = str(r['category'])
        base = f"{cat}_idx{idx}_true{y}_pred{pred}_p{prob:.3f}"
        render_single(raw_rgb, None, main_dir / f'{base}_raw.png',
                      f'{cat} raw | true={y}, pred={pred}, p={prob:.3f}', args.mpp, args.scalebar_um,
                      args.bg_opacity, args.low_alpha, args.max_alpha, args.saliency_floor,
                      args.alpha_gamma, args.saliency_cap, args.show_low_background, overlay=False)
        render_single(raw_rgb, sal, main_dir / f'{base}_attribution.png',
                      f'{cat} attribution | true={y}, pred={pred}, p={prob:.3f}', args.mpp, args.scalebar_um,
                      args.bg_opacity, args.low_alpha, args.max_alpha, args.saliency_floor,
                      args.alpha_gamma, args.saliency_cap, args.show_low_background, overlay=True)
        ex = {
            'category': cat, 'index': idx, 'y_true': y, 'pred_selected': pred,
            'probability': prob, 'raw_rgb': raw_rgb, 'saliency': sal,
        }
        main_examples.append(ex)
    render_main_panel(main_examples, main_dir / 'figure_main_final11m_attribution_panel.png',
                      args.mpp, args.scalebar_um, args.bg_opacity, args.low_alpha, args.max_alpha,
                      args.saliency_floor, args.alpha_gamma, args.saliency_cap, args.show_low_background)

    # 4) Optional bulk all-map generation
    all_records = []
    if args.generate_all_maps:
        full_df = pred_df.copy()
        if args.max_all_maps > 0:
            full_df = full_df.iloc[:args.max_all_maps].copy()
        print(f'Generating all maps for {len(full_df)} samples...')
        for i, r in full_df.iterrows():
            idx = int(r['index']); cat = str(r['category'])
            x_chw, y = reader.get(idx)
            raw_rgb = chw_to_rgb(x_chw)
            prob, sal = compute_smoothgrad_saliency(model, x_chw, device,
                                                    n_smooth=args.all_smoothgrad_n,
                                                    noise_std=args.noise_std)
            prob = float(r['probability']); pred = int(r['pred_selected'])
            base = f"{cat}_idx{idx}_true{y}_pred{pred}_p{prob:.3f}"
            raw_png = all_dir / 'raw' / cat / f'{base}_raw.png'
            map_png = all_dir / 'maps' / cat / f'{base}_attribution.png'
            render_single(raw_rgb, None, raw_png,
                          f'{cat} raw | true={y}, pred={pred}, p={prob:.3f}', args.mpp, args.scalebar_um,
                          args.bg_opacity, args.low_alpha, args.max_alpha, args.saliency_floor,
                          args.alpha_gamma, args.saliency_cap, args.show_low_background, overlay=False)
            render_single(raw_rgb, sal, map_png,
                          f'{cat} attribution | true={y}, pred={pred}, p={prob:.3f}', args.mpp, args.scalebar_um,
                          args.bg_opacity, args.low_alpha, args.max_alpha, args.saliency_floor,
                          args.alpha_gamma, args.saliency_cap, args.show_low_background, overlay=True)
            st = attribution_stats(sal)
            all_records.append({
                'index': idx, 'category': cat, 'y_true': y, 'pred_selected': pred,
                'probability': prob, 'raw_png': str(raw_png), 'map_png': str(map_png), **st
            })
            if (len(all_records) % 250) == 0:
                print(f'All maps: {len(all_records)}/{len(full_df)}')
        all_stats_df = pd.DataFrame(all_records)
        all_stats_df.to_csv(stats_dir / 'all_map_statistics.csv', index=False)
        if len(all_stats_df):
            attr_summary = all_stats_df.groupby('category')[
                ['saliency_mean', 'saliency_std', 'saliency_max', 'hotspot_frac_gt_050',
                 'hotspot_frac_gt_075', 'center_saliency_mean', 'border_saliency_mean']
            ].agg(['mean', 'std', 'median']).reset_index()
            attr_summary.to_csv(stats_dir / 'attribution_summary_by_category.csv', index=False)

    manifest = {
        'variant': args.variant,
        'checkpoint': args.checkpoint,
        'train_script': args.train_script,
        'threshold': args.threshold,
        'device': str(device),
        'split': args.split,
        'n_test_examples': int(len(pred_df)),
        'generate_all_maps': bool(args.generate_all_maps),
        'max_all_maps': int(args.max_all_maps),
        'main_smoothgrad_n': int(args.main_smoothgrad_n),
        'all_smoothgrad_n': int(args.all_smoothgrad_n),
        'noise_std': float(args.noise_std),
        'mpp': float(args.mpp),
        'scalebar_um': float(args.scalebar_um),
        'bg_opacity': float(args.bg_opacity),
        'low_alpha': float(args.low_alpha),
        'max_alpha': float(args.max_alpha),
        'saliency_floor': float(args.saliency_floor),
        'alpha_gamma': float(args.alpha_gamma),
        'saliency_cap': float(args.saliency_cap),
        'show_low_background': bool(args.show_low_background),
        'model_load_info': load_info,
        'key_outputs': {
            'predictions_csv': str(pred_csv),
            'main_selected_examples_csv': str(main_dir / 'main_selected_examples.csv'),
            'main_panel_png': str(main_dir / 'figure_main_final11m_attribution_panel.png'),
            'performance_summary_json': str(stats_dir / 'final11m_performance_summary.json'),
            'all_map_statistics_csv': str(stats_dir / 'all_map_statistics.csv') if args.generate_all_maps else '',
        },
        'important_note': 'Patch-level attribution only; not whole-slide localization and not segmentation.'
    }
    (outdir / 'pcam_attribution_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Done.')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()



