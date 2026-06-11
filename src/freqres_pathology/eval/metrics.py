from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_score, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (y_score >= lower) & (y_score <= upper)
        else:
            mask = (y_score >= lower) & (y_score < upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_score[mask]))
        accuracy = float(np.mean(y_true[mask]))
        error += (np.sum(mask) / len(y_true)) * abs(accuracy - confidence)
    return float(error)


def binary_metrics(y_true, y_score, threshold: float = 0.5, n_bins: int = 15) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    metrics = {
        "threshold": threshold,
        "n": int(len(y_true)),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_ppv": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "npv": npv,
        "positive_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score, n_bins=n_bins),
        "nll_log_loss": log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return {key: (float(value) if isinstance(value, np.floating) else value) for key, value in metrics.items()}


def metrics_to_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])
