from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import binary_metrics


def stratified_bootstrap_ci(
    y_true,
    y_score,
    threshold: float,
    n_bootstrap: int = 2000,
    seed: int = 7,
    alpha: float = 0.05,
    metric_names=None,
):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    rng = np.random.default_rng(seed)
    positive = np.where(y_true == 1)[0]
    negative = np.where(y_true == 0)[0]

    if metric_names is None:
        metric_names = [
            "roc_auc",
            "pr_auc",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "positive_f1",
            "mcc",
            "brier_score",
            "ece",
        ]

    bootstrap_rows = []
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        index = np.concatenate([
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ])
        metrics = binary_metrics(y_true[index], y_score[index], threshold=threshold)
        bootstrap_rows.append({name: metrics[name] for name in metric_names})

    bootstrap = pd.DataFrame(bootstrap_rows)
    point = binary_metrics(y_true, y_score, threshold=threshold)
    intervals = []
    for name in metric_names:
        intervals.append(
            {
                "metric": name,
                "point": point[name],
                "ci_low": float(np.nanpercentile(bootstrap[name], 100 * alpha / 2)),
                "ci_high": float(np.nanpercentile(bootstrap[name], 100 * (1 - alpha / 2))),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(intervals), bootstrap
