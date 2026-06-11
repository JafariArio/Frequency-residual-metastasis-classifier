from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def mcnemar_continuity(y_true, pred_a, pred_b) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    denom = n01 + n10
    if denom == 0:
        return {"mcnemar_n01": n01, "mcnemar_n10": n10, "mcnemar_chi2": 0.0, "mcnemar_p": 1.0}
    statistic = (abs(n01 - n10) - 1) ** 2 / denom
    return {
        "mcnemar_n01": n01,
        "mcnemar_n10": n10,
        "mcnemar_chi2": float(statistic),
        "mcnemar_p": float(1 - chi2.cdf(statistic, df=1)),
    }
