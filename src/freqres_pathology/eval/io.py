from __future__ import annotations

import pandas as pd

TRUE_CANDIDATES = ["y_true", "label", "target", "actual", "true_label"]
SCORE_CANDIDATES = ["y_score", "score", "prob", "probability", "positive_probability", "pred_prob", "p1"]
ID_CANDIDATES = ["sample_id", "id", "patch_id", "image_id"]


def infer_column(columns, candidates):
    by_lowercase = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in by_lowercase:
            return by_lowercase[candidate.lower()]
    return None


def load_prediction_csv(path, y_true_col=None, y_score_col=None):
    frame = pd.read_csv(path)
    y_true_col = y_true_col or infer_column(frame.columns, TRUE_CANDIDATES)
    y_score_col = y_score_col or infer_column(frame.columns, SCORE_CANDIDATES)
    if y_true_col is None or y_score_col is None:
        raise ValueError("Could not infer label and score columns. Pass column names explicitly.")
    return frame, y_true_col, y_score_col
