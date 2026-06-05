"""Minimal evaluation metrics for thesis simulation experiments.
already cleaned
"""


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def make_binary_labels(
    metadata: pd.DataFrame,
    row_labels: pd.DataFrame,
    *,
    outlier_type: str | None = None,
    id_column: str = "gebernummer",
    date_column: str = "reference_date_int",
) -> np.ndarray:
    """Return binary labels aligned to the rows of metadata."""
    label_column = "is_any_anomaly" if outlier_type is None else f"is_{outlier_type}_anomaly"

    labels = row_labels[[id_column, date_column, label_column]].copy()
    labels[date_column] = labels[date_column].astype(str)

    meta = metadata[[id_column, date_column]].copy()
    meta[date_column] = meta[date_column].astype(str)

    merged = meta.merge(labels, on=[id_column, date_column], how="left")
    return merged[label_column].fillna(False).astype(int).to_numpy()


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of true anomalies among the top-k scores."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    k = min(int(k), len(scores))
    top = np.argsort(scores)[::-1][:k]
    return float(y_true[top].mean()) if k > 0 else np.nan


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of all true anomalies found among the top-k scores."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    positives = int(y_true.sum())
    k = min(int(k), len(scores))
    top = np.argsort(scores)[::-1][:k]
    return float(y_true[top].sum() / positives) if positives > 0 and k > 0 else np.nan


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC, returning NaN if only one class is present."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    return np.nan if y_true.sum() in (0, len(y_true)) else float(roc_auc_score(y_true, scores))


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    k: int | None = None,
) -> pd.DataFrame:
    """Return one compact metric row.

    By default, k equals the number of true anomalies.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    k = int(y_true.sum()) if k is None else int(k)

    return pd.DataFrame([
        {
            "n": int(len(y_true)),
            "n_anomalies": int(y_true.sum()),
            "k": k,
            "precision_at_k": precision_at_k(y_true, scores, k),
            "recall_at_k": recall_at_k(y_true, scores, k),
            "roc_auc": safe_roc_auc(y_true, scores),
        }
    ])
