"""The six threshold-free measures of the CATCH / TAB benchmark, computed as in their code.

AUC-ROC and AUC-PR use scikit-learn (AUC-PR = average precision).  For R-AUC-ROC / R-AUC-PR the sliding window is the
median length of the anomaly segments in the test labels; VUS-ROC / VUS-PR use twice that window.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics

from .vus_metrics import generate_curve, metricor


def get_list_anomaly(labels: np.ndarray):
    """Lengths of the anomaly segments (same as ts_benchmark/evaluation/metrics/utils.py in CATCH / TAB)."""
    end_pos = np.diff(np.array(labels, dtype=int), append=0) < 0
    return np.diff(np.cumsum(labels)[end_pos], prepend=0)


def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    window = int(np.median(get_list_anomaly(labels)))
    r_auc_roc, r_auc_pr, *_ = metricor().RangeAUC(labels=labels, score=scores, window=window, plot_ROC=True)
    *_, vus_roc, vus_pr = generate_curve(labels, scores, 2 * window)
    return {
        "AUC-ROC": float(metrics.roc_auc_score(labels, scores)),
        "AUC-PR": float(metrics.average_precision_score(labels, scores)),
        "R-AUC-ROC": float(r_auc_roc),
        "R-AUC-PR": float(r_auc_pr),
        "VUS-ROC": float(vus_roc),
        "VUS-PR": float(vus_pr),
        "sliding_window": window,
    }
