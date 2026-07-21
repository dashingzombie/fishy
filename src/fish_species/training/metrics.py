"""Small metric helpers whose edge-case semantics are shared by all trainers."""

from __future__ import annotations

import math

import numpy as np


def safe_metric(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    default: float = float("nan"),
) -> float:
    if len(y_true) == 0:
        return default
    return float(metric_fn(y_true, y_pred))


def score_for_selection(metrics: dict, selection_metric: str) -> float:
    value = float(metrics.get(selection_metric, float("nan")))
    if math.isnan(value):
        return -float("inf")
    return value
