"""Pure metric and structured-table shaping for experiment logging."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


CLASSIFICATION_REPORT_COLUMNS = (
    "model",
    "task",
    "train_condition",
    "test_condition",
    "condition_relation",
    "class_name",
    "precision",
    "recall",
    "f1",
    "support",
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "ratio_to_original",
    "adaptation_gain",
)

_REPORT_AGGREGATES = frozenset({"accuracy", "macro avg", "weighted avg"})


def numeric_metrics(
    prefix: str, metrics: Mapping[str, Any]
) -> dict[str, int | float]:
    output: dict[str, int | float] = {}
    for key, value in metrics.items():
        if isinstance(value, numbers.Integral):
            output[f"{prefix}/{key}"] = int(value)
        elif isinstance(value, numbers.Real):
            output[f"{prefix}/{key}"] = float(value)
    return output


def robustness_ratio(transformed: Any, original: Any) -> float:
    """Compute a safe transformed/original ratio for robustness logging."""
    try:
        transformed_value = float(transformed)
        original_value = float(original)
    except (TypeError, ValueError):
        return float("nan")
    if (
        not math.isfinite(transformed_value)
        or not math.isfinite(original_value)
        or original_value == 0.0
    ):
        return float("nan")
    return transformed_value / original_value


def valid_confusion_labels(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    class_count: int,
) -> tuple[list[int], list[int]]:
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    for true_value, pred_value in zip(y_true, y_pred):
        try:
            true_index = int(true_value)
            pred_index = int(pred_value)
        except (TypeError, ValueError):
            continue
        if 0 <= true_index < class_count and 0 <= pred_index < class_count:
            true_labels.append(true_index)
            predicted_labels.append(pred_index)
    return true_labels, predicted_labels


def classification_report_rows(
    report: Any,
    *,
    model: Any,
    task: str,
    train_condition: str,
    test_condition: str,
    condition_relation: str,
    metrics: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if hasattr(report, "reset_index") and hasattr(report, "to_dict"):
        raw_rows = report.reset_index().to_dict(orient="records")
    elif isinstance(report, Mapping):
        raw_rows = []
        for class_name, values in report.items():
            if class_name in _REPORT_AGGREGATES or not isinstance(
                values, Mapping
            ):
                continue
            raw_rows.append({"class_name": class_name, **dict(values)})
    else:
        raw_rows = [dict(row) for row in report]

    metrics = metrics or {}
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        class_name = raw.get(
            "class_name", raw.get("index", raw.get("label"))
        )
        if class_name in _REPORT_AGGREGATES or class_name is None:
            continue
        rows.append({
            "model": model,
            "task": task,
            "train_condition": train_condition,
            "test_condition": test_condition,
            "condition_relation": condition_relation,
            "class_name": class_name,
            "precision": raw.get("precision"),
            "recall": raw.get("recall"),
            "f1": raw.get("f1", raw.get("f1-score")),
            "support": raw.get("support"),
            "macro_f1": metrics.get(f"{task}_macro_f1"),
            "balanced_accuracy": metrics.get(
                f"{task}_balanced_accuracy"
            ),
            "accuracy": metrics.get(f"{task}_accuracy"),
            "ratio_to_original": raw.get("ratio_to_original"),
            "adaptation_gain": raw.get("adaptation_gain"),
        })
    return rows


def canonical_table_records(rows: Any) -> list[dict[str, Any]]:
    """Return unique canonical columns, keeping the first duplicate column."""
    if hasattr(rows, "columns") and hasattr(rows, "iloc"):
        names = [str(name) for name in rows.columns]
        positions: list[int] = []
        seen: set[str] = set()
        for index, name in enumerate(names):
            if name not in seen:
                seen.add(name)
                positions.append(index)
        records = rows.iloc[:, positions].to_dict(orient="records")
    elif hasattr(rows, "to_dict"):
        records = rows.to_dict(orient="records")
    else:
        records = [dict(row) for row in rows]

    canonical: list[dict[str, Any]] = []
    for source in records:
        row: dict[str, Any] = {}
        aliases = {
            "condition": "test_condition",
            "evaluation_relation": "condition_relation",
            "f1-score": "f1",
        }
        for key, value in source.items():
            canonical_key = aliases.get(str(key), str(key))
            row.setdefault(canonical_key, value)
        canonical.append(row)
    return canonical


def unique_columns(
    rows: Iterable[Mapping[str, Any]],
    preferred: Sequence[str] | None = None,
) -> list[str]:
    available = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    if not preferred:
        return available
    ordered = [column for column in dict.fromkeys(preferred) if column in available]
    ordered.extend(column for column in available if column not in ordered)
    return ordered


__all__ = [
    "CLASSIFICATION_REPORT_COLUMNS",
    "canonical_table_records",
    "classification_report_rows",
    "numeric_metrics",
    "robustness_ratio",
    "unique_columns",
    "valid_confusion_labels",
]
