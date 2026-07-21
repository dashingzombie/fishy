"""Build dashboard-ready summaries from existing evaluation artefacts.

This module is deliberately post-processing only.  It reads lightweight result
metadata, classification reports, and confusion-matrix CSV files discovered by
``fish_species.results``.  It never opens a checkpoint, runs inference, or
writes beneath a scientific result directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .discovery import discover_results_root
from .normalization import canonical_condition_relation
from .readers import load_csv_rows
from .schemas import ArtifactRecord, RunRecord

SCHEMA_VERSION = 2
SOURCE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TASK_ALIASES = {
    "genus": "genus",
    "species": "species",
    "species_label": "species",
}


@dataclass(frozen=True)
class MatrixData:
    task: str
    path: str
    classes: tuple[str, ...]
    counts: tuple[tuple[float, ...], ...]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_cache_root(cache_root: str | Path, results_roots: Iterable[str | Path]) -> Path:
    """Reject cache locations within any scientific result tree."""

    cache = Path(cache_root).expanduser().resolve(strict=False)
    for value in results_roots:
        root = Path(value).expanduser().resolve(strict=True)
        if cache == root or _is_relative_to(cache, root):
            raise ValueError(f"derived cache must be outside results root: {root}")
    return cache


def parse_source(value: str) -> tuple[str, Path]:
    """Parse one ``LABEL=PATH`` command-line source."""

    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise ValueError("source must use LABEL=PATH syntax")
    if not SOURCE_LABEL_PATTERN.fullmatch(label):
        raise ValueError(f"invalid source label: {label!r}")
    path = Path(raw_path).expanduser().absolute()
    if not path.is_dir():
        raise ValueError(f"results root is not a directory: {path}")
    return label, path


def _warning(code: str, message: str, path: str | None = None) -> dict[str, str]:
    value = {"code": code, "message": message}
    if path is not None:
        value["path"] = path
    return value


def _task_from_name(name: str, prefix: str) -> str | None:
    stem = Path(name).stem
    if stem == prefix:
        return None
    marker = prefix + "_"
    return stem[len(marker) :] if stem.startswith(marker) else None


def _configured_single_task(run: RunRecord) -> str | None:
    data = run.config.get("data", {})
    if not isinstance(data, dict):
        return None
    value = data.get("target_col")
    return TASK_ALIASES.get(str(value)) if value is not None else None


def _metric(run: RunRecord, name: str, task: str | None = None) -> float | None:
    for item in run.metrics:
        if item.metric == name and item.task == task:
            return item.value
    return None


def _read_matrix(path: Path, task: str) -> MatrixData:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("matrix CSV has no labelled matrix")
    classes = tuple(cell.strip() for cell in rows[0][1:])
    if not classes or any(not item for item in classes):
        raise ValueError("matrix CSV has empty class names")
    if len(rows) != len(classes) + 1:
        raise ValueError("matrix CSV is not square")
    row_classes: list[str] = []
    matrix: list[tuple[float, ...]] = []
    for row in rows[1:]:
        if len(row) != len(classes) + 1:
            raise ValueError("matrix CSV has an inconsistent row width")
        row_classes.append(row[0].strip())
        values: list[float] = []
        for cell in row[1:]:
            value = float(cell)
            if not math.isfinite(value) or value < 0:
                raise ValueError("matrix CSV contains an invalid count")
            values.append(value)
        matrix.append(tuple(values))
    if tuple(row_classes) != classes:
        raise ValueError("matrix row and column labels differ")
    return MatrixData(task=task, path=str(path), classes=classes, counts=tuple(matrix))


def _report_macro_f1(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or "f1-score" not in rows.fieldnames:
            raise ValueError("classification report has no f1-score column")
        label_column = rows.fieldnames[0]
        for row in rows:
            if row.get(label_column, "").strip().lower() == "macro avg":
                value = float(row["f1-score"])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("classification report macro F1 is outside [0, 1]")
                return value
    raise ValueError("classification report has no macro avg row")


def _artifact_task(artifact: ArtifactRecord, prefix: str, default: str | None) -> str | None:
    named = _task_from_name(Path(artifact.path).name, prefix)
    if named:
        return TASK_ALIASES.get(named, named)
    return default


def _matrix_artifact_condition(artifact: ArtifactRecord, category: str) -> str | None:
    marker = f"condition_matrix/{category}/"
    if not artifact.kind.startswith(marker):
        return None
    remainder = artifact.kind[len(marker) :]
    condition, separator, _ = remainder.partition("/")
    return condition if separator and condition else None


def _condition_cache_directory(run_cache: Path, condition: str) -> Path:
    digest = hashlib.sha256(condition.encode("utf-8")).hexdigest()[:16]
    return run_cache / "conditions" / digest


def _stored_condition_task_metrics(
    run: RunRecord, warnings: list[dict[str, str]]
) -> dict[str, dict[str, float]]:
    artifact = run.artifact("condition_matrix/task_metrics.csv")
    if artifact is None or not artifact.available:
        return {}
    try:
        rows = load_csv_rows(artifact.path, max_rows=200_000)
    except Exception as exc:
        warnings.append(
            _warning("malformed_condition_task_metrics", str(exc), artifact.path)
        )
        return {}
    values: dict[str, dict[str, float]] = {}
    for row in rows:
        condition = row.get("test_condition")
        task = row.get("task")
        try:
            metric = float(row.get("macro_f1", ""))
        except (TypeError, ValueError):
            continue
        if condition and task and math.isfinite(metric) and 0 <= metric <= 1:
            values.setdefault(str(condition), {})[str(task)] = metric
    return values


def _source_signature(run: RunRecord, artifacts: Sequence[ArtifactRecord]) -> str:
    records = [
        {
            "path": item.relative_path,
            "size": item.size,
            "mtime_ns": item.mtime_ns,
            "link_target": item.link_target,
            "target_size": item.target_size,
            "target_mtime_ns": item.target_mtime_ns,
            "available": item.available,
        }
        for item in artifacts
    ]
    metrics = [
        {
            "phase": item.phase,
            "task": item.task,
            "metric": item.metric,
            "value": item.value,
            "n": item.n,
        }
        for item in run.metrics
    ]
    payload = {"run_signature": run.signature, "artifacts": records, "metrics": metrics}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _render_matrices(
    path: Path,
    matrices: Sequence[MatrixData],
    macro_f1: dict[str, float],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise RuntimeError("rendering requires matplotlib and numpy") from exc

    widest = max(len(item.classes) for item in matrices)
    figure, axes = plt.subplots(
        1,
        len(matrices),
        figsize=(max(5.0, widest * 0.75) * len(matrices), max(4.5, widest * 0.65)),
        squeeze=False,
    )
    for axis, matrix in zip(axes[0], matrices):
        counts = np.asarray(matrix.counts, dtype=float)
        totals = counts.sum(axis=1, keepdims=True)
        normalised = np.divide(counts, totals, out=np.zeros_like(counts), where=totals != 0)
        image = axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        labels = matrix.classes
        axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels=labels)
        title = matrix.task.replace("_", " ").title()
        if matrix.task in macro_f1:
            title += f"\nmacro-F1={macro_f1[matrix.task]:.4f}"
        axis.set_title(title)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for row in range(counts.shape[0]):
            for column in range(counts.shape[1]):
                count = counts[row, column]
                count_text = str(int(count)) if count.is_integer() else f"{count:g}"
                colour = "white" if normalised[row, column] > 0.5 else "black"
                axis.text(
                    column,
                    row,
                    f"{count_text}\n{normalised[row, column]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=colour,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Test confusion matrices (counts and row-normalised percentages)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=160, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _selected(run: RunRecord, source_label: str, selections: set[str]) -> bool:
    if not selections:
        return False
    candidates = {
        run.uid,
        run.run_name,
        run.relative_path,
        f"{source_label}:{run.uid}",
        f"{source_label}:{run.run_name}",
        f"{source_label}:{run.relative_path}",
    }
    return not candidates.isdisjoint(selections)


def _derive_run(
    run: RunRecord,
    source_label: str,
    cache_root: Path,
    *,
    should_render: bool,
) -> tuple[dict[str, Any], int]:
    warnings: list[dict[str, str]] = []
    configured_task = _configured_single_task(run)
    matrix_artifacts = run.iter_artifacts("confusion_matrix")
    report_artifacts = run.iter_artifacts("classification_report")
    condition_matrix_artifacts = [
        item
        for item in run.artifacts
        if item.kind.startswith("condition_matrix/confusion_matrix/")
    ]
    condition_report_artifacts = [
        item
        for item in run.artifacts
        if item.kind.startswith("condition_matrix/classification_report/")
    ]
    condition_metric_artifacts = run.iter_artifacts(
        "condition_matrix/task_metrics.csv"
    )
    relevant_artifacts = sorted(
        matrix_artifacts
        + report_artifacts
        + condition_matrix_artifacts
        + condition_report_artifacts
        + condition_metric_artifacts,
        key=lambda item: (item.kind, item.relative_path),
    )
    signature = _source_signature(run, relevant_artifacts)
    run_cache = cache_root / source_label / run.uid
    summary_path = run_cache / "summary.json"
    image_path = run_cache / "confusion_matrices.png"

    stored_mean = _metric(run, "mean_macro_f1")
    stored_scalar = _metric(run, "macro_f1")
    if stored_scalar is None and configured_task is not None:
        stored_scalar = _metric(run, "macro_f1", configured_task)
    has_task_named_matrix = any(
        _task_from_name(Path(item.path).name, "confusion_matrix")
        for item in matrix_artifacts
    )
    if stored_mean is not None:
        is_multitask = True
    elif stored_scalar is not None or run.schema_version == "single_task_legacy":
        is_multitask = False
    else:
        is_multitask = has_task_named_matrix
    run_type = "multitask" if is_multitask else "single_task"

    report_f1: dict[str, float] = {}
    for artifact in report_artifacts:
        task = _artifact_task(
            artifact,
            "classification_report",
            configured_task if run_type == "single_task" else None,
        )
        if not artifact.available:
            warnings.append(
                _warning(
                    "unavailable_report",
                    "Classification report is unavailable",
                    artifact.path,
                )
            )
            continue
        if task is None:
            warnings.append(
                _warning(
                    "unknown_report_task",
                    "Could not identify report task",
                    artifact.path,
                )
            )
            continue
        try:
            report_f1[task] = _report_macro_f1(Path(artifact.path))
        except (OSError, ValueError) as exc:
            warnings.append(_warning("malformed_report", str(exc), artifact.path))

    matrices: list[MatrixData] = []
    for artifact in matrix_artifacts:
        task = _artifact_task(
            artifact,
            "confusion_matrix",
            configured_task if run_type == "single_task" else None,
        )
        if not artifact.available:
            warnings.append(
                _warning(
                    "unavailable_matrix",
                    "Confusion matrix is unavailable",
                    artifact.path,
                )
            )
            continue
        if task is None:
            warnings.append(
                _warning(
                    "unknown_matrix_task",
                    "Could not identify matrix task",
                    artifact.path,
                )
            )
            continue
        try:
            matrices.append(_read_matrix(Path(artifact.path), task))
        except (OSError, ValueError) as exc:
            warnings.append(_warning("malformed_matrix", str(exc), artifact.path))
    matrices.sort(
        key=lambda item: (
            {"genus": 0, "species": 1}.get(item.task, 99),
            item.task,
        )
    )

    if not matrices:
        warnings.append(
            _warning(
                "missing_confusion_matrices",
                "No readable confusion-matrix CSV was found",
            )
        )
    elif run_type == "multitask":
        present = {item.task for item in matrices}
        expected = {"genus", "species"}
        missing = sorted(expected - present)
        if missing:
            warnings.append(
                _warning(
                    "missing_task_matrices",
                    f"Missing confusion matrices for: {', '.join(missing)}",
                )
            )

    per_task_metrics = {
        item.task: item.value
        for item in run.metrics
        if item.task is not None and item.metric == "macro_f1"
    }
    display_task_f1 = dict(report_f1)
    display_task_f1.update(per_task_metrics)
    expected_tasks = {"genus", "species"}
    fallback_mean = None
    if run_type == "multitask" and expected_tasks.issubset(report_f1):
        fallback_mean = sum(report_f1[task] for task in expected_tasks) / len(expected_tasks)
    elif run_type == "multitask" and stored_mean is None and report_f1:
        warnings.append(
            _warning(
                "incomplete_report_mean",
                "Mean macro-F1 was not derived because not all three task reports are readable",
            )
        )
    effective_mean = stored_mean if stored_mean is not None else fallback_mean
    effective_task = stored_scalar
    if run_type == "single_task" and effective_task is None and configured_task:
        effective_task = report_f1.get(configured_task)

    for task, metric_value in per_task_metrics.items():
        report_value = report_f1.get(task)
        if report_value is not None and not math.isclose(metric_value, report_value, abs_tol=1e-12):
            warnings.append(
                _warning(
                    "metric_report_mismatch",
                    "Stored and report macro-F1 differ for "
                    f"{task}: {metric_value} != {report_value}",
                )
            )
    if stored_mean is not None and per_task_metrics:
        recomputed = sum(per_task_metrics.values()) / len(per_task_metrics)
        if not math.isclose(stored_mean, recomputed, abs_tol=1e-12):
            warnings.append(
                _warning(
                    "stored_mean_mismatch",
                    f"Stored mean macro-F1 differs from task mean: {stored_mean} != {recomputed}",
                )
            )

    render_needed = should_render and bool(matrices)
    unchanged = False
    previous: dict[str, Any] | None = None
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            previous = loaded if isinstance(loaded, dict) else None
            unchanged = bool(previous and previous.get("source_signature") == signature)
        except (OSError, json.JSONDecodeError):
            unchanged = False
    rendered = False
    if render_needed and (not unchanged or not image_path.is_file()):
        _render_matrices(image_path, matrices, display_task_f1)
        rendered = True
    image_is_current = bool(
        matrices and image_path.is_file() and (render_needed or unchanged)
    )

    root_condition = run.train_condition or "original"
    condition_summaries: dict[str, dict[str, Any]] = {
        root_condition: {
            "test_condition": root_condition,
            "condition_relation": canonical_condition_relation(
                root_condition, root_condition
            ),
            "metrics": {
                "effective_mean_macro_f1": (
                    effective_mean if run_type == "multitask" else None
                ),
                "mean_macro_f1_source": (
                    "test_metrics"
                    if stored_mean is not None
                    else "classification_reports"
                    if fallback_mean is not None
                    else None
                ),
                "macro_f1_by_task": display_task_f1,
            },
            "confusion_matrices": [
                {
                    "task": item.task,
                    "source_path": item.path,
                    "classes": list(item.classes),
                    "shape": [
                        len(item.counts),
                        len(item.counts[0]) if item.counts else 0,
                    ],
                }
                for item in matrices
            ],
            "combined_confusion_matrix_image": (
                str(image_path.relative_to(cache_root)) if image_is_current else None
            ),
        }
    }
    stored_condition_metrics = _stored_condition_task_metrics(run, warnings)
    reports_by_condition: dict[str, list[ArtifactRecord]] = {}
    matrices_by_condition: dict[str, list[ArtifactRecord]] = {}
    for artifact in condition_report_artifacts:
        condition = _matrix_artifact_condition(artifact, "classification_report")
        if condition:
            reports_by_condition.setdefault(condition, []).append(artifact)
    for artifact in condition_matrix_artifacts:
        condition = _matrix_artifact_condition(artifact, "confusion_matrix")
        if condition:
            matrices_by_condition.setdefault(condition, []).append(artifact)

    nested_conditions = sorted(
        set(reports_by_condition)
        | set(matrices_by_condition)
        | set(stored_condition_metrics)
    )
    rendered_count = int(rendered)
    expected_condition_tasks = tuple(dict.fromkeys(run.tasks)) or (
        "genus",
        "species",
    )
    expected_condition_task_set = set(expected_condition_tasks)
    for condition in nested_conditions:
        # The root scientific test remains the compatibility source for the
        # matched condition even when the matrix evaluator saved a duplicate.
        if condition == root_condition:
            continue
        condition_report_f1: dict[str, float] = {}
        for artifact in reports_by_condition.get(condition, []):
            task = _artifact_task(artifact, "classification_report", None)
            if not artifact.available:
                warnings.append(
                    _warning(
                        "unavailable_condition_report",
                        f"Classification report is unavailable for {condition}",
                        artifact.path,
                    )
                )
                continue
            if task is None:
                warnings.append(
                    _warning(
                        "unknown_condition_report_task",
                        f"Could not identify report task for {condition}",
                        artifact.path,
                    )
                )
                continue
            try:
                condition_report_f1[task] = _report_macro_f1(Path(artifact.path))
            except (OSError, ValueError) as exc:
                warnings.append(
                    _warning("malformed_condition_report", str(exc), artifact.path)
                )

        condition_matrices: list[MatrixData] = []
        for artifact in matrices_by_condition.get(condition, []):
            task = _artifact_task(artifact, "confusion_matrix", None)
            if not artifact.available:
                warnings.append(
                    _warning(
                        "unavailable_condition_matrix",
                        f"Confusion matrix is unavailable for {condition}",
                        artifact.path,
                    )
                )
                continue
            if task is None:
                warnings.append(
                    _warning(
                        "unknown_condition_matrix_task",
                        f"Could not identify matrix task for {condition}",
                        artifact.path,
                    )
                )
                continue
            try:
                condition_matrices.append(_read_matrix(Path(artifact.path), task))
            except (OSError, ValueError) as exc:
                warnings.append(
                    _warning("malformed_condition_matrix", str(exc), artifact.path)
                )
        condition_matrices.sort(
            key=lambda item: (
                {"genus": 0, "species": 1}.get(item.task, 99),
                item.task,
            )
        )

        condition_task_f1 = dict(condition_report_f1)
        condition_task_f1.update(stored_condition_metrics.get(condition, {}))
        complete_tasks = expected_condition_task_set.issubset(condition_task_f1)
        condition_mean = (
            sum(condition_task_f1[task] for task in expected_condition_tasks)
            / len(expected_condition_tasks)
            if expected_condition_tasks and complete_tasks
            else None
        )
        if condition_task_f1 and not complete_tasks:
            warnings.append(
                _warning(
                    "incomplete_condition_mean",
                    f"Mean macro-F1 was not derived for {condition} because "
                    "not all configured tasks are available",
                )
            )

        condition_image = (
            _condition_cache_directory(run_cache, condition)
            / "confusion_matrices.png"
        )
        condition_render_needed = should_render and bool(condition_matrices)
        if condition_render_needed and (
            not unchanged or not condition_image.is_file()
        ):
            _render_matrices(
                condition_image, condition_matrices, condition_task_f1
            )
            rendered_count += 1
        condition_image_is_current = bool(
            condition_matrices
            and condition_image.is_file()
            and (condition_render_needed or unchanged)
        )
        condition_summaries[condition] = {
            "test_condition": condition,
            "condition_relation": canonical_condition_relation(
                root_condition, condition
            ),
            "metrics": {
                "effective_mean_macro_f1": condition_mean,
                "mean_macro_f1_source": (
                    "condition_matrix_task_metrics"
                    if complete_tasks
                    and expected_condition_task_set.issubset(
                        stored_condition_metrics.get(condition, {})
                    )
                    else "classification_reports"
                    if complete_tasks
                    else None
                ),
                "macro_f1_by_task": condition_task_f1,
            },
            "confusion_matrices": [
                {
                    "task": item.task,
                    "source_path": item.path,
                    "classes": list(item.classes),
                    "shape": [
                        len(item.counts),
                        len(item.counts[0]) if item.counts else 0,
                    ],
                }
                for item in condition_matrices
            ],
            "combined_confusion_matrix_image": (
                str(condition_image.relative_to(cache_root))
                if condition_image_is_current
                else None
            ),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_label": source_label,
        "source_signature": signature,
        "run_uid": run.uid,
        "run_name": run.run_name,
        "run_path": run.path,
        "run_relative_path": run.relative_path,
        "run_type": run_type,
        "task": configured_task if run_type == "single_task" else None,
        "metrics": {
            "stored_mean_macro_f1": stored_mean if run_type == "multitask" else None,
            "effective_mean_macro_f1": effective_mean if run_type == "multitask" else None,
            "mean_macro_f1_source": (
                "test_metrics"
                if stored_mean is not None
                else "classification_reports"
                if fallback_mean is not None
                else None
            ) if run_type == "multitask" else None,
            "stored_task_macro_f1": stored_scalar if run_type == "single_task" else None,
            "effective_task_macro_f1": effective_task if run_type == "single_task" else None,
            "task_macro_f1_source": (
                "test_metrics"
                if stored_scalar is not None
                else "classification_report"
                if effective_task is not None
                else None
            ) if run_type == "single_task" else None,
            "macro_f1_by_task": display_task_f1,
        },
        "confusion_matrices": [
            {
                "task": item.task,
                "source_path": item.path,
                "classes": list(item.classes),
                "shape": [len(item.counts), len(item.counts[0]) if item.counts else 0],
            }
            for item in matrices
        ],
        "combined_confusion_matrix_image": (
            str(image_path.relative_to(cache_root)) if image_is_current else None
        ),
        "conditions": condition_summaries,
        "warnings": warnings,
    }
    if previous != summary:
        _atomic_json(summary_path, summary)
    return summary, rendered_count


def derive_results(
    sources: Sequence[tuple[str, str | Path]],
    cache_root: str | Path,
    *,
    render: str = "all",
    selected_runs: Iterable[str] = (),
    max_depth: int = 8,
) -> dict[str, Any]:
    """Derive summaries for one or more labelled result roots."""

    if render not in {"all", "selected", "none"}:
        raise ValueError("render must be one of: all, selected, none")
    selections = set(selected_runs)
    if render == "selected" and not selections:
        raise ValueError("render=selected requires at least one selected run")
    labels = [label for label, _ in sources]
    if not sources:
        raise ValueError("at least one labelled source is required")
    if len(labels) != len(set(labels)):
        raise ValueError("source labels must be unique")
    for label in labels:
        if not SOURCE_LABEL_PATTERN.fullmatch(label):
            raise ValueError(f"invalid source label: {label!r}")

    roots = [Path(path).expanduser().absolute() for _, path in sources]
    cache = validate_cache_root(cache_root, roots)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_runs: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    rendered_count = 0
    matched_selections: set[str] = set()
    for (label, _), root in zip(sources, roots):
        discovery = discover_results_root(root, max_depth=max_depth)
        source_records.append(
            {
                "label": label,
                "results_root": str(root),
                "experiments": len(discovery.experiments),
                "runs": len(discovery.runs),
            }
        )
        for run in discovery.runs:
            selected = _selected(run, label, selections)
            if selected:
                matched_selections.update(
                    value
                    for value in selections
                    if value in {
                        run.uid,
                        run.run_name,
                        run.relative_path,
                        f"{label}:{run.uid}",
                        f"{label}:{run.run_name}",
                        f"{label}:{run.relative_path}",
                    }
                )
            should_render = render == "all" or (render == "selected" and selected)
            summary, rendered = _derive_run(
                run,
                label,
                cache,
                should_render=should_render,
            )
            rendered_count += int(rendered)
            manifest_runs.append(
                {
                    "source_label": label,
                    "run_uid": run.uid,
                    "run_name": run.run_name,
                    "summary": str((Path(label) / run.uid / "summary.json")),
                    "run_type": summary["run_type"],
                    "warnings": len(summary["warnings"]),
                }
            )
    unmatched = sorted(selections - matched_selections) if render == "selected" else []
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.time(),
        "cache_root": str(cache),
        "render_mode": render,
        "sources": source_records,
        "runs": manifest_runs,
        "rendered_images": rendered_count,
        "unmatched_selections": unmatched,
    }
    _atomic_json(cache / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create cache-only confusion-matrix and macro-F1 dashboard artefacts"
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Labelled result root; repeat for SLURM and single-task outputs",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".cache/fish-species-dashboard/derived"),
        help="External derived-artifact cache (must not be inside a result root)",
    )
    parser.add_argument("--render", choices=("all", "selected", "none"), default="all")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run UID/name/path to render when --render=selected; may be repeated",
    )
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print the manifest as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        sources = [parse_source(value) for value in args.source]
        manifest = derive_results(
            sources,
            args.cache,
            render=args.render,
            selected_runs=args.run,
            max_depth=args.max_depth,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            f"Derived {len(manifest['runs'])} run summaries from "
            f"{len(manifest['sources'])} sources; rendered "
            f"{manifest['rendered_images']} images in {manifest['cache_root']}"
        )
        if manifest["unmatched_selections"]:
            print("Unmatched selections: " + ", ".join(manifest["unmatched_selections"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
