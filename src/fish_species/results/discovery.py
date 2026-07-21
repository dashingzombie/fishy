"""Bounded, read-only discovery and parsing of heterogeneous result trees.

The scanner follows directory entries but never directory symlinks. Checkpoint
files are recorded as artifacts and are never opened.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .readers import (
    CHECKPOINT_SUFFIXES,
    load_csv_rows,
    load_json,
    load_text,
)
from .normalization import canonical_condition_relation
from .normalization import experiment_type as resolved_experiment_type
from .normalization import hyperparameter_facets
from .normalization import training_condition_identity
from .schemas import (
    ArtifactRecord,
    DiscoveryResult,
    ExperimentRecord,
    MetricRecord,
    RunRecord,
    RunStatus,
    WarningRecord,
    ExperimentSnapshot,
    SweepPlanEntry,
)
from .status import infer_filesystem_state

DEFAULT_MAX_DEPTH = 8
POSSIBLY_ACTIVE_SECONDS = 6 * 60 * 60

@dataclass(frozen=True)
class DiscoveryLimits:
    max_depth: int = DEFAULT_MAX_DEPTH
    active_window_seconds: float = POSSIBLY_ACTIVE_SECONDS


SKIP_DIRECTORIES = {
    ".cache",
    ".git",
    "__pycache__",
    "generated_slurm",
    "run_specs",
    "wandb",
}
SHALLOW_DIRECTORIES = {"logs", "slurm_logs"}
SUPPORT_RUN_MARKERS = {
    "config.json",
    "label_to_index.json",
    "label_to_index_by_task.json",
    "split_summary.json",
}
EXPERIMENT_ARTIFACT_NAMES = {
    "colour_robustness_summary.csv",
    "condition_manifest.json",
    "dual_cue_experiment_plan.json",
    "failed_runs.csv",
    "matched_condition_macro_f1_long.csv",
    "matched_condition_results.csv",
    "matched_vs_rgb_stress_test.csv",
    "condition_matrix_evaluations.csv",
    "condition_matrix_task_metrics.csv",
    "condition_matrix_collection_summary.json",
    "rgb_model_cue_suppression_macro_f1_ratios.csv",
    "rgb_model_cue_suppression_test_metrics.csv",
    "rgb_model_cue_suppression_transform_summary.csv",
    "sweep_manifest.json",
    "sweep_plan.tsv",
}
RUN_EXACT_ARTIFACT_NAMES = {
    "best_model.pt",
    "config.json",
    "history.csv",
    "label_to_index.json",
    "label_to_index_by_task.json",
    "multi_run_results.csv",
    "run_overrides.args",
    "run_status.txt",
    "run_summary.json",
    "split_summary.json",
    "test_metrics.json",
}
RUN_ARTIFACT_PATTERNS = (
    re.compile(r"classification_report(?:_.+)?\.csv$"),
    re.compile(r"confusion_matrix(?:_.+)?\.csv$"),
)
ARRAY_RUN_PATTERN = re.compile(r"run_\d+$")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _stable_uid(kind: str, root: Path, path: Path) -> str:
    lexical = _safe_relative(path.absolute(), root.absolute())
    digest = hashlib.sha256(f"{kind}\0{root.absolute()}\0{lexical}".encode()).hexdigest()
    return digest[:24]


def _file_available(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except OSError:
        return False


def artifact_record(path: Path, root: Path, kind: str | None = None) -> ArtifactRecord:
    """Return metadata for a file without opening it or resolving its identity."""

    stat = path.lstat()
    is_symlink = path.is_symlink()
    link_target = os.readlink(path) if is_symlink else None
    available = _file_available(path)
    target_size: int | None = None
    target_mtime_ns: int | None = None
    if is_symlink and available:
        try:
            target_stat = path.stat()
            target_size = target_stat.st_size
            target_mtime_ns = target_stat.st_mtime_ns
        except OSError:
            available = False
    suffix = path.suffix.lower()
    inferred_kind = kind or (
        "checkpoint" if suffix in CHECKPOINT_SUFFIXES else path.name
    )
    return ArtifactRecord(
        kind=inferred_kind,
        path=str(path.absolute()),
        relative_path=_safe_relative(path.absolute(), root.absolute()),
        available=available,
        is_symlink=is_symlink,
        link_target=link_target,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        target_size=target_size,
        target_mtime_ns=target_mtime_ns,
    )


def _scan_directories(root: Path, max_depth: int) -> dict[Path, dict[str, Path]]:
    """Inventory filenames without following directory symlinks."""

    inventory: dict[Path, dict[str, Path]] = {}
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        entries: dict[str, Path] = {}
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    path = Path(entry.path)
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in SHALLOW_DIRECTORIES:
                                # Enumerate direct log filenames but never recurse through
                                # a potentially large log hierarchy.
                                stack.append((path, max_depth))
                            elif depth < max_depth and entry.name not in SKIP_DIRECTORIES:
                                stack.append((path, depth + 1))
                        elif entry.is_file(follow_symlinks=False) or entry.is_symlink():
                            entries[entry.name] = path
                    except OSError:
                        entries[entry.name] = path
        except OSError:
            continue
        inventory[directory] = entries
    return inventory


def _is_run_directory(files: dict[str, Path]) -> bool:
    names = set(files)
    if names & {"test_metrics.json", "run_summary.json"}:
        return True
    return "history.csv" in names and bool(names & SUPPORT_RUN_MARKERS)


def _nearest_array_directory(path: Path, root: Path) -> Path | None:
    cursor = path
    while cursor != root and _is_relative_to(cursor, root):
        if ARRAY_RUN_PATTERN.fullmatch(cursor.name):
            return cursor
        cursor = cursor.parent
    return None


def _experiment_directory(run_dir: Path, root: Path) -> Path:
    relative = run_dir.relative_to(root)
    return root if len(relative.parts) == 1 else root / relative.parts[0]


def _warned_json(
    path: Path | None,
    warnings: list[WarningRecord],
    label: str,
) -> dict[str, Any]:
    if path is None:
        return {}
    if not _file_available(path):
        warnings.append(WarningRecord("unavailable_artifact", f"{label} is unavailable", str(path)))
        return {}
    try:
        return load_json(path)
    except Exception as exc:
        warnings.append(WarningRecord("malformed_json", f"Could not read {label}: {exc}", str(path)))
        return {}


def _warned_text(
    path: Path | None,
    warnings: list[WarningRecord],
    label: str,
) -> str | None:
    if path is None:
        return None
    if not _file_available(path):
        warnings.append(WarningRecord("unavailable_artifact", f"{label} is unavailable", str(path)))
        return None
    try:
        return load_text(path).strip()
    except Exception as exc:
        warnings.append(WarningRecord("malformed_text", f"Could not read {label}: {exc}", str(path)))
        return None


def _nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _source_identity(
    root: Path,
    source_kind: str | None,
    source_label: str | None,
) -> tuple[str, str]:
    """Return an explicit result provenance without inspecting run contents."""

    if source_kind:
        kind = str(source_kind)
    elif root.name == "outputs_slurm" or "outputs_slurm" in root.parts:
        kind = "slurm"
    elif "single_task" in root.parts:
        kind = "single_task"
    else:
        kind = "local"
    return kind, str(source_label or kind)


def _normalise_task_name(value: Any) -> str:
    text = str(value)
    return {
        "species_label": "species",
    }.get(text, text)


def _hyperparameter_facets(config: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper retained for callers of the private old helper."""

    return hyperparameter_facets(config)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _as_int(value: Any) -> int | None:
    result = _as_float(value)
    return int(result) if result is not None and result.is_integer() else None


def _task_metrics(
    metrics: dict[str, Any],
    source: Path | None,
    *,
    condition: str,
    condition_relation: str,
) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    source_text = str(source) if source else None
    counts: dict[str, int] = {}
    for key, value in metrics.items():
        if key.endswith("_n"):
            count = _as_int(value)
            if count is not None:
                counts[key[:-2]] = count
    for key, value in metrics.items():
        numeric = _as_float(value)
        if numeric is None or key.endswith("_n"):
            continue
        task: str | None = None
        metric = key
        for prefix in ("genus", "species"):
            if key.startswith(prefix + "_"):
                task = prefix
                metric = key[len(prefix) + 1 :]
                break
        records.append(
            MetricRecord(
                phase="test",
                task=task,
                metric=metric,
                value=numeric,
                n=counts.get(task) if task else None,
                source=source_text,
                condition=condition,
                condition_relation=condition_relation,
                canonical_key=(
                    f"test/{condition}/{task + '_' if task else ''}{metric}"
                ),
            )
        )
    return records


def _history_summary(
    path: Path | None,
    config: dict[str, Any],
    warnings: list[WarningRecord],
) -> tuple[int | None, int | None, float | None, str | None]:
    """Read bounded epoch metadata for dashboards, without altering run files."""

    if path is None or not _file_available(path):
        return None, None, None, None
    try:
        rows = load_csv_rows(path, max_rows=10_000)
    except Exception as exc:
        warnings.append(WarningRecord("malformed_history", f"Could not read history: {exc}", str(path)))
        return None, None, None, None
    epochs = [_as_int(row.get("epoch")) for row in rows]
    epochs_ran = max((epoch for epoch in epochs if epoch is not None), default=None)
    configured_selection = _nested(
        config, "multi_task", "selection_metric", default=None
    )
    candidates = [
        f"val_{configured_selection}" if configured_selection else None,
        "val_mean_macro_f1",
        "val_macro_f1",
    ]
    for column in candidates:
        if not column:
            continue
        scored = [
            (_as_float(row.get(column)), _as_int(row.get("epoch")))
            for row in rows
        ]
        valid = [(score, epoch) for score, epoch in scored if score is not None]
        if valid:
            best_score, best_epoch = max(valid, key=lambda item: item[0])
            return epochs_ran, best_epoch, best_score, column.removeprefix("val_")
    return epochs_ran, None, None, None


def _artifact_signature(artifacts: Iterable[ArtifactRecord]) -> str:
    rows = [
        (
            item.relative_path,
            item.size,
            item.mtime_ns,
            item.link_target,
            item.target_size,
            item.target_mtime_ns,
            item.available,
        )
        for item in artifacts
    ]
    return hashlib.sha256(json.dumps(sorted(rows), separators=(",", ":")).encode()).hexdigest()


def _classify_run_artifact(name: str) -> str | None:
    if name in RUN_EXACT_ARTIFACT_NAMES:
        return "checkpoint" if Path(name).suffix.lower() in CHECKPOINT_SUFFIXES else name
    for pattern in RUN_ARTIFACT_PATTERNS:
        if pattern.fullmatch(name):
            return "classification_report" if name.startswith("classification") else "confusion_matrix"
    return None


def _collect_run_artifacts(
    run_dir: Path,
    array_dir: Path | None,
    root: Path,
    inventory: dict[Path, dict[str, Path]],
) -> list[ArtifactRecord]:
    artifacts: list[ArtifactRecord] = []
    seen: set[Path] = set()

    def add(directory: Path, include_parent: bool = False) -> None:
        for name, path in inventory.get(directory, {}).items():
            kind = _classify_run_artifact(name)
            if include_parent and name not in {"run_overrides.args", "run_status.txt", "multi_run_results.csv"} and path.suffix.lower() not in {".out", ".err"}:
                continue
            if path.suffix.lower() in {".out", ".err"}:
                kind = "slurm_log"
            if kind and path not in seen:
                try:
                    artifacts.append(artifact_record(path, root, kind))
                    seen.add(path)
                except OSError:
                    continue

    add(run_dir)
    if array_dir and array_dir != run_dir:
        add(array_dir, include_parent=True)
    cue_dir = run_dir / "cue_suppression"
    for name, path in inventory.get(cue_dir, {}).items():
        if name in {
            "cue_suppression_config.json",
            "macro_f1_ratios.csv",
            "test_condition_metrics.csv",
            "transform_summary.csv",
        }:
            try:
                artifacts.append(artifact_record(path, root, f"cue_suppression/{name}"))
            except OSError:
                continue
    matrix_dir = run_dir / "condition_matrix_evaluation"
    for name, path in inventory.get(matrix_dir, {}).items():
        if name in {"manifest.json", "condition_metrics.csv", "task_metrics.csv"}:
            try:
                artifacts.append(
                    artifact_record(path, root, f"condition_matrix/{name}")
                )
            except OSError:
                continue
    for directory, files in inventory.items():
        try:
            relative = directory.relative_to(matrix_dir)
        except ValueError:
            continue
        if len(relative.parts) != 2:
            continue
        category, condition = relative.parts
        if category not in {"classification_reports", "confusion_matrices"}:
            continue
        for name, path in files.items():
            if (
                category == "classification_reports"
                and not name.startswith("classification_report_")
            ) or (
                category == "confusion_matrices"
                and not name.startswith("confusion_matrix_")
            ):
                continue
            if not name.endswith(".csv"):
                continue
            kind = (
                "condition_matrix/classification_report"
                if category == "classification_reports"
                else "condition_matrix/confusion_matrix"
            )
            try:
                artifacts.append(
                    artifact_record(path, root, f"{kind}/{condition}/{name}")
                )
            except OSError:
                continue
    return sorted(artifacts, key=lambda item: item.relative_path)


def _status_for_run(
    files: dict[str, Path],
    array_files: dict[str, Path],
    failed_status: str | None,
    updated_at: float,
    warnings: list[WarningRecord],
    *,
    now: float,
    active_window_seconds: float,
) -> tuple[RunStatus, str, str | None, bool]:
    status_path = array_files.get("run_status.txt") or files.get("run_status.txt")
    status_text = _warned_text(status_path, warnings, "run status")
    terminal_output = any(
        _file_available(files[name])
        for name in ("test_metrics.json", "run_summary.json")
        if name in files
    )
    status, evidence = infer_filesystem_state(
        raw_exit_status=status_text,
        failed_table_status=failed_status,
        terminal_metrics_present=terminal_output,
        updated_at=updated_at,
        now=now,
        active_window_seconds=active_window_seconds,
    )
    return status, evidence, status_text, terminal_output


def _failed_statuses(experiment: Path, warnings: list[WarningRecord]) -> dict[str, str]:
    path = experiment / "failed_runs.csv"
    if not _file_available(path):
        return {}
    try:
        rows = load_csv_rows(path, max_rows=100_000)
    except Exception as exc:
        warnings.append(WarningRecord("malformed_csv", f"Could not read failed runs: {exc}", str(path)))
        return {}
    result: dict[str, str] = {}
    for row in rows:
        name = row.get("array_run") or row.get("run_name")
        if name:
            result[name] = row.get("status", "unknown")
    return result


def _build_run(
    run_dir: Path,
    root: Path,
    inventory: dict[Path, dict[str, Path]],
    experiment_uid: str,
    experiment_name: str,
    failed_statuses: dict[str, str],
    source_kind: str,
    source_label: str,
    *,
    now: float,
    active_window_seconds: float,
) -> RunRecord:
    files = inventory[run_dir]
    array_dir = _nearest_array_directory(run_dir, root)
    array_files = inventory.get(array_dir, {}) if array_dir else {}
    warnings: list[WarningRecord] = []
    artifacts = _collect_run_artifacts(run_dir, array_dir, root, inventory)
    available_mtimes = [
        (item.target_mtime_ns or item.mtime_ns) / 1_000_000_000 for item in artifacts
    ]
    updated_at = max(available_mtimes, default=run_dir.stat().st_mtime)
    array_run = array_dir.name if array_dir else None
    status, status_evidence, raw_exit_status, terminal_metrics_present = _status_for_run(
        files,
        array_files,
        failed_statuses.get(array_run or ""),
        updated_at,
        warnings,
        now=now,
        active_window_seconds=active_window_seconds,
    )

    config = _warned_json(files.get("config.json"), warnings, "run config")
    summary = _warned_json(files.get("run_summary.json"), warnings, "run summary")
    test_metrics = _warned_json(files.get("test_metrics.json"), warnings, "test metrics")
    label_map_path = files.get("label_to_index_by_task.json") or files.get("label_to_index.json")
    label_map = _warned_json(
        label_map_path,
        warnings,
        "label map",
    )
    overrides = _warned_text(
        array_files.get("run_overrides.args") or files.get("run_overrides.args"),
        warnings,
        "run overrides",
    )
    epochs_ran, history_best_epoch, history_best_score, history_selection = _history_summary(
        files.get("history.csv"), config, warnings
    )

    model = summary.get("model") or _nested(config, "model", "name")
    configured_tasks = _nested(config, "data", "target_cols", default=None)
    if isinstance(configured_tasks, dict):
        tasks = [str(key) for key in configured_tasks]
    elif isinstance(configured_tasks, list):
        tasks = [_normalise_task_name(item) for item in configured_tasks]
    else:
        legacy_tasks = _nested(config, "multi_task", "target_cols", default=[])
        tasks = (
            [_normalise_task_name(item) for item in legacy_tasks]
            if isinstance(legacy_tasks, list)
            else []
        )
    if not tasks and _nested(config, "data", "target_col") is not None:
        tasks = [_normalise_task_name(_nested(config, "data", "target_col"))]
    if not tasks and label_map_path and label_map_path.name == "label_to_index_by_task.json":
        tasks = sorted(str(key) for key in label_map)

    input_condition = _nested(config, "input_condition", default={})
    if not isinstance(input_condition, dict):
        input_condition = {}
    condition_identity = training_condition_identity(config, summary)
    condition_enabled = bool(input_condition.get("enabled"))
    train_condition = (
        summary.get("train_condition")
        or input_condition.get("name")
        or input_condition.get("condition")
    )
    train_feature = summary.get("train_feature") or input_condition.get("feature")
    train_transform = summary.get("train_transform") or input_condition.get("transform")
    train_strength = _as_float(summary.get("train_strength", input_condition.get("strength")))
    training_mode = (
        "matched_condition" if condition_enabled or train_condition else
        "baseline"
    )
    fixed_rgb_stress = bool(
        summary.get("cue_suppression_enabled")
        or _nested(
            config,
            "evaluation",
            "test_conditions",
            "enabled",
            default=False,
        )
        or _nested(config, "test_cue_suppression", "enabled", default=False)
    )
    root_test_condition = str(train_condition or condition_identity["name"] or "original")
    root_condition_relation = canonical_condition_relation(
        root_test_condition, root_test_condition
    )

    if test_metrics and "mean_macro_f1" in test_metrics:
        schema_version = "multitask_hierarchy" if "hierarchy_loss" in test_metrics else "multitask_legacy"
    elif test_metrics and "macro_f1" in test_metrics:
        schema_version = "single_task_legacy"
    elif summary:
        schema_version = "run_summary_only"
    else:
        schema_version = "partial"
    metric_source = files.get("test_metrics.json")
    metrics = _task_metrics(
        test_metrics,
        metric_source,
        condition=root_test_condition,
        condition_relation=root_condition_relation,
    )
    if schema_version == "single_task_legacy" and len(tasks) == 1:
        for metric_record in metrics:
            metric_record.task = tasks[0]
            metric_record.canonical_key = (
                f"test/{root_test_condition}/{tasks[0]}_{metric_record.metric}"
            )
    if not metrics and summary:
        metrics = _task_metrics(
            {key[5:]: value for key, value in summary.items() if key.startswith("test_")},
            files.get("run_summary.json"),
            condition=root_test_condition,
            condition_relation=root_condition_relation,
        )

    mean_macro_f1 = _as_float(test_metrics.get("mean_macro_f1"))
    task_macro_f1 = _as_float(test_metrics.get("macro_f1"))
    effective_macro_f1 = mean_macro_f1 if mean_macro_f1 is not None else task_macro_f1
    effective_macro_f1_label = (
        "mean macro-F1 across tasks"
        if mean_macro_f1 is not None
        else "single-task macro-F1"
        if task_macro_f1 is not None
        else None
    )

    run_name = str(summary.get("run_name") or run_dir.name)
    return RunRecord(
        uid=_stable_uid("run", root, run_dir),
        experiment_uid=experiment_uid,
        experiment_name=experiment_name,
        array_run=array_run,
        run_name=run_name,
        path=str(run_dir.absolute()),
        relative_path=_safe_relative(run_dir.absolute(), root.absolute()),
        source_kind=source_kind,
        source_label=source_label,
        source_root=str(root.absolute()),
        status=status,
        status_evidence=status_evidence,
        updated_at=updated_at,
        model=str(model) if model is not None else None,
        tasks=tasks,
        training_mode=training_mode,
        train_condition=str(train_condition) if train_condition is not None else None,
        train_feature=str(train_feature) if train_feature is not None else None,
        train_transform=str(train_transform) if train_transform is not None else None,
        train_strength=train_strength,
        fixed_rgb_stress_evaluation=fixed_rgb_stress,
        best_epoch=_as_int(summary.get("best_epoch")) or history_best_epoch,
        best_val_score=(
            _as_float(summary.get("best_val_score"))
            if _as_float(summary.get("best_val_score")) is not None
            else history_best_score
        ),
        selection_metric=(
            str(summary["selection_metric"])
            if summary.get("selection_metric") is not None
            else history_selection
        ),
        config=config,
        overrides=overrides,
        metrics=metrics,
        artifacts=artifacts,
        warnings=warnings,
        schema_version=schema_version,
        signature=_artifact_signature(artifacts),
        raw_exit_status=raw_exit_status,
        terminal_metrics_present=terminal_metrics_present,
        hyperparameters=hyperparameter_facets(config, summary),
        effective_macro_f1=effective_macro_f1,
        effective_macro_f1_label=effective_macro_f1_label,
        epochs_ran=epochs_ran,
        experiment_type=resolved_experiment_type(config),
        image_size=_as_int(
            _nested(
                config,
                "preprocessing",
                "image_size",
                default=_nested(config, "data", "image_size"),
            )
        ),
        train_condition_parameters=condition_identity["parameters"],
    )


def _sweep_plan_entries(
    path: Path,
    warnings: list[WarningRecord],
) -> list[SweepPlanEntry]:
    if not _file_available(path):
        return []
    try:
        rows = load_csv_rows(path, max_rows=1_000_000)
    except Exception as exc:
        warnings.append(WarningRecord("malformed_sweep_plan", f"Could not read sweep plan: {exc}", str(path)))
        return []
    entries: list[SweepPlanEntry] = []
    for row in rows:
        run_index = _as_int(row.get("run_index"))
        entries.append(
            SweepPlanEntry(
                run_index=run_index,
                array_name=row.get("array_name") or None,
                values={str(key): str(value) for key, value in row.items()},
            )
        )
    return entries


def _discover(
    results_root: str | Path,
    *,
    limits: DiscoveryLimits,
    now: float,
    single_experiment: bool,
    source_kind: str | None,
    source_label: str | None,
) -> DiscoveryResult:
    """Discover result records without writing to or below ``results_root``."""

    root = Path(results_root).expanduser().absolute()
    if not root.is_dir():
        raise ValueError(f"results root is not a directory: {root}")
    if limits.max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    resolved_source_kind, resolved_source_label = _source_identity(
        root, source_kind, source_label
    )

    inventory = _scan_directories(root, limits.max_depth)
    run_dirs = sorted(path for path, files in inventory.items() if _is_run_directory(files))
    represented_array_dirs = {
        array_dir
        for run_dir in run_dirs
        if (array_dir := _nearest_array_directory(run_dir, root)) is not None
    }
    # Interrupted array jobs can have only their wrapper markers and no nested
    # scientific run directory yet. Index those wrappers without duplicating
    # wrappers that already contain a discovered run.
    for path, files in inventory.items():
        if (
            ARRAY_RUN_PATTERN.fullmatch(path.name)
            and path not in represented_array_dirs
            and set(files) & {"run_overrides.args", "run_status.txt"}
        ):
            run_dirs.append(path)
    run_dirs.sort()
    if single_experiment:
        experiment_paths = [root]
    else:
        experiment_paths = sorted({_experiment_directory(path, root) for path in run_dirs})
        # Include sweep/aggregate-only experiments that do not yet contain a completed run.
        for child in sorted(path for path in inventory if path.parent == root and path != root):
            names = set(inventory[child])
            if names & EXPERIMENT_ARTIFACT_NAMES and child not in experiment_paths:
                experiment_paths.append(child)
        experiment_paths.sort()

    experiments: list[ExperimentRecord] = []
    experiment_by_path: dict[Path, ExperimentRecord] = {}
    failed_by_experiment: dict[Path, dict[str, str]] = {}
    for path in experiment_paths:
        warnings: list[WarningRecord] = []
        artifacts: list[ArtifactRecord] = []
        for name, artifact_path in inventory.get(path, {}).items():
            if name in EXPERIMENT_ARTIFACT_NAMES:
                try:
                    artifacts.append(artifact_record(artifact_path, root, name))
                except OSError as exc:
                    warnings.append(WarningRecord("unreadable_artifact", str(exc), str(artifact_path)))
        for log_directory_name in SHALLOW_DIRECTORIES:
            log_directory = path / log_directory_name
            for artifact_path in inventory.get(log_directory, {}).values():
                if artifact_path.suffix.lower() not in {".out", ".err", ".log", ".txt"}:
                    continue
                try:
                    artifacts.append(artifact_record(artifact_path, root, "slurm_log"))
                except OSError as exc:
                    warnings.append(WarningRecord("unreadable_log", str(exc), str(artifact_path)))
        artifacts.sort(key=lambda item: item.relative_path)
        plan_entries = _sweep_plan_entries(path / "sweep_plan.tsv", warnings)
        expected_count = len(plan_entries) if _file_available(path / "sweep_plan.tsv") else None
        updated_at = max(
            ((item.target_mtime_ns or item.mtime_ns) / 1_000_000_000 for item in artifacts),
            default=path.stat().st_mtime,
        )
        record = ExperimentRecord(
            uid=_stable_uid("experiment", root, path),
            name=path.name if path != root else root.name,
            path=str(path.absolute()),
            relative_path=_safe_relative(path.absolute(), root.absolute()),
            source_kind=resolved_source_kind,
            source_label=resolved_source_label,
            source_root=str(root.absolute()),
            updated_at=updated_at,
            expected_run_count=expected_count,
            artifacts=artifacts,
            warnings=warnings,
            signature=_artifact_signature(artifacts),
            plan_entries=plan_entries,
        )
        experiments.append(record)
        experiment_by_path[path] = record
        failed_by_experiment[path] = _failed_statuses(path, warnings)

    runs: list[RunRecord] = []
    root_warnings: list[WarningRecord] = []
    for run_dir in run_dirs:
        experiment_path = root if single_experiment else _experiment_directory(run_dir, root)
        experiment = experiment_by_path.get(experiment_path)
        if experiment is None:
            root_warnings.append(
                WarningRecord("missing_experiment", "Could not assign run to an experiment", str(run_dir))
            )
            continue
        try:
            runs.append(
                _build_run(
                    run_dir,
                    root,
                    inventory,
                    experiment.uid,
                    experiment.name,
                    failed_by_experiment.get(experiment_path, {}),
                    resolved_source_kind,
                    resolved_source_label,
                    now=now,
                    active_window_seconds=limits.active_window_seconds,
                )
            )
        except Exception as exc:
            root_warnings.append(
                WarningRecord("run_parse_error", f"Could not index run: {exc}", str(run_dir))
            )

    names: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        names[(run.experiment_uid, run.run_name)].append(run)
    for (_, duplicate_name), duplicate_runs in names.items():
        if len(duplicate_runs) > 1:
            for run in duplicate_runs:
                run.warnings.append(
                    WarningRecord(
                        "duplicate_run_name",
                        f"Run name occurs at {len(duplicate_runs)} paths: {duplicate_name}",
                        run.path,
                    )
                )

    return DiscoveryResult(
        results_root=str(root),
        experiments=experiments,
        runs=runs,
        warnings=root_warnings,
    )


def discover_results_root(
    results_root: str | Path,
    *,
    limits: DiscoveryLimits | None = None,
    max_depth: int | None = None,
    now: float | None = None,
    source_kind: str | None = None,
    source_label: str | None = None,
) -> DiscoveryResult:
    """Discover experiments beneath a parent results directory."""

    selected_limits = limits or DiscoveryLimits()
    if max_depth is not None:
        selected_limits = DiscoveryLimits(
            max_depth=max_depth,
            active_window_seconds=selected_limits.active_window_seconds,
        )
    return _discover(
        results_root,
        limits=selected_limits,
        now=time.time() if now is None else now,
        single_experiment=False,
        source_kind=source_kind,
        source_label=source_label,
    )


def discover_experiment(
    experiment_root: str | Path,
    *,
    limits: DiscoveryLimits | None = None,
    max_depth: int | None = None,
    now: float | None = None,
    source_kind: str | None = None,
    source_label: str | None = None,
) -> ExperimentSnapshot:
    """Discover exactly one experiment rooted at ``experiment_root``."""

    selected_limits = limits or DiscoveryLimits()
    if max_depth is not None:
        selected_limits = DiscoveryLimits(
            max_depth=max_depth,
            active_window_seconds=selected_limits.active_window_seconds,
        )
    result = _discover(
        experiment_root,
        limits=selected_limits,
        now=time.time() if now is None else now,
        single_experiment=True,
        source_kind=source_kind,
        source_label=source_label,
    )
    return ExperimentSnapshot(
        experiment=result.experiments[0],
        runs=result.runs,
        warnings=result.warnings,
    )


def artifact_as_dict(artifact: ArtifactRecord) -> dict[str, Any]:
    """Public serialization helper for UI and tests."""

    return asdict(artifact)
