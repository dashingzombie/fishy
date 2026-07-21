"""Post-training evaluation across explicit train/test condition pairs.

This workflow is deliberately separate from fixed-RGB cue suppression.  It
never expands training runs: one already selected checkpoint is evaluated on
an explicit, validated list of deterministic test conditions.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .cue_suppression import generate_test_cue_conditions
from .cue_suppression import make_test_condition_loader
from .cue_suppression import _runtime_condition
from ..training.epochs import run_hierarchy_epoch as run_epoch


SCHEMA_VERSION = 1
OUTPUT_DIRECTORY = "condition_matrix_evaluation"
RELATIONS = frozenset({"matched", "rgb_stress", "cross_condition"})


def evaluation_relation(train_condition: str, test_condition: str) -> str:
    """Classify one cell without changing either condition's semantics."""
    train_name = str(train_condition)
    test_name = str(test_condition)
    if train_name == test_name:
        return "matched"
    if train_name == "original":
        return "rgb_stress"
    return "cross_condition"


def resolve_condition_matrix_conditions(cfg: dict) -> list[dict[str, Any]]:
    """Resolve the matrix allow-list against the configured cue catalogue.

    The cue catalogue is reused only as a registry of deterministic transform
    parameters.  Its runtime ``enabled`` flag and optional fixed-RGB allow-list
    do not control this independent evaluator.
    """
    evaluation = cfg.get("evaluation")
    canonical_matrix = (
        evaluation.get("condition_matrix", {}) or {}
        if isinstance(evaluation, dict)
        else {}
    )
    legacy_matrix = cfg.get("condition_matrix_evaluation", {}) or {}
    use_canonical = (
        isinstance(evaluation, dict)
        and "condition_matrix" in evaluation
        and not (
            isinstance(canonical_matrix, dict)
            and not canonical_matrix.get("conditions")
            and isinstance(legacy_matrix, dict)
            and bool(legacy_matrix.get("enabled", False))
        )
    )
    if use_canonical:
        matrix = evaluation.get("condition_matrix", {}) or {}
        if not isinstance(matrix, dict):
            raise TypeError("evaluation.condition_matrix must be a mapping")
        configured = matrix.get("conditions", [])
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "evaluation.condition_matrix.conditions must be a non-empty list"
            )
        registry: dict[str, dict[str, Any]] = {
            "original": {
                "name": "original",
                "feature": "baseline",
                "transform": "original",
                "strength": 0.0,
                "parameters": {},
            }
        }
        sweep = cfg.get("sweep", {}) or {}
        if isinstance(sweep, dict) and sweep.get("conditions"):
            from ..config.normalization import normalize_conditions

            registry.update({
                str(item["name"]): item
                for item in normalize_conditions(sweep["conditions"])
            })
        test_schedule = evaluation.get("test_conditions", {}) or {}
        if isinstance(test_schedule, dict):
            for item in test_schedule.get("conditions", []) or []:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("condition")
                    if name:
                        registry[str(name)] = item
        resolved = []
        for item in configured:
            if isinstance(item, str):
                if item not in registry:
                    raise ValueError(
                        "Unknown evaluation.condition_matrix condition "
                        f"{item!r}"
                    )
                item = registry[item]
            if not isinstance(item, dict):
                raise TypeError(
                    "evaluation.condition_matrix.conditions entries must be "
                    "names or complete condition mappings"
                )
            resolved.append(_runtime_condition(item))
        names = [item["condition"] for item in resolved]
        if len(names) != len(set(names)):
            raise ValueError(
                "evaluation.condition_matrix.conditions contains duplicate names"
            )
        return resolved

    matrix = cfg.get("condition_matrix_evaluation", {}) or {}
    if not isinstance(matrix, dict):
        raise TypeError("condition_matrix_evaluation must be a mapping")
    requested = matrix.get("condition_names")
    path = "condition_matrix_evaluation.condition_names"
    if not isinstance(requested, list) or not requested:
        raise ValueError(f"{path} must be a non-empty list")
    if any(not isinstance(name, str) or not name.strip() for name in requested):
        raise ValueError(f"{path} must contain non-empty condition-name strings")
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise ValueError(f"{path} contains duplicate names: {duplicates}")

    catalogue_cfg = copy.deepcopy(cfg)
    cue = dict(catalogue_cfg.get("test_cue_suppression", {}) or {})
    cue["enabled"] = True
    cue.pop("condition_names", None)
    catalogue_cfg["test_cue_suppression"] = cue
    catalogue = [
        {
            "condition": "original",
            "feature": "baseline",
            "transform": "original",
            "strength": 0.0,
        },
        *generate_test_cue_conditions(catalogue_cfg),
    ]
    by_name = {str(condition["condition"]): condition for condition in catalogue}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(
            f"Unknown {path}: {unknown}; available conditions: {list(by_name)}"
        )
    return [copy.deepcopy(by_name[name]) for name in requested]


def _condition_parameters(condition: dict[str, Any]) -> str:
    return json.dumps(
        {
            key: value
            for key, value in condition.items()
            if key not in {"condition", "feature", "transform", "strength"}
        },
        sort_keys=True,
    )


def _identity_fields(
    *,
    run_name: str,
    model_name: Any,
    training_condition: dict[str, Any],
    test_condition: dict[str, Any],
    reused: bool,
) -> dict[str, Any]:
    train_name = str(training_condition["condition"])
    test_name = str(test_condition["condition"])
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "model": model_name,
        "train_condition": train_name,
        "train_feature": training_condition.get("feature"),
        "train_transform": training_condition.get("transform"),
        "train_strength": training_condition.get("strength"),
        "train_parameters": _condition_parameters(training_condition),
        "test_condition": test_name,
        "test_feature": test_condition.get("feature"),
        "test_transform": test_condition.get("transform"),
        "test_strength": test_condition.get("strength"),
        "test_parameters": _condition_parameters(test_condition),
        "evaluation_relation": evaluation_relation(train_name, test_name),
        "reused_matched_evaluation": bool(reused),
    }


def _write_task_reports(
    root: Path,
    condition_name: str,
    true: dict[str, Any],
    pred: dict[str, Any],
    index_to_label_by_task: dict[str, dict[int, str]],
    *,
    training_condition: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    wandb_logger: Any = None,
) -> None:
    report_dir = root / "classification_reports" / condition_name
    matrix_dir = root / "confusion_matrices" / condition_name
    report_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for task, index_to_label in index_to_label_by_task.items():
        labels = list(range(len(index_to_label)))
        names = [index_to_label[index] for index in labels]
        y_true = np.asarray(true.get(task, []), dtype=int)
        y_pred = np.asarray(pred.get(task, []), dtype=int)
        report_path = report_dir / f"classification_report_{task}.csv"
        matrix_path = matrix_dir / f"confusion_matrix_{task}.csv"
        if not len(y_true):
            pd.DataFrame(
                [{"note": "No labelled test examples for this task."}]
            ).to_csv(report_path, index=False)
            pd.DataFrame().to_csv(matrix_path)
            continue
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        pd.DataFrame(report).transpose().to_csv(report_path)
        pd.DataFrame(matrix, index=names, columns=names).to_csv(matrix_path)
        if wandb_logger is not None:
            wandb_logger.log_classification_report(
                condition=condition_name,
                task=task,
                report=report,
                metrics=metrics,
                train_condition=training_condition,
            )
            wandb_logger.log_confusion_matrix(
                condition=condition_name,
                task=task,
                y_true=y_true,
                y_pred=y_pred,
                class_names=names,
            )


def _task_rows(
    identity: dict[str, Any], metrics: dict[str, Any], tasks: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        rows.append({
            **identity,
            "task": task,
            "n": metrics.get(f"{task}_n"),
            "loss": metrics.get(f"{task}_loss"),
            "accuracy": metrics.get(f"{task}_accuracy"),
            "balanced_accuracy": metrics.get(f"{task}_balanced_accuracy"),
            "macro_f1": metrics.get(f"{task}_macro_f1"),
        })
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluate_condition_matrix(
    *,
    cfg: dict,
    run_name: str,
    out_dir: Path,
    model: Any,
    training_condition: dict[str, Any],
    baseline_metrics: dict[str, Any],
    baseline_true: dict[str, Any],
    baseline_pred: dict[str, Any],
    test_loader_context: dict[str, Any] | None,
    criteria: dict[str, Any],
    target_cols: dict[str, str],
    index_to_label_by_task: dict[str, dict[int, str]],
    device: Any,
    use_amp: bool,
    task_loss_weights: dict[str, float],
    normalize_loss_by_active_tasks: bool,
    hierarchy_cfg: dict,
    child_to_parent_matrix: Any,
    metric_context: dict | None = None,
    wandb_logger: Any = None,
) -> dict[str, Any]:
    """Evaluate one selected checkpoint across the configured test matrix."""
    evaluation = cfg.get("evaluation", {}) or {}
    canonical_matrix = (
        evaluation.get("condition_matrix", {}) or {}
        if isinstance(evaluation, dict)
        else {}
    )
    legacy_matrix = cfg.get("condition_matrix_evaluation", {}) or {}
    matrix_cfg = canonical_matrix
    if (
        not canonical_matrix.get("conditions")
        and isinstance(legacy_matrix, dict)
        and bool(legacy_matrix.get("enabled", False))
    ):
        matrix_cfg = legacy_matrix
    if not bool(matrix_cfg.get("enabled", False)):
        return {"enabled": False, "n_conditions": 0, "n_task_rows": 0}
    if test_loader_context is None:
        raise ValueError(
            "condition_matrix_evaluation requires resolved condition-loader context"
        )

    conditions = resolve_condition_matrix_conditions(cfg)
    training_name = str(training_condition["condition"])
    condition_names = [str(condition["condition"]) for condition in conditions]
    if training_name not in condition_names:
        raise ValueError(
            "condition_matrix_evaluation.condition_names must include the "
            f"resolved training condition {training_name!r}"
        )

    matrix_dir = Path(out_dir) / OUTPUT_DIRECTORY
    matrix_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = matrix_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    condition_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    relation_counts = {relation: 0 for relation in sorted(RELATIONS)}

    for test_condition in conditions:
        test_name = str(test_condition["condition"])
        reused = test_name == training_name
        if reused:
            metrics = baseline_metrics
            true = baseline_true
            pred = baseline_pred
        else:
            loader = make_test_condition_loader(test_loader_context, test_condition)
            metrics, true, pred = run_epoch(
                model=model,
                loader=loader,
                criteria=criteria,
                optimizer=None,
                device=device,
                train=False,
                scaler=None,
                use_amp=use_amp,
                task_loss_weights=task_loss_weights,
                normalize_loss_by_active_tasks=normalize_loss_by_active_tasks,
                hierarchy_cfg=hierarchy_cfg,
                child_to_parent_matrix=child_to_parent_matrix,
                metric_context=metric_context,
            )
        identity = _identity_fields(
            run_name=run_name,
            model_name=(cfg.get("model", {}) or {}).get("name"),
            training_condition=training_condition,
            test_condition=test_condition,
            reused=reused,
        )
        relation = str(identity["evaluation_relation"])
        relation_counts[relation] += 1
        condition_rows.append({**identity, **metrics})
        task_rows.extend(_task_rows(identity, metrics, list(target_cols)))
        if bool(matrix_cfg.get("write_reports", True)):
            _write_task_reports(
                matrix_dir,
                test_name,
                true,
                pred,
                index_to_label_by_task,
                training_condition=training_condition,
                metrics=metrics,
                wandb_logger=wandb_logger,
            )

    condition_path = matrix_dir / "condition_metrics.csv"
    task_path = matrix_dir / "task_metrics.csv"
    pd.DataFrame(condition_rows).to_csv(condition_path, index=False)
    pd.DataFrame(task_rows).to_csv(task_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_name": run_name,
        "model": (cfg.get("model", {}) or {}).get("name"),
        "train_condition": training_name,
        "test_conditions": condition_names,
        "expected_condition_cells": len(conditions),
        "completed_condition_cells": len(condition_rows),
        "expected_task_rows": len(conditions) * len(target_cols),
        "completed_task_rows": len(task_rows),
        "relation_counts": relation_counts,
        "reused_matched_evaluation": True,
        "write_reports": bool(matrix_cfg.get("write_reports", True)),
        "condition_metrics": str(condition_path),
        "task_metrics": str(task_path),
    }
    _atomic_json(manifest_path, manifest)
    if wandb_logger is not None:
        wandb_logger.log_test_metrics_table(condition_rows)
        for row in condition_rows:
            test_name = str(row["test_condition"])
            if training_name == "original" and test_name == "original":
                continue
            wandb_logger.log_test_condition(
                test_name,
                {
                    key: value
                    for key, value in row.items()
                    if key not in {
                        "schema_version", "run_name", "model",
                        "train_condition", "train_feature", "train_transform",
                        "train_strength", "train_parameters", "test_condition",
                        "test_feature", "test_transform", "test_strength",
                        "test_parameters", "evaluation_relation",
                        "reused_matched_evaluation",
                    }
                },
                train_condition=training_name,
                update_summary=False,
            )
    return {
        "enabled": True,
        "n_conditions": len(condition_rows),
        "n_task_rows": len(task_rows),
        "relation_counts": relation_counts,
        "manifest_path": str(manifest_path),
        "condition_metrics_path": str(condition_path),
        "task_metrics_path": str(task_path),
    }


__all__ = [
    "OUTPUT_DIRECTORY",
    "RELATIONS",
    "SCHEMA_VERSION",
    "evaluate_condition_matrix",
    "evaluation_relation",
    "resolve_condition_matrix_conditions",
]
