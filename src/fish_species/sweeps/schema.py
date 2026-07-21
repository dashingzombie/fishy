"""Loading and validation for sequential sweep-pipeline YAML files."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class PipelineConfigError(ValueError):
    """A pipeline definition is incomplete or internally inconsistent."""


PHASE_TYPES = {"cartesian", "conditions", "evaluation_only", "fine_tune"}
DIRECTIONS = {"min", "max"}
OPERATORS = {
    "greater_equal", "less_equal", "greater", "less", "equal",
    "drop_from_phase_baseline_less_equal",
}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineConfigError(f"{path} must be a mapping")
    return value


def _selection(value: Any, path: str) -> None:
    selection = _mapping(value, path)
    if "primary_metric" in selection and not isinstance(selection["primary_metric"], str):
        raise PipelineConfigError(f"{path}.primary_metric must be a string")
    if selection.get("direction", "max") not in DIRECTIONS:
        raise PipelineConfigError(f"{path}.direction must be min or max")
    top_k = selection.get("top_k", 1)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise PipelineConfigError(f"{path}.top_k must be a positive integer")
    tolerance = selection.get("tie_tolerance", 1e-12)
    if not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise PipelineConfigError(f"{path}.tie_tolerance must be non-negative")
    for index, tie in enumerate(selection.get("tie_breakers", []) or []):
        tie = _mapping(tie, f"{path}.tie_breakers[{index}]")
        if not isinstance(tie.get("metric"), str) or tie.get("direction", "max") not in DIRECTIONS:
            raise PipelineConfigError(f"invalid tie breaker at {path}.tie_breakers[{index}]")
    for index, constraint in enumerate(selection.get("constraints", []) or []):
        constraint = _mapping(constraint, f"{path}.constraints[{index}]")
        if not isinstance(constraint.get("metric"), str):
            raise PipelineConfigError(f"{path}.constraints[{index}].metric is required")
        if constraint.get("operator") not in OPERATORS:
            raise PipelineConfigError(f"unsupported constraint operator at {path}.constraints[{index}]")
        if "value" not in constraint and constraint.get("value_from") != "phase_max":
            raise PipelineConfigError(f"{path}.constraints[{index}] needs value or value_from=phase_max")
        if constraint.get("value_from") == "phase_max":
            fraction = constraint.get("fraction", 1.0)
            if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 <= float(fraction) <= 1:
                raise PipelineConfigError(f"{path}.constraints[{index}].fraction must be in [0, 1]")
    baseline = selection.get("baseline_condition")
    if baseline is not None and (not isinstance(baseline, str) or not baseline):
        raise PipelineConfigError(f"{path}.baseline_condition must be a non-empty string")


def validate_pipeline(document: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete pipeline without writing state or artifacts."""
    pipeline = _mapping(document.get("pipeline"), "pipeline")
    for key in ("name", "base_config", "output_root"):
        if not isinstance(pipeline.get(key), str) or not pipeline[key].strip():
            raise PipelineConfigError(f"pipeline.{key} must be a non-empty string")
    execution = _mapping(pipeline.get("execution", {}), "pipeline.execution")
    if execution.get("backend", "slurm") not in {"slurm", "local"}:
        raise PipelineConfigError("pipeline.execution.backend must be slurm or local")
    retries = execution.get("maximum_retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise PipelineConfigError("pipeline.execution.maximum_retries must be non-negative")
    results = _mapping(pipeline.get("results", {}), "pipeline.results")
    if results.get("source", "local") not in {"local", "wandb"}:
        raise PipelineConfigError("pipeline.results.source must be local or wandb")
    if results.get("source", "local") == "wandb":
        for key in ("entity", "project"):
            if not isinstance(results.get(key), str) or not results[key]:
                raise PipelineConfigError(f"pipeline.results.{key} is required for W&B")
    metric_file = Path(str(results.get("metric_file", "metrics/validation_summary.json")))
    if metric_file.is_absolute() or ".." in metric_file.parts:
        raise PipelineConfigError("pipeline.results.metric_file must stay below each run directory")
    _selection(_mapping(pipeline.get("selection", {}), "pipeline.selection"), "pipeline.selection")

    phases = pipeline.get("phases")
    if not isinstance(phases, list) or not phases:
        raise PipelineConfigError("pipeline.phases must be a non-empty list")
    names: list[str] = []
    parents: dict[str, str] = {}
    for index, raw in enumerate(phases):
        phase = _mapping(raw, f"pipeline.phases[{index}]")
        name = phase.get("name")
        if not isinstance(name, str) or not name:
            raise PipelineConfigError(f"pipeline.phases[{index}].name is required")
        if name in names:
            raise PipelineConfigError(f"duplicate phase name: {name}")
        names.append(name)
        phase_type = phase.get("type")
        if phase_type not in PHASE_TYPES:
            raise PipelineConfigError(f"phase {name!r} has unsupported type {phase_type!r}")
        parent = phase.get("parent", "base")
        if not isinstance(parent, str) or not parent:
            raise PipelineConfigError(f"phase {name!r} parent must be a string")
        parents[name] = parent
        for field in ("overrides", "common_overrides"):
            if field in phase:
                _mapping(phase[field], f"phase {name}.{field}")
        if phase_type == "cartesian":
            parameters = _mapping(phase.get("parameters"), f"phase {name}.parameters")
            if not parameters or any(not isinstance(values, list) or not values for values in parameters.values()):
                raise PipelineConfigError(f"phase {name!r} parameters must contain non-empty lists")
        if phase_type in {"conditions", "evaluation_only"}:
            conditions = phase.get("conditions")
            if not isinstance(conditions, list) or not conditions:
                raise PipelineConfigError(f"phase {name!r} requires conditions")
            condition_names: list[str] = []
            for condition_index, condition in enumerate(conditions):
                condition = _mapping(condition, f"phase {name}.conditions[{condition_index}]")
                condition_name = condition.get("name")
                if not isinstance(condition_name, str) or not condition_name:
                    raise PipelineConfigError(f"phase {name!r} condition name is required")
                if condition_name in condition_names:
                    raise PipelineConfigError(f"phase {name!r} has duplicate condition {condition_name!r}")
                condition_names.append(condition_name)
                _mapping(condition.get("overrides", {}), f"phase {name}.{condition_name}.overrides")
        if phase_type == "evaluation_only" and phase.get("retrain", False) is not False:
            raise PipelineConfigError(f"phase {name!r} evaluation_only must not retrain")
        if phase_type == "fine_tune":
            checkpoint = _mapping(phase.get("checkpoint", {}), f"phase {name}.checkpoint")
            if checkpoint.get("source", "parent_best") != "parent_best":
                raise PipelineConfigError(f"phase {name!r} only supports checkpoint.source=parent_best")
            if checkpoint.get("load_optimizer", False):
                raise PipelineConfigError(f"phase {name!r} must use fresh optimizer state")
        if "selection" in phase:
            merged = copy.deepcopy(pipeline.get("selection", {}))
            merged.update(phase["selection"] or {})
            _selection(merged, f"phase {name}.selection")

    # Detect missing parents and cycles independently of declaration order.
    for name, parent in parents.items():
        if parent != "base" and parent not in parents:
            raise PipelineConfigError(f"phase {name!r} references unknown parent {parent!r}")
    for start in names:
        seen: set[str] = set()
        current = start
        while current != "base":
            if current in seen:
                raise PipelineConfigError(f"phase dependency cycle involving {current!r}")
            seen.add(current)
            current = parents[current]
    # Sequential lifecycle requires every parent to have appeared already.
    appeared = {"base"}
    for name in names:
        if parents[name] not in appeared:
            raise PipelineConfigError(
                f"phase {name!r} parent {parents[name]!r} must precede it"
            )
        appeared.add(name)
    return document


def load_pipeline(path: str | Path) -> dict[str, Any]:
    """Load and validate one pipeline YAML document."""
    pipeline_path = Path(path).resolve()
    try:
        document = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineConfigError(f"pipeline file not found: {pipeline_path}") from exc
    if not isinstance(document, dict):
        raise PipelineConfigError("pipeline file must contain a mapping")
    validate_pipeline(document)
    document = copy.deepcopy(document)
    document["_pipeline_file"] = str(pipeline_path)
    return document


__all__ = ["PipelineConfigError", "load_pipeline", "validate_pipeline"]
