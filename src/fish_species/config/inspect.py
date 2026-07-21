from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .loading import load_config
from .normalization import normalize_config_with_report
from .overrides import apply_overrides
from .validation import (
    ConfigValidationError,
    resolve_workflow,
    validate_config,
    validate_override_items,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a resolved experiment configuration without running it."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Apply existing dotted key=value overrides before inspection.",
    )
    parser.add_argument(
        "--workflow",
        choices=("auto", "training", "run_specs", "saved"),
        default="auto",
    )
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--format", choices=("yaml", "json"), default="yaml")
    return parser


def identify_experiment_type(config: dict[str, Any]) -> str:
    input_condition = config.get("input_condition", {}) or {}
    if isinstance(input_condition, dict) and bool(input_condition.get("enabled", False)):
        transform = str(input_condition.get("transform", "original")).lower()
        return "matched_condition_execution" if transform != "original" else "rgb_training_execution"
    matched = config.get("matched_condition_training", {}) or {}
    if isinstance(matched, dict) and bool(matched.get("enabled", False)):
        return "matched_condition_plan"
    cue = config.get("test_cue_suppression", {}) or {}
    if isinstance(cue, dict) and bool(cue.get("enabled", False)):
        return "fixed_rgb_stress_evaluation"
    return "ordinary_training"


def _fixed_rgb_test_conditions(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    from ..evaluation.cue_suppression import generate_test_cue_conditions

    cue = config.get("test_cue_suppression", {}) or {}
    if not isinstance(cue, dict):
        return [], []
    catalogue_config = dict(config)
    catalogue_cue = dict(cue)
    catalogue_cue["enabled"] = True
    catalogue_config["test_cue_suppression"] = catalogue_cue
    configured = [
        condition["condition"]
        for condition in generate_test_cue_conditions(catalogue_config)
    ]
    effective = configured if bool(cue.get("enabled", False)) else []
    return configured, effective


def _slurm_inspection(config: dict[str, Any]) -> dict[str, Any] | None:
    slurm = config.get("slurm")
    if not isinstance(slurm, dict):
        return None
    resource_keys = (
        "account",
        "partition",
        "nodes",
        "ntasks",
        "cpus_per_task",
        "memory",
        "time_limit",
        "gpus_per_task",
    )
    resources = {key: slurm[key] for key in resource_keys if key in slurm}
    array = slurm.get("array", {}) or {}
    if isinstance(array, dict) and "max_active" in array:
        resources["array_max_active"] = array["max_active"]
    return {
        "enabled": slurm.get("enabled"),
        "cluster_profile": slurm.get("cluster_profile"),
        "resources": resources,
        "paths": dict(slurm.get("paths", {}) or {}),
    }


def inspection_summary(config: dict[str, Any], workflow: str) -> dict[str, Any]:
    # Lazy imports keep config loading and --help lightweight.
    from ..experiments.conditions import generate_conditions, sweep_combinations
    combinations = sweep_combinations(config)
    matched = config.get("matched_condition_training", {}) or {}
    input_condition = config.get("input_condition", {}) or {}
    input_condition_enabled = isinstance(input_condition, dict) and bool(
        input_condition.get("enabled", False)
    )
    matched_plan_enabled = isinstance(matched, dict) and bool(
        matched.get("enabled", False)
    )
    canonical_sweep = config.get("sweep", {}) or {}
    canonical_training_conditions = bool(
        isinstance(canonical_sweep, dict)
        and canonical_sweep.get("conditions")
    )

    def condition_name(condition: Any) -> str:
        if isinstance(condition, dict):
            return str(condition.get("condition") or condition.get("name"))
        return str(condition)
    if input_condition_enabled:
        conditions = [{
            "condition": str(
                input_condition.get("condition")
                or input_condition.get("name")
                or input_condition.get("transform", "original")
            ),
            "transform": str(input_condition.get("transform", "original")),
        }]
    elif matched_plan_enabled:
        conditions = generate_conditions(config)
    elif (
        isinstance(canonical_sweep, dict)
        and bool(canonical_sweep.get("enabled", False))
        and "conditions" in canonical_sweep
    ):
        from .normalization import normalize_conditions

        conditions = [
            {"condition": item["name"], "transform": item["transform"]}
            for item in normalize_conditions(canonical_sweep["conditions"])
        ]
    else:
        conditions = [{"condition": "original", "transform": "original"}]

    model = config.get("model", {}) or {}
    base_model = str(model.get("name", "model"))
    models = {
        str(combination.get("model.name", base_model))
        for combination in combinations
    }
    pretrained_values = {
        bool(combination.get("model.pretrained", model.get("pretrained", True)))
        for combination in combinations
    }
    freeze_values = {
        bool(combination.get(
            "model.freeze_backbone", model.get("freeze_backbone", False)
        ))
        for combination in combinations
    }
    target_cols = dict((config.get("data", {}) or {}).get("target_cols", {}) or {})
    multi_task = config.get("multi_task", {}) or {}
    configured_weights = multi_task.get("loss_weights", {}) or {}
    effective_weights = {
        task: configured_weights.get(task, 1.0) for task in target_cols
    }
    training = config.get("training", {}) or {}
    wandb = config.get("wandb", {}) or {}
    cue = config.get("test_cue_suppression", {}) or {}
    canonical_evaluation = config.get("evaluation", {}) or {}
    canonical_test = (
        canonical_evaluation.get("test_conditions", {}) or {}
        if isinstance(canonical_evaluation, dict)
        else {}
    )
    if isinstance(canonical_test, dict) and "conditions" in canonical_test:
        configured_test_names = [
            str(item.get("name")) if isinstance(item, dict) else str(item)
            for item in canonical_test.get("conditions", [])
        ]
        effective_test_names = (
            configured_test_names
            if bool(canonical_test.get("enabled", False))
            else []
        )
    else:
        configured_test_names, effective_test_names = _fixed_rgb_test_conditions(config)
    matrix = config.get("condition_matrix_evaluation", {}) or {}
    canonical_matrix = (
        canonical_evaluation.get("condition_matrix", {}) or {}
        if isinstance(canonical_evaluation, dict)
        else {}
    )
    if isinstance(canonical_matrix, dict) and canonical_matrix:
        matrix = canonical_matrix
    matrix_enabled = isinstance(matrix, dict) and bool(matrix.get("enabled", False))
    if matrix_enabled and "condition_names" in matrix:
        from ..evaluation.condition_matrix import resolve_condition_matrix_conditions

        matrix_conditions = resolve_condition_matrix_conditions(config)
    elif matrix_enabled:
        matrix_conditions = matrix.get("conditions", [])
    else:
        matrix_conditions = []

    dimensions = []
    if bool((config.get("sweep", {}) or {}).get("enabled", False)):
        dimensions.append("sweep")
    if matched_plan_enabled:
        dimensions.append("matched_condition_training")
    if "matched_condition_training" in dimensions:
        expansion_owner = "matched_condition_training"
    elif "sweep" in dimensions:
        expansion_owner = "sweep"
    else:
        expansion_owner = "none"

    summary = {
        "experiment_type": identify_experiment_type(config),
        "workflow": workflow,
        "expected_model_count": len(models),
        "expected_sweep_combination_count": len(combinations),
        "expected_condition_count": len(conditions),
        "expected_total_run_count": len(combinations) * len(conditions),
        "models": sorted(models),
        "condition_names": [condition["condition"] for condition in conditions],
        "model": {
            "configured_name": base_model,
            "planned_names": sorted(models),
            "pretrained": model.get("pretrained", True),
            "freeze_backbone": model.get("freeze_backbone", False),
            "planned_pretrained_values": sorted(pretrained_values),
            "planned_freeze_backbone_values": sorted(freeze_values),
        },
        "data": {
            "image_size": (
                (config.get("preprocessing", {}) or {}).get("image_size")
                if "preprocessing" in config
                else (config.get("data", {}) or {}).get("image_size")
            ),
        },
        "tasks": {
            "target_columns": target_cols,
            "loss_weights": effective_weights,
        },
        "training": {
            "epochs": training.get("epochs"),
            "batch_size": training.get("batch_size"),
            "learning_rate": training.get("lr"),
            "weight_decay": training.get("weight_decay"),
            "class_weight": training.get("class_weight", True),
            "use_amp": training.get("use_amp", True),
            "label_contract": "complete_supervised_rows",
        },
        "wandb": {
            key: wandb.get(key)
            for key in ("enabled", "mode", "project", "entity", "group", "name")
        },
        "matched_training": {
            "enabled": matched_plan_enabled or input_condition_enabled or canonical_training_conditions,
            "planning_enabled": matched_plan_enabled or canonical_training_conditions,
            "resolved_input_condition_enabled": input_condition_enabled,
            "requested_condition_names": matched.get("condition_names"),
            "resolved_condition_names": [
                condition["condition"] for condition in conditions
            ] if matched_plan_enabled or input_condition_enabled or canonical_training_conditions else [],
        },
        "fixed_rgb_test": {
            "enabled": bool(cue.get("enabled", False)) or bool(canonical_test.get("enabled", False)),
            "requested_condition_names": cue.get("condition_names"),
            "configured_condition_names": configured_test_names,
            "effective_condition_names": effective_test_names,
        },
        "condition_matrix_evaluation": {
            "enabled": matrix_enabled,
            "condition_names": [
                condition_name(condition) for condition in matrix_conditions
            ],
            "test_condition_count": len(matrix_conditions),
            "expected_condition_cells": (
                len(combinations) * len(conditions) * len(matrix_conditions)
            ),
            "expected_task_rows": (
                len(combinations)
                * len(conditions)
                * len(matrix_conditions)
                * len(target_cols)
            ),
            "expands_training_runs": False,
        },
        "expansion": {
            "owner": expansion_owner,
            "dimensions": dimensions,
            "sweep_combination_count": len(combinations),
            "training_condition_count": len(conditions),
            "expected_total_run_count": len(combinations) * len(conditions),
            "expected_internal_training_runs_per_resolved_spec": 1,
        },
    }
    slurm_summary = _slurm_inspection(config)
    if slurm_summary is not None:
        summary["slurm"] = slurm_summary
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_override_items(args.override)
        config = apply_overrides(load_config(args.config), args.override)
        normalization = normalize_config_with_report(config)
        workflow = resolve_workflow(config, args.workflow)
        validate_config(
            config,
            workflow=args.workflow,
            check_paths=args.check_paths,
            check_model_registry=True,
        )
        summary = inspection_summary(config, workflow)
    except (ConfigValidationError, OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = {
        "summary": summary,
        "applied_overrides": list(args.override),
        "resolved_config": config,
        "canonical_resolved_config": normalization.config,
        "compatibility_warnings": [
            warning.as_dict() for warning in normalization.warnings
        ],
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(yaml.safe_dump(payload, sort_keys=False).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
