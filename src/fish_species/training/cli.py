"""Canonical and byte-compatible legacy command-line entry points."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from ..config.loading import load_config
from ..config.normalization import normalize_config
from ..config.overrides import apply_overrides
from ..config.sweeps import generate_sweep_configs
from .modes import get_profile
from .modes import infer_experiment_type
from .modes import resolve_configured_profile
from .modes import resolved_run_name
from .modes import stress_evaluation_enabled
from .modes import validate_training_semantics


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help=(
            "Override config values, e.g. model.name=vit_b_16 "
            "training.lr=1e-5"
        ),
    )
    parser.add_argument(
        "--sweep",
        nargs="*",
        default=[],
        help=(
            "Multi-run sweep, e.g. model.name=resnet18,vit_b_16 "
            "data.image_col=rel_path_seg,rel_path_raw"
        ),
    )
    return parser


def _canonical_parser() -> argparse.ArgumentParser:
    parser = _legacy_parser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    parser.add_argument("--single-run", action="store_true")
    return parser


def resolve_plan(
    config_path: str,
    overrides: list[str],
    sweep: list[str],
    explicit_profile: str | None,
):
    source = load_config(config_path)
    overridden = apply_overrides(source, overrides)
    canonical_requested = any(
        key in overridden for key in ("preprocessing", "augmentation", "evaluation")
    ) or bool((overridden.get("sweep", {}) or {}).get("conditions"))
    base_config = normalize_config(overridden) if canonical_requested else overridden
    configured_profile = overridden.get("training", {}).get("profile")
    if explicit_profile is None and configured_profile is not None:
        raise ValueError(
            "training.profile is legacy-only; select canonical behavior with "
            "the explicit training, hierarchy-loss, W&B, condition, and "
            "experiment switches"
        )
    chosen = explicit_profile
    compatibility_profile = get_profile(str(chosen)) if chosen else None
    expansion_profile = compatibility_profile or resolve_configured_profile(base_config)
    expanded = generate_sweep_configs(
        base_config,
        sweep,
    )
    resolved = []
    resolved_types = []
    resolved_profiles = []
    profile = compatibility_profile

    for item in expanded:
        cfg = copy.deepcopy(item)
        if compatibility_profile is None:
            item_profile = resolve_configured_profile(cfg)
            experiment_type = infer_experiment_type(cfg)
            validate_training_semantics(cfg, item_profile, experiment_type)
            resolved.append(cfg)
            resolved_types.append(experiment_type)
            resolved_profiles.append(item_profile)
            continue

        condition = cfg.get("input_condition", {}) or {}
        transformed = bool(condition.get("enabled", False)) and str(
            condition.get("transform", "original")
        ) != "original"
        stress = bool(
            (cfg.get("test_cue_suppression", {}) or {}).get("enabled", False)
        )

        if profile.loader_mode == "standard":
            default_type = "standard"
        elif stress:
            default_type = "rgb_stress_test"
        elif condition.get("enabled", False):
            default_type = "matched_condition"
        else:
            default_type = "standard"

        experiment_type = str(
            (cfg.get("experiment", {}) or {}).get("type") or default_type
        )
        allowed = {
            "standard",
            "matched_condition",
            "rgb_stress_test",
            "matched_and_rgb_stress",
        }
        if experiment_type not in allowed:
            raise ValueError(f"Unknown experiment.type {experiment_type!r}")
        if (
            experiment_type in {"rgb_stress_test", "matched_and_rgb_stress"}
            and profile.name != "cue_suppression"
        ):
            raise ValueError(
                f"experiment.type={experiment_type} requires profile cue_suppression"
            )
        if stress and transformed:
            raise ValueError(
                "Fixed-RGB stress evaluation requires an original-trained "
                "input condition"
            )
        if (
            experiment_type in {"rgb_stress_test", "matched_and_rgb_stress"}
            and not stress
        ):
            raise ValueError(
                f"experiment.type={experiment_type} requires "
                "test_cue_suppression.enabled=true"
            )
        if profile.loader_mode == "standard" and experiment_type != "standard":
            raise ValueError(
                f"profile {profile.name} requires experiment.type=standard"
            )
        if (
            profile.loader_mode == "condition"
            and experiment_type == "standard"
            and (condition.get("enabled", False) or stress)
        ):
            raise ValueError(
                "cue_suppression standard experiment cannot enable "
                "input_condition or stress evaluation"
            )

        resolved.append(cfg)
        resolved_types.append(experiment_type)

    profile = compatibility_profile or resolved_profiles[0]
    if compatibility_profile is None and any(
        item_profile != profile for item_profile in resolved_profiles[1:]
    ):
        raise ValueError(
            "One canonical invocation cannot sweep over training feature "
            "switches that resolve to different loader or output contracts"
        )

    external = bool(
        (overridden.get("input_condition", {}) or {}).get("enabled", False)
    )
    if external:
        active = []
        if sweep:
            active.append("CLI --sweep")
        for key in ("sweep", "matched_condition_training"):
            if bool((overridden.get(key, {}) or {}).get("enabled", False)):
                active.append(f"{key}.enabled")
        if active or len(resolved) != 1:
            raise ValueError(
                "External input_condition requires exactly one run and all "
                "internal expanders disabled; active: " + ", ".join(active)
            )

    return profile, resolved, resolved_types


def _plan_summary(profile, configs, experiment_types):
    models = sorted({str(c.get("model", {}).get("name")) for c in configs})
    conditions = sorted(
        {
            str(
                (c.get("input_condition", {}) or {}).get("condition")
                or (c.get("input_condition", {}) or {}).get("name", "original")
            )
            for c in configs
        }
    )
    first = configs[0]
    tasks = first.get("data", {}).get("target_cols", {})
    hierarchy = (
        first.get("multi_task", {}).get("hierarchy_loss", {})
        if profile.hierarchy
        else {}
    )
    evaluation = first.get("evaluation", {}) or {}
    matrix_cfg = (
        evaluation.get("condition_matrix", {}) or {}
        if isinstance(evaluation, dict) and "condition_matrix" in evaluation
        else first.get("condition_matrix_evaluation", {}) or {}
    )
    matrix_enabled = bool(matrix_cfg.get("enabled", False))
    if matrix_enabled:
        from ..evaluation.condition_matrix import resolve_condition_matrix_conditions

        matrix_conditions = resolve_condition_matrix_conditions(first)
    else:
        matrix_conditions = []
    summary = {
        "configuration_driven": profile.name == "configured",
        "training_selection": (
            "explicit_config_toggles"
            if profile.name == "configured"
            else "legacy_compatibility"
        ),
        "loader_mode": profile.loader_mode,
        "experiment_type": experiment_types[0],
        "expected_internal_training_runs": len(configs),
        "model_count": len(models),
        "models": models,
        "tasks": tasks,
        "loss_weights": first.get("multi_task", {}).get("loss_weights", {}),
        "normalize_loss_by_active_tasks": first.get("multi_task", {}).get(
            "normalize_loss_by_active_tasks", True
        ),
        "hierarchy_enabled": bool(hierarchy.get("enabled", False)),
        "hierarchy_weight": hierarchy.get("weight"),
        "wandb_enabled": bool(
            profile.wandb and first.get("wandb", {}).get("enabled", False)
        ),
        "labels": "complete_supervised_rows",
        "condition_count": len(conditions),
        "resolved_training_conditions": conditions,
        "post_training_rgb_stress": bool(
            profile.stress_evaluation
            and stress_evaluation_enabled(first)
        ),
        "post_training_condition_matrix": matrix_enabled,
        "condition_matrix_test_conditions": [
            condition.get("condition") or condition.get("name")
            for condition in matrix_conditions
        ],
        "condition_matrix_evaluation_cells_per_training_run": len(
            matrix_conditions
        ),
        "condition_matrix_task_rows_per_training_run": (
            len(matrix_conditions) * len(tasks)
        ),
        "expected_output_paths": [
            str(
                Path(c.get("output", {}).get("out_dir", "outputs"))
                / resolved_run_name(c, profile)
            )
            for c in configs
        ],
    }
    if profile.name != "configured":
        summary["selected_profile"] = profile.name
    return summary


def execute(args, forced_profile: str | None = None):
    profile, configs, experiment_types = resolve_plan(
        args.config,
        args.override,
        args.sweep,
        forced_profile or getattr(args, "profile", None),
    )
    if getattr(args, "single_run", False) and len(configs) != 1:
        raise ValueError(
            f"--single-run requires exactly one resolved run, got {len(configs)}"
        )

    summary = _plan_summary(profile, configs, experiment_types)
    if getattr(args, "dry_run", False) or getattr(
        args, "print_resolved_config", False
    ):
        print(
            json.dumps(
                {"plan": summary, "resolved_configs": configs},
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return []

    import pandas as pd

    from .runner import run_one

    results = []
    for cfg in configs:
        try:
            results.append(run_one(cfg, profile))
        except Exception as exc:
            pipeline_run = cfg.get("pipeline_run", {}) or {}
            if pipeline_run.get("configuration_hash") and int(os.environ.get("RANK", "0")) == 0:
                run_id = str(pipeline_run.get("run_id") or resolved_run_name(cfg, profile))
                out_dir = Path(cfg["output"]["out_dir"]) / run_id
                out_dir.mkdir(parents=True, exist_ok=True)
                text = str(exc).lower()
                if "out of memory" in text:
                    category = "cuda_oom"
                elif "checkpoint" in text or "configuration" in text or "invalid" in text:
                    category = "configuration"
                elif "dataset" in text or "no such file" in text:
                    category = "missing_dataset"
                else:
                    category = "training_failure"
                temporary = out_dir / f".run_status.json.tmp-{os.getpid()}"
                temporary.write_text(json.dumps({
                    "status": "failed", "exit_code": 1,
                    "configuration_hash": pipeline_run["configuration_hash"],
                    "failure_category": category, "error": str(exc),
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(temporary, out_dir / "run_status.json")
            raise
    if int(os.environ.get("RANK", "0")) != 0:
        return results
    out = Path(configs[0]["output"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(out / "multi_run_results.csv", index=False)
    return results


def main(argv=None):
    return execute(_canonical_parser().parse_args(argv))


def legacy_main(profile: str, argv=None):
    return execute(_legacy_parser().parse_args(argv), forced_profile=profile)
