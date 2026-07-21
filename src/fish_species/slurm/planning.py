"""Pure run-spec and dependency planning for canonical SLURM submission."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import yaml

from datetime import datetime

from ..config.overrides import apply_overrides
from ..config.normalization import normalize_config
from ..config.sweeps import expand_sweep_items
from ..config.validation import ConfigValidationError, validate_config
from ..experiments.conditions import (
    condition_overrides,
    format_override,
)
from ..training.modes import infer_experiment_type
from ..training.modes import resolve_configured_profile
from ..training.modes import validate_training_semantics
from .config import SlurmConfigError, validate_slurm_config


_EXTERNAL_DISABLE_OVERRIDES = (
    "sweep.enabled=false",
    "matched_condition_training.enabled=false",
)


@dataclass(frozen=True)
class RunSpec:
    index: int
    run_id: str
    model: str
    training_condition: str
    training_transform: str
    experiment_type: str
    training_mode: str
    overrides: tuple[str, ...]
    trainer_overrides: tuple[str, ...]
    output_relpath: str
    resolved_config: dict[str, Any]
    config_sha256: str

    @property
    def args_text(self) -> str:
        """Legacy-compatible run-spec bytes (scheduler controls stay separate)."""
        return "\n".join(self.overrides) + ("\n" if self.overrides else "")

    @property
    def trainer_command(self) -> tuple[str, ...]:
        return (
            "python",
            "-m",
            "fish_species.training",
            "--config",
            "resolved_run_config.yaml",
            "--single-run",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "run_id": self.run_id,
            "model": self.model,
            "training_condition": self.training_condition,
            "training_transform": self.training_transform,
            "experiment_type": self.experiment_type,
            "training_mode": self.training_mode,
            "trainer_selection": "configuration",
            "overrides": list(self.overrides),
            "trainer_overrides": list(self.trainer_overrides),
            "output_relpath": self.output_relpath,
            "config_sha256": self.config_sha256,
            "trainer_command": list(self.trainer_command),
        }


@dataclass(frozen=True)
class Dependency:
    upstream: str
    downstream: str
    kind: str


@dataclass(frozen=True)
class SubmissionPlan:
    schema_version: int
    experiment_type: str
    cluster_profile: str
    results_root: str
    array_size: int
    array_max_active: int
    models: tuple[str, ...]
    conditions: tuple[str, ...]
    training_modes: tuple[str, ...]
    run_specs: tuple[RunSpec, ...]
    dependencies: tuple[Dependency, ...]
    canonical_trainer_command: tuple[str, ...]
    resolved_config_sha256: str

    @property
    def expected_internal_training_runs_per_task(self) -> int:
        return 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_type": self.experiment_type,
            "cluster_profile": self.cluster_profile,
            "results_root": self.results_root,
            "array_size": self.array_size,
            "array_max_active": self.array_max_active,
            "models": list(self.models),
            "conditions": list(self.conditions),
            "training_modes": list(self.training_modes),
            "trainer_selection": "configuration",
            "expected_internal_training_runs_per_task": 1,
            "canonical_trainer_command": list(self.canonical_trainer_command),
            "resolved_config_sha256": self.resolved_config_sha256,
            "dependencies": [dependency.__dict__ for dependency in self.dependencies],
            "run_specs": [run_spec.as_dict() for run_spec in self.run_specs],
        }


def _config_hash(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_condition(condition: dict[str, Any]) -> dict[str, Any]:
    """Materialise the established run-spec spelling from one canonical object."""
    legacy = {
        "condition": condition["name"],
        "feature": condition.get("feature", "baseline"),
        "transform": condition["transform"],
        "strength": condition.get("strength", 0.0),
    }
    parameters = condition.get("parameters", {}) or {}
    if not isinstance(parameters, dict):
        raise SlurmConfigError("Canonical condition parameters must be a mapping")
    legacy.update(copy.deepcopy(parameters))
    return legacy


def generate_external_specs(
    config: dict[str, Any],
) -> list[tuple[str, list[str], str, str]]:
    """Expand every experiment through the one canonical sweep engine."""
    canonical = normalize_config(config)
    items = expand_sweep_items(canonical)
    planning = (config.get("slurm", {}) or {}).get("planning", {}) or {}
    compatibility_kind = str(planning.get("external_expansion", "sweep"))
    evaluation = canonical.get("evaluation", {}) or {}
    test_schedule = (
        evaluation.get("test_conditions", {}) or {}
        if isinstance(evaluation, dict)
        else {}
    )
    evaluate_original = bool(
        isinstance(test_schedule, dict)
        and test_schedule.get("evaluate_original_training", False)
    )
    legacy_matched = config.get("matched_condition_training", {}) or {}
    legacy_matched_enabled = bool(
        isinstance(legacy_matched, dict)
        and legacy_matched.get("enabled", False)
    )

    specs: list[tuple[str, list[str], str, str]] = []
    for index, item in enumerate(items):
        assignments = item.parameter_values
        overrides = [
            f"{key}={format_override(value)}"
            for key, value in assignments.items()
        ]
        model = str(
            assignments.get(
                "model.name", canonical.get("model", {}).get("name", "model")
            )
        )
        condition_name = "original"
        run_id = f"run_{index:03d}_{model}_{condition_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
        if item.condition is not None:
            condition = _legacy_condition(item.condition)
            condition_name = str(condition["condition"])
            overrides.extend(condition_overrides(condition))
            run_id = f"run_{index:03d}_{model}_{condition_name}"
            if compatibility_kind == "dual_cue" or legacy_matched_enabled:
                cue_enabled = (
                    evaluate_original
                    and condition["transform"] == "original"
                )
                overrides.append(
                    "test_cue_suppression.enabled="
                    + ("true" if cue_enabled else "false")
                )
                overrides.append("matched_condition_training.enabled=false")
        specs.append((run_id, overrides, model, condition_name))
    return specs


def _validate_canonical_training_semantics(config: dict[str, Any]) -> str:
    if "profile" in (config.get("training", {}) or {}):
        raise SlurmConfigError(
            "training.profile is not supported by canonical SLURM runs; "
            "set the explicit trainer feature switches instead"
        )
    profile = resolve_configured_profile(config)
    experiment_type = infer_experiment_type(config)
    try:
        validate_training_semantics(config, profile, experiment_type)
    except ValueError as exc:
        raise SlurmConfigError(str(exc)) from exc
    return experiment_type


def _resolve_one_run(
    config: dict[str, Any], overrides: list[str]
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    scientific = copy.deepcopy(config)
    scientific.pop("slurm", None)
    resolved = apply_overrides(scientific, [*overrides, *_EXTERNAL_DISABLE_OVERRIDES])
    if bool((resolved.get("sweep", {}) or {}).get("enabled", False)):
        raise SlurmConfigError("External run specification left sweep.enabled=true")
    if bool((resolved.get("matched_condition_training", {}) or {}).get("enabled", False)):
        raise SlurmConfigError(
            "External run specification left matched_condition_training.enabled=true"
        )
    per_run_controls = _EXTERNAL_DISABLE_OVERRIDES
    try:
        validate_config(
            resolved,
            workflow="training",
            check_paths=False,
            check_model_registry=False,
        )
    except ConfigValidationError as exc:
        raise SlurmConfigError(str(exc)) from exc
    experiment_type = _validate_canonical_training_semantics(resolved)
    return resolved, per_run_controls, experiment_type


def _dependency_plan(slurm: dict[str, Any]) -> tuple[Dependency, ...]:
    dependencies: list[Dependency] = []
    setup = slurm.get("setup", {})
    collection = slurm.get("collection", {})
    cleanup = slurm.get("cleanup", {})
    if bool(setup.get("enabled", False)):
        dependencies.append(Dependency("setup", "train_array", "afterok"))
    if bool(collection.get("enabled", False)):
        dependencies.append(Dependency("train_array", "collect", "afterany"))
    if bool(cleanup.get("enabled", False)):
        dependencies.append(Dependency("train_array", "cleanup", "afterany"))
    return tuple(dependencies)


def _validate_result_paths(
    results_root: str, specs: list[RunSpec], config: dict[str, Any]
) -> None:
    root = PurePosixPath(results_root)
    if str(root) in {"", ".", "/"}:
        raise SlurmConfigError(
            "slurm.paths.results_root must be a dedicated result directory"
        )
    protected = []
    paths = config.get("slurm", {}).get("paths", {})
    for key in ("project_root", "data_root", "cache_root"):
        raw = paths.get(key)
        if isinstance(raw, str) and raw:
            protected.append(PurePosixPath(raw))
    if any(root == item for item in protected):
        raise SlurmConfigError("Result root collides with a project, data, or cache root")
    outputs = [root / spec.output_relpath for spec in specs]
    if len(outputs) != len(set(outputs)):
        raise SlurmConfigError("Generated run specifications have colliding result paths")
    if any(".." in output.parts for output in outputs):
        raise SlurmConfigError("Generated result paths must remain below results_root")


def plan_submission(config: dict[str, Any]) -> SubmissionPlan:
    """Return a validated submission plan without writing or submitting anything."""
    validate_slurm_config(config)
    planning = config.get("slurm", {}).get("planning", {}) or {}
    experiment_type = str(planning.get("experiment_type", "standard"))
    raw_specs = generate_external_specs(config)
    if not raw_specs:
        raise SlurmConfigError("No run specifications were generated")

    specs: list[RunSpec] = []
    for index, (run_id, overrides, model, condition) in enumerate(raw_specs):
        resolved, trainer_overrides, run_experiment_type = _resolve_one_run(
            config, overrides
        )
        transform = str(
            (resolved.get("input_condition", {}) or {}).get(
                "transform", "original"
            )
        )
        specs.append(
            RunSpec(
                index=index,
                run_id=run_id,
                model=model,
                training_condition=condition,
                training_transform=transform,
                experiment_type=run_experiment_type,
                training_mode=str(
                    (resolved.get("training", {}) or {}).get("mode", "multitask")
                ),
                overrides=tuple(overrides),
                trainer_overrides=trainer_overrides,
                output_relpath=run_id,
                resolved_config=resolved,
                config_sha256=_config_hash(resolved),
            )
        )

    run_ids = [spec.run_id for spec in specs]
    if len(run_ids) != len(set(run_ids)):
        duplicates = sorted(
            {run_id for run_id in run_ids if run_ids.count(run_id) > 1}
        )
        raise SlurmConfigError(f"Duplicate run identifiers: {duplicates}")
    config_hashes = [spec.config_sha256 for spec in specs]
    if len(config_hashes) != len(set(config_hashes)):
        raise SlurmConfigError("Duplicate externally resolved training configurations")

    slurm = config["slurm"]
    results_root = str(slurm.get("paths", {}).get("results_root", "outputs_slurm"))
    _validate_result_paths(results_root, specs, config)
    models = tuple(dict.fromkeys(spec.model for spec in specs))
    conditions = tuple(dict.fromkeys(spec.training_condition for spec in specs))
    training_modes = tuple(dict.fromkeys(spec.training_mode for spec in specs))
    if len(training_modes) != 1:
        raise SlurmConfigError(
            f"A submission plan must use one training mode, got {training_modes}"
        )
    plan = SubmissionPlan(
        schema_version=2,
        experiment_type=experiment_type,
        cluster_profile=str(slurm.get("cluster_profile", "unspecified")),
        results_root=results_root,
        array_size=len(specs),
        array_max_active=int(slurm.get("array", {}).get("max_active", 1)),
        models=models,
        conditions=conditions,
        training_modes=training_modes,
        run_specs=tuple(specs),
        dependencies=_dependency_plan(slurm),
        canonical_trainer_command=(
            "python",
            "-m",
            "fish_species.training",
            "--config",
            "resolved_run_config.yaml",
            "--single-run",
        ),
        resolved_config_sha256=_config_hash(config),
    )
    if plan.array_size != len(plan.run_specs):
        raise SlurmConfigError("Array size differs from generated run-spec count")
    return plan
