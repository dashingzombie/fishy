"""Resolve canonical fish-training behaviour."""

from __future__ import annotations

from dataclasses import dataclass

from .naming import make_run_name


@dataclass(frozen=True)
class TrainingProfile:
    """Fully resolved runtime behaviour.

    ``PROFILES`` below are compatibility adapters for historical commands.
    Preferred training derives the same fields directly from configuration via
    :func:`resolve_configured_profile`.
    """

    name: str
    loader_mode: str
    hierarchy: bool
    wandb: bool
    run_summary: bool = False
    stress_evaluation: bool = False


PROFILES = {
    "standard": TrainingProfile("standard", "standard", False, False),
    "hierarchy": TrainingProfile("hierarchy", "standard", True, False),
    "hierarchy_wandb": TrainingProfile(
        "hierarchy_wandb", "standard", True, True
    ),
    "cue_suppression": TrainingProfile(
        "cue_suppression", "condition", True, True, True, True
    ),
}
DEFAULT_PROFILE = "hierarchy_wandb"
CONFIGURED_PROFILE = "configured"


def get_profile(name: str) -> TrainingProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown training profile {name!r}; choose from {sorted(PROFILES)}"
        ) from exc


def stress_evaluation_enabled(config: dict) -> bool:
    legacy_enabled = bool(
        (config.get("test_cue_suppression", {}) or {}).get("enabled", False)
    )
    evaluation = config.get("evaluation", {}) or {}
    if isinstance(evaluation, dict) and "test_conditions" in evaluation:
        schedule = evaluation.get("test_conditions", {}) or {}
        return legacy_enabled or (
            isinstance(schedule, dict) and bool(schedule.get("enabled", False))
        )
    return legacy_enabled


def infer_experiment_type(config: dict) -> str:
    """Infer a per-process experiment type from explicit feature switches."""
    configured = str((config.get("experiment", {}) or {}).get("type") or "")
    if configured:
        return configured

    condition = config.get("input_condition", {}) or {}
    stress_enabled = stress_evaluation_enabled(config)
    if stress_enabled and bool(condition.get("enabled", False)):
        return "matched_and_rgb_stress"
    if stress_enabled:
        return "rgb_stress_test"
    if bool(condition.get("enabled", False)):
        return "matched_condition"
    return "standard"


def resolve_configured_profile(config: dict) -> TrainingProfile:
    """Build runtime behaviour without selecting a named training profile."""
    experiment_type = infer_experiment_type(config)
    condition_enabled = bool(
        (config.get("input_condition", {}) or {}).get("enabled", False)
    )
    stress_enabled = stress_evaluation_enabled(config)
    if condition_enabled or stress_enabled or experiment_type in {
        "rgb_stress_test",
        "matched_and_rgb_stress",
    }:
        loader_mode = "condition"
    elif experiment_type == "matched_condition":
        loader_mode = "condition"
    else:
        loader_mode = "standard"

    return TrainingProfile(
        name=CONFIGURED_PROFILE,
        loader_mode=loader_mode,
        hierarchy=bool(
            (config.get("multi_task", {}).get("hierarchy_loss", {}) or {}).get(
                "enabled", False
            )
        ),
        wandb=bool((config.get("wandb", {}) or {}).get("enabled", False)),
        run_summary=loader_mode == "condition",
        stress_evaluation=stress_enabled,
    )


def validate_training_semantics(
    config: dict,
    profile: TrainingProfile,
    experiment_type: str,
) -> None:
    """Reject contradictory switches before loaders, outputs, or W&B start."""
    allowed = {
        "standard",
        "matched_condition",
        "rgb_stress_test",
        "matched_and_rgb_stress",
    }
    if experiment_type not in allowed:
        raise ValueError(f"Unknown experiment.type {experiment_type!r}")

    condition = config.get("input_condition", {}) or {}
    condition_enabled = bool(condition.get("enabled", False))
    transformed = condition_enabled and str(
        condition.get("transform", "original")
    ).lower() != "original"
    stress_enabled = stress_evaluation_enabled(config)
    if stress_enabled and transformed:
        raise ValueError(
            "Fixed-RGB stress evaluation requires an original-trained input "
            "condition"
        )
    if experiment_type == "standard" and (condition_enabled or stress_enabled):
        raise ValueError(
            "experiment.type=standard cannot enable input_condition or "
            "fixed-RGB stress evaluation"
        )
    if experiment_type == "matched_condition" and stress_enabled:
        raise ValueError(
            "experiment.type=matched_condition cannot enable fixed-RGB stress; "
            "use matched_and_rgb_stress for an original matched condition"
        )
    if (
        experiment_type in {"rgb_stress_test", "matched_and_rgb_stress"}
        and not stress_enabled
    ):
        raise ValueError(
            f"experiment.type={experiment_type} requires "
            "test_cue_suppression.enabled=true"
        )
    if experiment_type == "rgb_stress_test" and transformed:
        raise ValueError(
            "experiment.type=rgb_stress_test requires original/RGB training"
        )
    if experiment_type == "rgb_stress_test" and condition_enabled:
        raise ValueError(
            "experiment.type=rgb_stress_test cannot enable a matched training "
            "condition; use matched_and_rgb_stress for an original condition"
        )
    if experiment_type == "matched_and_rgb_stress" and not condition_enabled:
        raise ValueError(
            "experiment.type=matched_and_rgb_stress requires an enabled "
            "original matched training condition"
        )
    if experiment_type == "matched_and_rgb_stress" and transformed:
        raise ValueError(
            "experiment.type=matched_and_rgb_stress requires an original "
            "matched training condition"
        )

    if profile.loader_mode == "standard" and experiment_type != "standard":
        raise ValueError(
            f"Resolved standard loaders are incompatible with "
            f"experiment.type={experiment_type}"
        )


def resolved_run_name(cfg: dict, profile: TrainingProfile) -> str:
    base = make_run_name(cfg)
    if profile.loader_mode == "standard":
        return base

    raw = cfg.get("input_condition", {}) or {}
    condition = (
        str(
            raw.get("condition")
            or raw.get("name")
            or raw.get("transform", "original")
        )
        if raw.get("enabled", False)
        else "original"
    )
    suffix = f"train_{condition.replace(' ', '_')}"

    return base if suffix in base else f"{base}_{suffix}"
