"""Pure canonical configuration and run-identity helpers for logging."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def flatten_slash_config(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a resolved configuration into stable slash-separated keys."""
    flattened: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(flatten_slash_config(child, child_prefix))
    elif prefix:
        flattened[prefix] = value
    return flattened


def canonical_condition_relation(
    train_condition: str, test_condition: str
) -> str:
    """Return the canonical relation for one train/test condition pair."""
    train_name = str(train_condition)
    test_name = str(test_condition)
    if train_name == "original" and test_name == "original":
        return "original"
    if train_name == test_name:
        return "matched"
    if train_name == "original":
        return "rgb_stress"
    return "cross_condition"


def condition_name(condition: str | Mapping[str, Any] | None) -> str:
    if condition is None:
        return "original"
    if isinstance(condition, Mapping):
        return str(
            condition.get("condition")
            or condition.get("name")
            or condition.get("transform")
            or "original"
        )
    return str(condition)


def condition_metadata(
    condition: str | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(condition, Mapping):
        name = condition_name(condition)
        return {
            "name": name,
            "transform": "original" if name == "original" else None,
            "strength": 0.0 if name == "original" else None,
        }
    return {
        "name": condition_name(condition),
        "transform": condition.get("transform", "original"),
        "strength": condition.get("strength", 0.0),
    }


def resolved_training_condition(cfg: Mapping[str, Any]) -> dict[str, Any]:
    canonical = cfg.get("training_condition", {}) or {}
    if isinstance(canonical, Mapping) and (
        canonical.get("name") or canonical.get("condition")
    ):
        return condition_metadata(canonical)
    raw = cfg.get("input_condition", {}) or {}
    if isinstance(raw, Mapping) and bool(raw.get("enabled", False)):
        return condition_metadata(raw)

    data = cfg.get("data", {}) or {}
    retention = (
        data.get("colour_retention", 1.0)
        if isinstance(data, Mapping)
        else 1.0
    )
    try:
        retention_value = float(retention)
    except (TypeError, ValueError):
        retention_value = 1.0
    if retention_value != 1.0:
        return {
            "name": f"colour_{int(round(retention_value * 100)):03d}pct",
            "transform": "saturation",
            "strength": 1.0 - retention_value,
        }
    return {"name": "original", "transform": "original", "strength": 0.0}


def experiment_type(cfg: Mapping[str, Any]) -> str:
    experiment = cfg.get("experiment", {}) or {}
    if isinstance(experiment, Mapping) and experiment.get("type"):
        return str(experiment["type"])
    condition = cfg.get("input_condition", {}) or {}
    stress = cfg.get("test_cue_suppression", {}) or {}
    condition_enabled = isinstance(condition, Mapping) and bool(
        condition.get("enabled", False)
    )
    stress_enabled = isinstance(stress, Mapping) and bool(
        stress.get("enabled", False)
    )
    if condition_enabled and stress_enabled:
        return "matched_and_rgb_stress"
    if stress_enabled:
        return "rgb_stress_test"
    if condition_enabled:
        return "matched_condition"
    return "standard"


def _move(
    flattened: dict[str, Any], source: str, destination: str
) -> None:
    if source not in flattened:
        return
    flattened.setdefault(destination, flattened[source])
    del flattened[source]


def canonical_tracking_config(
    cfg: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    """Return one uploaded field per resolved setting.

    Compatibility-only input paths are migrated in the uploaded view.  The
    source configuration is not mutated and no legacy duplicate columns are
    retained beside their canonical forms.
    """
    flattened = flatten_slash_config(cfg)
    flattened.update(
        {f"runtime/{key}": value for key, value in runtime.items()}
    )

    _move(flattened, "seed", "training/seed")
    _move(flattened, "data/image_size", "preprocessing/image_size")
    _move(
        flattened,
        "augmentation/horizontal_flip/enabled",
        "augmentation/horizontal_flip",
    )
    _move(
        flattened,
        "augmentation/vertical_flip/enabled",
        "augmentation/vertical_flip",
    )
    _move(
        flattened,
        "augmentation/rotation/degrees",
        "augmentation/rotation_degrees",
    )

    input_prefix = "input_condition/"
    input_fields = {
        key: flattened.pop(key)
        for key in list(flattened)
        if key.startswith(input_prefix)
    }
    if input_fields:
        condition_value = input_fields.pop("input_condition/condition", None)
        name_value = input_fields.pop("input_condition/name", None)
        name = (
            condition_value
            or name_value
            or input_fields.get("input_condition/transform")
            or "original"
        )
        flattened.setdefault("training_condition/name", name)
        for key, value in input_fields.items():
            suffix = key.removeprefix(input_prefix)
            flattened.setdefault(f"training_condition/{suffix}", value)
    else:
        condition = resolved_training_condition(cfg)
        flattened.setdefault("training_condition/name", condition["name"])
        flattened.setdefault(
            "training_condition/transform", condition["transform"]
        )
        flattened.setdefault(
            "training_condition/strength", condition["strength"]
        )

    return flattened


def identity_summary(
    cfg: Mapping[str, Any], *, run_name: str
) -> dict[str, Any]:
    """Return non-duplicated filter metadata for the W&B run summary."""
    model = cfg.get("model", {}) or {}
    training = cfg.get("training", {}) or {}
    condition = resolved_training_condition(cfg)
    architecture = model.get("name") if isinstance(model, Mapping) else None
    seed = (
        training.get("seed", cfg.get("seed"))
        if isinstance(training, Mapping)
        else cfg.get("seed")
    )
    return {
        "run_name": run_name,
        "architecture": architecture,
        "training_condition": condition["name"],
        "test_condition": condition["name"],
        "experiment_type": experiment_type(cfg),
        "seed": seed,
        "condition_relation": canonical_condition_relation(
            condition["name"], condition["name"]
        ),
    }


__all__ = [
    "canonical_condition_relation",
    "canonical_tracking_config",
    "condition_metadata",
    "condition_name",
    "experiment_type",
    "flatten_slash_config",
    "identity_summary",
    "resolved_training_condition",
]
