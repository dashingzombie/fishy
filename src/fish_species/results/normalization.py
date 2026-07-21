"""Pure compatibility normalization for heterogeneous result metadata.

The scientific files below a result root are immutable inputs to this module.
These helpers only build additive, dashboard-facing identities from historical
and canonical configuration shapes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


_CONDITION_IDENTITY_KEYS = frozenset(
    {"enabled", "name", "condition", "feature", "transform", "strength", "parameters"}
)


def nested(value: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Return one nested value without assuming every level is a mapping."""

    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    result = as_float(value)
    return int(result) if result is not None and result.is_integer() else None


def stable_json(value: Any) -> str:
    """Return a deterministic display/filter value for structured metadata."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _summary_parameters(summary: Mapping[str, Any]) -> dict[str, Any]:
    value = summary.get("train_condition_parameters")
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def training_condition_identity(
    config: Mapping[str, Any], summary: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve one condition while retaining all transform-specific parameters.

    ``name`` is canonical. ``condition`` and top-level transform parameters are
    historical aliases. A scalar ``strength`` is retained for compatibility but
    is never used as a substitute for transform-specific parameters such as
    ``sigma``, ``grid_size``, or ``retention``.
    """

    summary = summary or {}
    raw = _mapping(config.get("input_condition"))
    parameters = _mapping(raw.get("parameters"))
    parameters.update(
        {
            str(key): value
            for key, value in raw.items()
            if key not in _CONDITION_IDENTITY_KEYS
        }
    )
    parameters.update(_summary_parameters(summary))

    name = (
        summary.get("train_condition")
        or raw.get("name")
        or raw.get("condition")
        or raw.get("transform")
        or "original"
    )
    transform = summary.get("train_transform") or raw.get("transform")
    if transform is None and str(name) == "original":
        transform = "original"
    feature = summary.get("train_feature") or raw.get("feature")
    if feature is None and str(name) == "original":
        feature = "baseline"
    strength = as_float(
        summary.get("train_strength")
        if summary.get("train_strength") is not None
        else raw.get("strength")
    )
    return {
        "name": str(name),
        "feature": str(feature) if feature is not None else None,
        "transform": str(transform) if transform is not None else None,
        "strength": strength,
        "parameters": parameters,
        "enabled": bool(raw.get("enabled", False)),
    }


def canonical_condition_relation(train_condition: Any, test_condition: Any) -> str:
    """Return the logging relation, where original/original is explicit."""

    train_name = str(train_condition)
    test_name = str(test_condition)
    if train_name == "original" and test_name == "original":
        return "original"
    if train_name == test_name:
        return "matched"
    if train_name == "original":
        return "rgb_stress"
    return "cross_condition"


def matrix_evaluation_relation(train_condition: Any, test_condition: Any) -> str:
    """Return the historical scientific matrix relation.

    Unlike the logging relation, the matrix contract classifies
    original/original as a matched evaluation. Keeping both functions explicit
    prevents a dashboard alias from silently changing an existing CSV schema.
    """

    train_name = str(train_condition)
    test_name = str(test_condition)
    if train_name == test_name:
        return "matched"
    if train_name == "original":
        return "rgb_stress"
    return "cross_condition"


def experiment_type(config: Mapping[str, Any]) -> str:
    configured = nested(config, "experiment", "type")
    if configured is not None:
        return str(configured)
    condition = training_condition_identity(config)
    stress = bool(
        nested(config, "evaluation", "test_conditions", "enabled", default=False)
        or nested(config, "test_cue_suppression", "enabled", default=False)
    )
    if condition["enabled"] and stress:
        return "matched_and_rgb_stress"
    if stress:
        return "rgb_stress_test"
    if condition["enabled"]:
        return "matched_condition"
    return "standard"


def _enabled_and_value(section: Any, value_key: str) -> tuple[Any, Any]:
    if isinstance(section, Mapping):
        return section.get("enabled"), section.get(value_key)
    return None, section


def _flatten_parameter_facets(
    value: Mapping[str, Any], prefix: str = "condition_parameter"
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in sorted(value, key=str):
        child = value[key]
        path = f"{prefix}.{key}"
        if isinstance(child, Mapping):
            output.update(_flatten_parameter_facets(child, path))
        elif isinstance(child, (list, tuple, set)):
            output[path] = stable_json(list(child))
        else:
            output[path] = child
    return output


def hyperparameter_facets(
    config: Mapping[str, Any], summary: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Extract stable old and new dashboard facets from a resolved config."""

    weights = _mapping(nested(config, "multi_task", "loss_weights", default={}))
    preprocessing = _mapping(config.get("preprocessing"))
    augmentation = _mapping(config.get("augmentation"))
    normalisation = _mapping(preprocessing.get("normalisation"))
    horizontal_enabled, horizontal_probability = _enabled_and_value(
        augmentation.get("horizontal_flip"), "probability"
    )
    vertical_enabled, vertical_probability = _enabled_and_value(
        augmentation.get("vertical_flip"), "probability"
    )
    rotation_enabled, rotation_degrees = _enabled_and_value(
        augmentation.get("rotation"), "degrees"
    )
    condition = training_condition_identity(config, summary)
    image_size = preprocessing.get("image_size")
    if image_size is None:
        image_size = nested(config, "data", "image_size")
    training_seed = nested(config, "training", "seed")
    if training_seed is None:
        training_seed = config.get("seed")

    facets = {
        # Historical names retained byte-for-byte.
        "epochs": as_int(nested(config, "training", "epochs")),
        "batch_size": as_int(nested(config, "training", "batch_size")),
        "learning_rate": as_float(nested(config, "training", "lr")),
        "weight_decay": as_float(nested(config, "training", "weight_decay")),
        "class_weight": nested(config, "training", "class_weight"),
        "use_amp": nested(config, "training", "use_amp"),
        "pretrained": nested(config, "model", "pretrained"),
        "freeze_backbone": nested(config, "model", "freeze_backbone"),
        "seed": as_int(training_seed),
        "genus_loss_weight": as_float(weights.get("genus")),
        "species_loss_weight": as_float(weights.get("species")),
        "hierarchy_loss_enabled": bool(
            nested(config, "multi_task", "hierarchy_loss", "enabled", default=False)
        ),
        "hierarchy_loss_weight": as_float(
            nested(config, "multi_task", "hierarchy_loss", "weight")
        ),
        "wandb_enabled": bool(nested(config, "wandb", "enabled", default=False)),
        "early_stopping_enabled": bool(
            nested(config, "early_stopping", "enabled", default=False)
        ),
        "early_stopping_patience": as_int(
            nested(config, "early_stopping", "patience")
        ),
        # Canonical additive names.
        "experiment_type": experiment_type(config),
        "image_size": as_int(image_size),
        "normalisation_enabled": (
            normalisation.get("enabled") if normalisation else None
        ),
        "normalisation_mean": (
            stable_json(normalisation.get("mean"))
            if normalisation.get("mean") is not None
            else None
        ),
        "normalisation_std": (
            stable_json(normalisation.get("std"))
            if normalisation.get("std") is not None
            else None
        ),
        "augmentation_enabled": augmentation.get("enabled"),
        "horizontal_flip_enabled": horizontal_enabled,
        "horizontal_flip_probability": as_float(horizontal_probability),
        "vertical_flip_enabled": vertical_enabled,
        "vertical_flip_probability": as_float(vertical_probability),
        "rotation_enabled": rotation_enabled,
        "rotation_degrees": as_float(rotation_degrees),
        "wandb_mode": nested(config, "wandb", "mode"),
        "wandb_group": nested(config, "wandb", "group"),
        "wandb_job_type": nested(config, "wandb", "job_type"),
        "condition_parameters": stable_json(condition["parameters"]),
    }
    facets.update(_flatten_parameter_facets(condition["parameters"]))
    return facets


__all__ = [
    "as_float",
    "as_int",
    "canonical_condition_relation",
    "experiment_type",
    "hyperparameter_facets",
    "matrix_evaluation_relation",
    "nested",
    "stable_json",
    "training_condition_identity",
]
