from __future__ import annotations

import itertools
import math
import re
from typing import Any


def inclusive_sequence(start: float, stop: float, step: float) -> list[float]:
    start = float(start)
    stop = float(stop)
    step = abs(float(step))
    if step <= 0:
        raise ValueError("step must be greater than zero")

    direction = -1.0 if start > stop else 1.0
    tolerance = step * 1e-6
    values: list[float] = []
    current = start
    if direction < 0:
        while current >= stop - tolerance:
            values.append(round(current, 10))
            current -= step
    else:
        while current <= stop + tolerance:
            values.append(round(current, 10))
            current += step
    if not values or not math.isclose(values[-1], stop, abs_tol=tolerance):
        values.append(stop)
    return values


def slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "value"


def format_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def generate_conditions(config: dict) -> list[dict]:
    # Canonical experiment files declare complete training conditions directly.
    # Keep this historical helper as a thin spelling adapter for callers that
    # still consume the legacy runtime shape; expansion remains owned by the
    # config normalizer and generic sweep engine.
    sweep_config = config.get("sweep", {}) or {}
    canonical_conditions = (
        sweep_config.get("conditions")
        if isinstance(sweep_config, dict)
        else None
    )
    if canonical_conditions:
        from ..config.normalization import normalize_conditions

        return [
            {
                "condition": condition["name"],
                "feature": condition.get("feature", "baseline"),
                "transform": condition["transform"],
                "strength": condition.get("strength", 0.0),
                **dict(condition.get("parameters", {}) or {}),
            }
            for condition in normalize_conditions(canonical_conditions)
        ]

    matched_config = config.get("matched_condition_training", {}) or {}
    if not bool(matched_config.get("enabled", False)):
        return [{
            "condition": "original",
            "feature": "baseline",
            "transform": "original",
            "strength": 0.0,
        }]

    cue_config = config.get("test_cue_suppression", {}) or {}
    include_original = bool(matched_config.get("include_original", True))
    deduplicate = bool(matched_config.get("deduplicate_equivalent_conditions", True))
    conditions: list[dict] = []

    if include_original:
        conditions.append({
            "condition": "original",
            "feature": "baseline",
            "transform": "original",
            "strength": 0.0,
        })

    saturation_config = cue_config.get("saturation", {}) or {}
    if bool(saturation_config.get("enabled", True)):
        values = saturation_config.get("values")
        if values is None:
            values = inclusive_sequence(
                saturation_config.get("start", 1.0),
                saturation_config.get("stop", 0.0),
                saturation_config.get("step", 0.01),
            )
        for raw_retention in values:
            retention = float(raw_retention)
            if not 0.0 <= retention <= 1.0:
                raise ValueError(f"Saturation retention must be in [0, 1], got {retention}")
            if deduplicate and include_original and math.isclose(retention, 1.0, abs_tol=1e-12):
                continue
            grayscale_enabled = bool((cue_config.get("grayscale", {}) or {}).get("enabled", True))
            if deduplicate and grayscale_enabled and math.isclose(retention, 0.0, abs_tol=1e-12):
                continue
            percentage = int(round(retention * 100))
            conditions.append({
                "condition": f"saturation_{percentage:03d}pct",
                "feature": "colour",
                "transform": "saturation",
                "strength": round(float(1.0 - retention), 10),
                "retention": retention,
            })

    grayscale_config = cue_config.get("grayscale", {}) or {}
    if bool(grayscale_config.get("enabled", True)):
        conditions.append({
            "condition": "grayscale",
            "feature": "colour",
            "transform": "grayscale",
            "strength": 1.0,
        })

    channel_config = cue_config.get("channel_shuffle", {}) or {}
    if bool(channel_config.get("enabled", True)):
        for order in channel_config.get("orders", [[2, 0, 1]]):
            order = [int(index) for index in order]
            if sorted(order) != [0, 1, 2]:
                raise ValueError(f"Invalid RGB channel order: {order}")
            conditions.append({
                "condition": "channel_shuffle_" + "".join(str(index) for index in order),
                "feature": "colour",
                "transform": "channel_shuffle",
                "strength": 1.0,
                "order": order,
            })

    bilateral_config = cue_config.get("bilateral_filter", {}) or {}
    if bool(bilateral_config.get("enabled", True)):
        settings = bilateral_config.get("settings", [
            {"diameter": 5, "sigma_colour": 25, "sigma_space": 25},
            {"diameter": 7, "sigma_colour": 50, "sigma_space": 50},
            {"diameter": 9, "sigma_colour": 100, "sigma_space": 100},
        ])
        for setting in settings:
            diameter = int(setting["diameter"])
            sigma_colour = float(setting["sigma_colour"])
            sigma_space = float(setting["sigma_space"])
            conditions.append({
                "condition": f"bilateral_d{diameter}_c{sigma_colour:g}_s{sigma_space:g}",
                "feature": "texture",
                "transform": "bilateral_filter",
                "strength": sigma_colour,
                "diameter": diameter,
                "sigma_colour": sigma_colour,
                "sigma_space": sigma_space,
            })

    gaussian_config = cue_config.get("gaussian_blur", {}) or {}
    if bool(gaussian_config.get("enabled", True)):
        for raw_sigma in gaussian_config.get("sigmas", [0.5, 1.0, 2.0, 4.0]):
            sigma = float(raw_sigma)
            conditions.append({
                "condition": f"gaussian_sigma_{sigma:g}",
                "feature": "texture",
                "transform": "gaussian_blur",
                "strength": sigma,
                "sigma": sigma,
            })

    patch_config = cue_config.get("patch_shuffle", {}) or {}
    if bool(patch_config.get("enabled", True)):
        seed = int(patch_config.get("seed", config.get("seed", 0)))
        for raw_grid_size in patch_config.get("grid_sizes", [2, 4, 8]):
            grid_size = int(raw_grid_size)
            conditions.append({
                "condition": f"patch_shuffle_grid_{grid_size}",
                "feature": "shape",
                "transform": "patch_shuffle",
                "strength": grid_size,
                "grid_size": grid_size,
                "seed": seed,
            })

    requested_names = matched_config.get("condition_names")
    if requested_names:
        requested = {str(name) for name in requested_names}
        conditions = [condition for condition in conditions if condition["condition"] in requested]
        missing = requested - {condition["condition"] for condition in conditions}
        if missing:
            raise ValueError(f"Unknown matched training condition names: {sorted(missing)}")

    names = [condition["condition"] for condition in conditions]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate matched training conditions: {duplicates}")
    if not conditions:
        raise ValueError("No matched training conditions were generated")
    return conditions


def sweep_combinations(config: dict) -> list[dict[str, Any]]:
    sweep_config = config.get("sweep", {}) or {}
    if not bool(sweep_config.get("enabled", False)):
        return [{}]
    parameters = sweep_config.get("parameters", {}) or {}
    if not isinstance(parameters, dict):
        raise TypeError("sweep.parameters must be a dictionary")
    if not parameters:
        return [{}]
    keys = list(parameters)
    value_lists: list[list[Any]] = []
    for key in keys:
        values = parameters[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.parameters.{key} must be a non-empty list")
        value_lists.append(values)
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def condition_overrides(condition: dict) -> list[str]:
    lines = [
        "input_condition.enabled=true",
        f"input_condition.condition={format_override(condition['condition'])}",
        f"input_condition.feature={format_override(condition['feature'])}",
        f"input_condition.transform={format_override(condition['transform'])}",
        f"input_condition.strength={format_override(condition.get('strength', 0.0))}",
    ]
    for key in (
        "retention", "order", "diameter", "sigma_colour", "sigma_space",
        "sigma", "grid_size", "seed",
    ):
        if key in condition:
            lines.append(f"input_condition.{key}={format_override(condition[key])}")
    return lines
