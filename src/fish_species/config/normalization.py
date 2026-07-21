"""Pure normalization of the canonical experiment sweep representation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .ranges import (
    DecimalRange,
    RangeExpansionError,
    decimal_to_number,
    format_range_name,
)


class ConfigNormalizationError(ValueError):
    """Canonical configuration syntax cannot be normalized safely."""


@dataclass(frozen=True)
class CompatibilityWarning:
    """One accepted legacy path and the canonical path replacing it."""

    path: str
    canonical_path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "canonical_path": self.canonical_path,
            "message": self.message,
        }


@dataclass(frozen=True)
class NormalizationResult:
    config: dict[str, Any]
    warnings: tuple[CompatibilityWarning, ...] = ()


def _require_text(raw: Mapping[str, Any], key: str, path: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigNormalizationError(f"{path}.{key} must be a non-empty string")
    return value


def _explicit_condition(raw: Mapping[str, Any], path: str) -> dict[str, Any]:
    name = _require_text(raw, "name", path)
    transform = _require_text(raw, "transform", path)
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ConfigNormalizationError(f"{path}.parameters must be a mapping")
    condition = copy.deepcopy(dict(raw))
    condition["name"] = name
    condition["transform"] = transform
    condition["parameters"] = copy.deepcopy(dict(parameters))
    return condition


def _range_conditions(raw: Mapping[str, Any], path: str) -> list[dict[str, Any]]:
    template = _require_text(raw, "name_template", path)
    transform = _require_text(raw, "transform", path)
    parameter = _require_text(raw, "parameter", path)
    range_raw = raw.get("range")
    if not isinstance(range_raw, Mapping):
        raise ConfigNormalizationError(f"{path}.range must be a mapping")
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ConfigNormalizationError(f"{path}.parameters must be a mapping")
    if parameter in parameters:
        raise ConfigNormalizationError(
            f"{path}.parameters.{parameter} duplicates the ranged parameter"
        )
    try:
        number_range = DecimalRange.from_mapping(range_raw)
    except RangeExpansionError as exc:
        raise ConfigNormalizationError(f"{path}.{exc}") from exc

    metadata = {
        key: copy.deepcopy(value)
        for key, value in raw.items()
        if key not in {"name_template", "parameter", "range", "parameters"}
    }
    conditions: list[dict[str, Any]] = []
    for index, value in enumerate(number_range.decimals()):
        try:
            name = format_range_name(
                template,
                parameter=parameter,
                value=value,
                index=index,
            )
        except RangeExpansionError as exc:
            raise ConfigNormalizationError(f"{path}.{exc}") from exc
        condition = copy.deepcopy(metadata)
        condition["name"] = name
        condition["transform"] = transform
        condition["parameters"] = copy.deepcopy(dict(parameters))
        condition["parameters"][parameter] = decimal_to_number(
            value,
            prefer_float=number_range.prefer_float,
        )
        if (
            "strength" not in condition
            and transform == "saturation"
            and parameter == "retention"
        ):
            condition["strength"] = decimal_to_number(
                1 - value,
                prefer_float=True,
            )
        conditions.append(condition)
    return conditions


def normalize_conditions(raw_conditions: Any) -> list[dict[str, Any]]:
    """Return complete, atomic condition objects in configured order."""
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ConfigNormalizationError("sweep.conditions must be a non-empty list")
    conditions: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_conditions):
        path = f"sweep.conditions[{index}]"
        if not isinstance(raw, Mapping):
            raise ConfigNormalizationError(f"{path} must be a mapping")
        has_name = "name" in raw
        has_range = "name_template" in raw or "range" in raw or "parameter" in raw
        if has_name and has_range:
            raise ConfigNormalizationError(
                f"{path} must be either an explicit condition or a range, not both"
            )
        if has_name:
            conditions.append(_explicit_condition(raw, path))
        elif has_range:
            conditions.extend(_range_conditions(raw, path))
        else:
            raise ConfigNormalizationError(
                f"{path} requires either name or name_template/range/parameter"
            )

    first_index: dict[str, int] = {}
    for index, condition in enumerate(conditions):
        name = condition["name"]
        if name in first_index:
            raise ConfigNormalizationError(
                "sweep.conditions contains duplicate condition identifier "
                f"{name!r} at indices {first_index[name]} and {index}"
            )
        first_index[name] = index
    return conditions


def normalize_condition_references(raw_conditions: Any) -> list[Any]:
    """Expand complete/range objects while preserving named references."""
    if not isinstance(raw_conditions, list):
        raise ConfigNormalizationError("conditions must be a list")
    resolved: list[Any] = []
    for item in raw_conditions:
        if isinstance(item, str):
            resolved.append(item)
        else:
            resolved.extend(normalize_conditions([item]))
    identifiers = [
        item if isinstance(item, str) else item["name"] for item in resolved
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ConfigNormalizationError("conditions contains duplicate identifiers")
    return resolved


def _warning(path: str, canonical_path: str) -> CompatibilityWarning:
    return CompatibilityWarning(
        path=path,
        canonical_path=canonical_path,
        message=f"{path} is deprecated; use {canonical_path}",
    )


def _condition(
    name: str,
    transform: str,
    *,
    feature: str,
    strength: int | float,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "feature": feature,
        "transform": transform,
        "strength": strength,
        "parameters": copy.deepcopy(dict(parameters or {})),
    }


def _legacy_sequence(start: Any, stop: Any, step: Any) -> list[float]:
    start_number = float(start)
    stop_number = float(stop)
    step_number = abs(float(step))
    if step_number == 0:
        raise ConfigNormalizationError("legacy saturation step must be greater than zero")
    signed_step = -step_number if start_number > stop_number else step_number
    number_range = DecimalRange.from_mapping({
        "start": start,
        "stop": stop,
        "step": signed_step,
    })
    return [float(value) for value in number_range.decimals()]


def _legacy_cue_catalogue(config: Mapping[str, Any], cue: Mapping[str, Any]) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    saturation = cue.get("saturation", {}) or {}
    if isinstance(saturation, Mapping) and bool(saturation.get("enabled", True)):
        values = saturation.get("values")
        if values is None:
            values = _legacy_sequence(
                saturation.get("start", 1.0),
                saturation.get("stop", 0.0),
                saturation.get("step", 0.01),
            )
        for raw_retention in values:
            retention = float(raw_retention)
            percentage = int(round(retention * 100))
            conditions.append(_condition(
                f"saturation_{percentage:03d}pct",
                "saturation",
                feature="colour",
                strength=round(1.0 - retention, 10),
                parameters={"retention": retention},
            ))

    grayscale = cue.get("grayscale", {}) or {}
    if isinstance(grayscale, Mapping) and bool(grayscale.get("enabled", True)):
        conditions.append(_condition(
            "grayscale", "grayscale", feature="colour", strength=1.0
        ))

    channel = cue.get("channel_shuffle", {}) or {}
    if isinstance(channel, Mapping) and bool(channel.get("enabled", True)):
        for raw_order in channel.get("orders", [[2, 0, 1]]):
            order = [int(index) for index in raw_order]
            conditions.append(_condition(
                "channel_shuffle_" + "".join(str(index) for index in order),
                "channel_shuffle",
                feature="colour",
                strength=1.0,
                parameters={"order": order},
            ))

    bilateral = cue.get("bilateral_filter", {}) or {}
    if isinstance(bilateral, Mapping) and bool(bilateral.get("enabled", True)):
        settings = bilateral.get("settings", [
            {"diameter": 5, "sigma_colour": 25, "sigma_space": 25},
            {"diameter": 7, "sigma_colour": 50, "sigma_space": 50},
            {"diameter": 9, "sigma_colour": 100, "sigma_space": 100},
        ])
        for setting in settings:
            diameter = int(setting["diameter"])
            sigma_colour = float(setting["sigma_colour"])
            sigma_space = float(setting["sigma_space"])
            conditions.append(_condition(
                f"bilateral_d{diameter}_c{sigma_colour:g}_s{sigma_space:g}",
                "bilateral_filter",
                feature="texture",
                strength=sigma_colour,
                parameters={
                    "diameter": diameter,
                    "sigma_colour": sigma_colour,
                    "sigma_space": sigma_space,
                },
            ))

    gaussian = cue.get("gaussian_blur", {}) or {}
    if isinstance(gaussian, Mapping) and bool(gaussian.get("enabled", True)):
        for raw_sigma in gaussian.get("sigmas", [0.5, 1.0, 2.0, 4.0]):
            sigma = float(raw_sigma)
            conditions.append(_condition(
                f"gaussian_sigma_{sigma:g}",
                "gaussian_blur",
                feature="texture",
                strength=sigma,
                parameters={"sigma": sigma},
            ))

    patch = cue.get("patch_shuffle", {}) or {}
    if isinstance(patch, Mapping) and bool(patch.get("enabled", True)):
        seed = int(patch.get("seed", config.get("seed", 0)))
        for raw_grid in patch.get("grid_sizes", [2, 4, 8]):
            grid = int(raw_grid)
            conditions.append(_condition(
                f"patch_shuffle_grid_{grid}",
                "patch_shuffle",
                feature="shape",
                strength=grid,
                parameters={"grid_size": grid, "seed": seed},
            ))

    names = [condition["name"] for condition in conditions]
    if len(names) != len(set(names)):
        raise ConfigNormalizationError("legacy cue catalogue contains duplicate names")
    requested = cue.get("condition_names")
    if requested is not None:
        requested_set = set(requested)
        unknown = requested_set - set(names)
        if unknown:
            raise ConfigNormalizationError(
                f"Unknown legacy cue condition names: {sorted(unknown)}"
            )
        conditions = [item for item in conditions if item["name"] in requested_set]
    return conditions


def _normalize_input_condition(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigNormalizationError("input_condition must be a mapping")
    condition = copy.deepcopy(dict(raw))
    if "name" not in condition and "condition" in condition:
        condition["name"] = condition["condition"]
    condition.pop("condition", None)
    parameters = condition.get("parameters", {}) or {}
    if not isinstance(parameters, Mapping):
        raise ConfigNormalizationError("input_condition.parameters must be a mapping")
    canonical_parameters = copy.deepcopy(dict(parameters))
    for key in (
        "retention", "order", "diameter", "sigma_colour", "sigma_space",
        "sigma", "grid_size", "seed",
    ):
        if key in condition and key not in canonical_parameters:
            canonical_parameters[key] = condition[key]
        condition.pop(key, None)
    condition["parameters"] = canonical_parameters
    return condition


def _set_if_absent(mapping: dict[str, Any], key: str, value: Any) -> None:
    if key not in mapping:
        mapping[key] = copy.deepcopy(value)


def _normalize_legacy_aliases(config: dict[str, Any]) -> NormalizationResult:
    normalized = copy.deepcopy(config)
    warnings: list[CompatibilityWarning] = []

    data = normalized.get("data")
    if isinstance(data, dict) and "image_size" in data:
        warnings.append(_warning("data.image_size", "preprocessing.image_size"))
        preprocessing = normalized.setdefault("preprocessing", {})
        if not isinstance(preprocessing, dict):
            raise ConfigNormalizationError("preprocessing must be a mapping")
        _set_if_absent(preprocessing, "image_size", data["image_size"])
        data.pop("image_size")

    raw_input = normalized.get("input_condition")
    if raw_input is not None:
        if isinstance(raw_input, Mapping) and any(
            key in raw_input
            for key in ("condition", "retention", "order", "diameter", "sigma_colour", "sigma_space", "sigma", "grid_size", "seed")
        ):
            warnings.append(_warning("input_condition.*", "input_condition.parameters.*"))
        normalized["input_condition"] = _normalize_input_condition(raw_input)

    sweep = normalized.get("sweep", {}) or {}
    if not isinstance(sweep, dict):
        raise ConfigNormalizationError("sweep must be a mapping")
    parameters = sweep.get("parameters", {}) or {}
    if not isinstance(parameters, dict):
        raise ConfigNormalizationError("sweep.parameters must be a mapping")
    remapped_parameters: dict[str, Any] = {}
    for key, values in parameters.items():
        if key == "data.image_size":
            warnings.append(_warning(key, "preprocessing.image_size"))
            remapped_parameters["preprocessing.image_size"] = copy.deepcopy(values)
        else:
            remapped_parameters[key] = copy.deepcopy(values)
    if parameters:
        sweep["parameters"] = remapped_parameters

    cue = normalized.pop("test_cue_suppression", None)
    if cue is not None:
        warnings.append(_warning("test_cue_suppression.*", "evaluation.test_conditions"))
    cue_mapping = cue if isinstance(cue, Mapping) else {}
    catalogue = _legacy_cue_catalogue(normalized, cue_mapping) if cue is not None else []

    matched = normalized.pop("matched_condition_training", None)
    if matched is not None:
        warnings.append(_warning("matched_condition_training.*", "sweep.conditions"))
    if isinstance(matched, Mapping) and bool(matched.get("enabled", False)) and "conditions" not in sweep:
        conditions = copy.deepcopy(catalogue)
        include_original = bool(matched.get("include_original", True))
        deduplicate = bool(matched.get("deduplicate_equivalent_conditions", True))
        if deduplicate:
            conditions = [
                item for item in conditions
                if not (
                    item["transform"] == "saturation"
                    and math.isclose(float(item["parameters"]["retention"]), 1.0, abs_tol=1e-12)
                )
                and not (
                    item["transform"] == "saturation"
                    and any(c["transform"] == "grayscale" for c in conditions)
                    and math.isclose(float(item["parameters"]["retention"]), 0.0, abs_tol=1e-12)
                )
            ]
        if include_original:
            conditions.insert(0, _condition(
                "original", "original", feature="baseline", strength=0.0
            ))
        requested = matched.get("condition_names")
        if requested:
            requested_set = set(requested)
            available = {item["name"] for item in conditions}
            unknown = requested_set - available
            if unknown:
                raise ConfigNormalizationError(
                    f"Unknown matched training condition names: {sorted(unknown)}"
                )
            conditions = [item for item in conditions if item["name"] in requested_set]
        sweep["conditions"] = conditions
        sweep["enabled"] = True

    if sweep or "sweep" in normalized:
        normalized["sweep"] = sweep

    evaluation = normalized.get("evaluation", {}) or {}
    if not isinstance(evaluation, dict):
        raise ConfigNormalizationError("evaluation must be a mapping")
    if cue is not None:
        legacy_enabled = bool(cue_mapping.get("enabled", False))
        legacy_evaluate_original = bool(
            isinstance(matched, Mapping)
            and matched.get("enabled", False)
            and matched.get(
                "evaluate_original_model_on_all_test_conditions", True
            )
        )
        existing_test = evaluation.get("test_conditions")
        if not isinstance(existing_test, dict):
            existing_test = {}
        if "test_conditions" not in evaluation or (
            not existing_test.get("conditions")
            and (legacy_enabled or legacy_evaluate_original)
        ):
            existing_test = {
                "enabled": legacy_enabled,
                "conditions": copy.deepcopy(catalogue),
                "evaluate_original_training": legacy_evaluate_original,
            }
            evaluation["test_conditions"] = existing_test
        else:
            if legacy_enabled:
                existing_test["enabled"] = True
            if legacy_evaluate_original:
                existing_test["evaluate_original_training"] = True

    matrix = normalized.pop("condition_matrix_evaluation", None)
    if matrix is not None:
        warnings.append(_warning("condition_matrix_evaluation.*", "evaluation.condition_matrix"))
    existing_matrix = evaluation.get("condition_matrix")
    legacy_matrix_enabled = bool(
        isinstance(matrix, Mapping) and matrix.get("enabled", False)
    )
    if isinstance(matrix, Mapping) and (
        "condition_matrix" not in evaluation
        or (
            isinstance(existing_matrix, dict)
            and not existing_matrix.get("conditions")
            and legacy_matrix_enabled
        )
    ):
        requested = matrix.get("condition_names", [])
        by_name = {
            item["name"]: item
            for item in [
                _condition("original", "original", feature="baseline", strength=0.0),
                *(_legacy_cue_catalogue(normalized, {**cue_mapping, "condition_names": None}) if cue is not None else []),
            ]
        }
        evaluation["condition_matrix"] = {
            "enabled": bool(matrix.get("enabled", False)),
            "conditions": [copy.deepcopy(by_name.get(name, name)) for name in requested],
            "write_reports": bool(matrix.get("write_reports", True)),
        }
    if evaluation or "evaluation" in normalized:
        normalized["evaluation"] = evaluation
    return NormalizationResult(normalized, tuple(warnings))


def normalize_config_with_report(config: dict[str, Any]) -> NormalizationResult:
    """Deep-copy and normalize canonical sweep ranges into condition objects.

    The operation is non-mutating and idempotent. It deliberately leaves
    evaluation sections untouched because evaluation schedules do not create
    model fits.
    """
    if not isinstance(config, dict):
        raise ConfigNormalizationError("config must be a mapping")
    result = _normalize_legacy_aliases(config)
    normalized = result.config
    sweep = normalized.get("sweep")
    if sweep is None:
        return NormalizationResult(normalized, result.warnings)
    if not isinstance(sweep, dict):
        raise ConfigNormalizationError("sweep must be a mapping")
    if "conditions" in sweep:
        sweep["conditions"] = normalize_conditions(sweep["conditions"])
    evaluation = normalized.get("evaluation", {}) or {}
    if isinstance(evaluation, dict):
        for section_name in ("test_conditions", "condition_matrix"):
            section = evaluation.get(section_name)
            if isinstance(section, dict) and "conditions" in section:
                section["conditions"] = normalize_condition_references(
                    section["conditions"]
                )
    return NormalizationResult(normalized, result.warnings)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only the canonical configuration from normalization."""
    return normalize_config_with_report(config).config


# British spelling is convenient beside the public ``normalisation`` config
# key while the module and primary API retain the project's Python spelling.
normalise_config = normalize_config


__all__ = [
    "ConfigNormalizationError",
    "CompatibilityWarning",
    "NormalizationResult",
    "normalise_config",
    "normalize_conditions",
    "normalize_condition_references",
    "normalize_config",
    "normalize_config_with_report",
]
