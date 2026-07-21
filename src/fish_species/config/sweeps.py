from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any

from .normalization import normalize_conditions
from .overrides import parse_scalar, set_nested


@dataclass
class SweepItem:
    """One externally resolvable training fit from the canonical sweep."""

    index: int
    assignments: tuple[tuple[str, Any], ...]
    condition: dict[str, Any] | None = None

    @property
    def parameter_values(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.assignments))


def parse_sweep_item(item: str) -> tuple[str, list[Any]]:
    """Parse ``key=v1,v2`` using the legacy scalar rules."""
    if "=" not in item:
        raise ValueError(f"Sweep item must look like key=v1,v2. Got: {item}")

    key, values = item.split("=", 1)
    parsed = [parse_scalar(value) for value in values.split(",") if value.strip()]
    if len(parsed) == 0:
        raise ValueError(f"No values supplied for sweep key: {key}")
    return key, parsed


def get_sweep_parameters_from_config(config: dict[str, Any]) -> dict[str, list[Any]]:
    """Return the ordinary configured sweep without expanding it."""
    sweep_config = config.get("sweep", {})
    if not sweep_config.get("enabled", False):
        return {}
    parameters = sweep_config.get("parameters", {})
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise ValueError("sweep.parameters must be a dictionary.")
    return parameters


def get_sweep_conditions_from_config(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return complete canonical conditions without creating a product within them."""
    sweep_config = config.get("sweep", {})
    if not sweep_config.get("enabled", False) or "conditions" not in sweep_config:
        return []
    return normalize_conditions(sweep_config["conditions"])


def get_sweep_parameters_from_cli(items: list[str]) -> dict[str, list[Any]]:
    parameters: dict[str, list[Any]] = {}
    for item in items:
        key, values = parse_sweep_item(item)
        parameters[key] = values
    return parameters


def expand_sweep_items(
    base_config: dict[str, Any],
    cli_sweep_items: list[str] | None = None,
) -> list[SweepItem]:
    """Expand one canonical training layer in deterministic configured order.

    Ordinary dotted parameters form a Cartesian product. Complete condition
    objects are an additional single dimension and remain atomic. Sections
    outside ``sweep`` -- notably ``evaluation`` -- never add training fits.
    """
    cli_sweep_items = cli_sweep_items or []
    if cli_sweep_items:
        parameters = get_sweep_parameters_from_cli(cli_sweep_items)
    else:
        parameters = get_sweep_parameters_from_config(base_config)

    conditions = get_sweep_conditions_from_config(base_config)
    keys = list(parameters)
    parameter_products = (
        itertools.product(*(parameters[key] for key in keys))
        if keys
        else [()]
    )
    condition_values: list[dict[str, Any] | None] = conditions or [None]
    items: list[SweepItem] = []
    for combination in parameter_products:
        assignments = tuple(
            (key, copy.deepcopy(value))
            for key, value in zip(keys, combination)
        )
        for condition in condition_values:
            items.append(
                SweepItem(
                    index=len(items),
                    assignments=assignments,
                    condition=copy.deepcopy(condition),
                )
            )
    return items


def apply_sweep_item(
    base_config: dict[str, Any],
    item: SweepItem,
    *,
    disable_sweep: bool = False,
) -> dict[str, Any]:
    """Apply one sweep item to a deep copy of ``base_config``."""
    config = copy.deepcopy(base_config)
    for key, value in item.assignments:
        set_nested(config, key, copy.deepcopy(value))
    if item.condition is not None:
        input_condition = copy.deepcopy(item.condition)
        input_condition["enabled"] = True
        config["input_condition"] = input_condition
    if disable_sweep:
        sweep = config.setdefault("sweep", {})
        if not isinstance(sweep, dict):
            raise TypeError("sweep must be a dictionary")
        sweep["enabled"] = False
    return config


def generate_sweep_configs(
    base_config: dict[str, Any],
    cli_sweep_items: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Expand exactly one sweep layer into independent deep-copied configs."""
    items = expand_sweep_items(
        base_config,
        cli_sweep_items,
    )
    if (
        len(items) == 1
        and not items[0].assignments
        and items[0].condition is None
    ):
        return [base_config]
    return [apply_sweep_item(base_config, item) for item in items]
