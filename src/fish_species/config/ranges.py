"""Deterministic expansion helpers for compact numerical sweep ranges."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


class RangeExpansionError(ValueError):
    """A compact numerical range cannot be expanded deterministically."""


def _decimal(value: Any, path: str) -> Decimal:
    if isinstance(value, bool):
        raise RangeExpansionError(f"{path} must be a finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RangeExpansionError(f"{path} must be a finite number") from exc
    if not result.is_finite():
        raise RangeExpansionError(f"{path} must be a finite number")
    return result


def _prefers_float(*values: Any) -> bool:
    for value in values:
        if isinstance(value, (float, Decimal)):
            return True
        if isinstance(value, str) and any(marker in value.lower() for marker in (".", "e")):
            return True
    return False


def decimal_to_number(value: Decimal, *, prefer_float: bool = False) -> int | float:
    """Convert an exact range value to a normal YAML-compatible scalar."""
    if value.is_zero():
        value = abs(value)
    if not prefer_float and value == value.to_integral_value():
        return int(value)
    return float(value)


@dataclass(frozen=True)
class DecimalRange:
    """An inclusive, direction-aware range represented without float drift."""

    start: Decimal
    stop: Decimal
    step: Decimal
    prefer_float: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DecimalRange":
        if not isinstance(raw, Mapping):
            raise RangeExpansionError("range must be a mapping")
        missing = [key for key in ("start", "stop", "step") if key not in raw]
        if missing:
            raise RangeExpansionError(
                "range requires " + ", ".join(f"range.{key}" for key in missing)
            )
        start = _decimal(raw["start"], "range.start")
        stop = _decimal(raw["stop"], "range.stop")
        step = _decimal(raw["step"], "range.step")
        if step == 0:
            raise RangeExpansionError("range.step must not be zero")
        if start < stop and step < 0:
            raise RangeExpansionError("range.step must be positive when start < stop")
        if start > stop and step > 0:
            raise RangeExpansionError("range.step must be negative when start > stop")
        return cls(
            start=start,
            stop=stop,
            step=step,
            prefer_float=_prefers_float(raw["start"], raw["stop"], raw["step"]),
        )

    def decimals(self) -> tuple[Decimal, ...]:
        if self.start == self.stop:
            return (self.start,)

        values: list[Decimal] = []
        current = self.start
        if self.step > 0:
            while current <= self.stop:
                values.append(current)
                current += self.step
        else:
            while current >= self.stop:
                values.append(current)
                current += self.step

        # Compact ranges promise both endpoints, even when the interval is not
        # an exact multiple of the step. Never duplicate an already-hit stop.
        if not values or values[-1] != self.stop:
            values.append(self.stop)
        return tuple(values)

    def numbers(self) -> tuple[int | float, ...]:
        return tuple(
            decimal_to_number(value, prefer_float=self.prefer_float)
            for value in self.decimals()
        )


def expand_decimal_range(raw: Mapping[str, Any]) -> tuple[Decimal, ...]:
    """Expand ``start``/``stop``/``step`` inclusively using :class:`Decimal`."""
    return DecimalRange.from_mapping(raw).decimals()


def _template_value(value: Decimal) -> int | Decimal:
    if value == value.to_integral_value():
        return int(value)
    return value


def format_range_name(
    template: str,
    *,
    parameter: str,
    value: Decimal,
    index: int,
) -> str:
    """Format a deterministic condition name from an exact range value.

    Templates may use ``value``, the configured parameter name, ``percent``
    (the value multiplied by 100), and ``index``. Integral Decimal values are
    exposed as integers so formats such as ``{percent:03d}`` are reliable.
    """
    if not isinstance(template, str) or not template.strip():
        raise RangeExpansionError("name_template must be a non-empty string")
    context = {
        "value": _template_value(value),
        "percent": _template_value(value * Decimal(100)),
        "index": index,
        parameter: _template_value(value),
    }
    try:
        name = template.format_map(context)
    except (KeyError, ValueError, TypeError) as exc:
        raise RangeExpansionError(
            f"Could not format name_template {template!r}: {exc}"
        ) from exc
    if not name.strip():
        raise RangeExpansionError("name_template produced an empty condition name")
    return name


__all__ = [
    "DecimalRange",
    "RangeExpansionError",
    "decimal_to_number",
    "expand_decimal_range",
    "format_range_name",
]
