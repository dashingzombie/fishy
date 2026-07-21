"""Configuration loading, dotted overrides, and sweep expansion."""

from .loading import ConfigLoadError, deep_merge, load_config
from .normalization import (
    CompatibilityWarning,
    ConfigNormalizationError,
    NormalizationResult,
    normalize_condition_references,
    normalize_conditions,
    normalize_config,
    normalize_config_with_report,
)
from .overrides import apply_overrides, parse_scalar, set_nested
from .sweeps import (
    generate_sweep_configs,
    get_sweep_parameters_from_cli,
    get_sweep_parameters_from_config,
    parse_sweep_item,
)
from .schema import CONFIG_FIELDS, ConfigField, field_for_path, is_known_config_path
from .validation import (
    ConfigValidationError,
    ValidationIssue,
    resolve_workflow,
    validate_config,
    validate_override_items,
)

__all__ = [
    "apply_overrides",
    "CONFIG_FIELDS",
    "ConfigField",
    "ConfigLoadError",
    "ConfigNormalizationError",
    "ConfigValidationError",
    "CompatibilityWarning",
    "deep_merge",
    "generate_sweep_configs",
    "get_sweep_parameters_from_cli",
    "get_sweep_parameters_from_config",
    "load_config",
    "NormalizationResult",
    "normalize_condition_references",
    "normalize_conditions",
    "normalize_config",
    "normalize_config_with_report",
    "parse_scalar",
    "parse_sweep_item",
    "set_nested",
    "ValidationIssue",
    "field_for_path",
    "is_known_config_path",
    "resolve_workflow",
    "validate_config",
    "validate_override_items",
]
