from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from .normalization import ConfigNormalizationError, normalize_conditions
from .schema import CONFIG_FIELDS, MISSING_DEFAULT, field_for_path, is_known_config_path


Workflow = Literal["auto", "training", "run_specs", "saved"]
KNOWN_TRANSFORMS = frozenset({
    "original",
    "saturation",
    "grayscale",
    "channel_shuffle",
    "bilateral_filter",
    "gaussian_blur",
    "patch_shuffle",
})
_ABSENT = object()


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class ConfigValidationError(ValueError):
    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("ConfigValidationError requires at least one issue")
        super().__init__("Invalid configuration:\n" + "\n".join(
            f"  - {issue}" for issue in self.issues
        ))


def _get(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _ABSENT
        value = value[part]
    return value


def _is_type(value: Any, expected: tuple[type, ...]) -> bool:
    if value is None:
        return type(None) in expected
    if isinstance(value, bool):
        return bool in expected
    numeric = tuple(item for item in expected if item in {int, float})
    if numeric and isinstance(value, (int, float)):
        return True
    return isinstance(value, expected)


def _type_names(expected: tuple[type, ...]) -> str:
    return " or ".join(item.__name__ for item in expected)


def _number(
    issues: list[ValidationIssue],
    path: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(ValidationIssue(path, "must be a number"))
        return False
    numeric = float(value)
    if not math.isfinite(numeric):
        issues.append(ValidationIssue(path, "must be finite"))
        return False
    if minimum is not None:
        invalid = numeric <= minimum if exclusive_minimum else numeric < minimum
        if invalid:
            operator = ">" if exclusive_minimum else ">="
            issues.append(ValidationIssue(path, f"must be {operator} {minimum:g}"))
            return False
    if maximum is not None and numeric > maximum:
        issues.append(ValidationIssue(path, f"must be <= {maximum:g}"))
        return False
    return True


def resolve_workflow(config: dict[str, Any], workflow: Workflow = "auto") -> str:
    if workflow != "auto":
        return workflow
    if bool(_get(config, "input_condition.enabled") is True):
        return "training"
    if bool(_get(config, "matched_condition_training.enabled") is True):
        return "run_specs"
    return "training"


def validate_override_items(items: Iterable[str]) -> tuple[str, ...]:
    """Validate dotted override keys without changing legacy value parsing."""
    issues: list[ValidationIssue] = []
    keys: list[str] = []
    for item in items:
        if "=" not in item:
            issues.append(ValidationIssue("override", f"must look like key=value, got {item!r}"))
            continue
        raw_key, _ = item.split("=", 1)
        key = raw_key.strip()
        if not key:
            issues.append(ValidationIssue("override", "key must not be empty"))
            continue
        if not is_known_config_path(key):
            issues.append(ValidationIssue(key, "unknown configuration override path"))
            continue
        keys.append(key)
    if issues:
        raise ConfigValidationError(issues)
    return tuple(keys)


def _validate_channel_order(
    issues: list[ValidationIssue], path: str, order: Any
) -> tuple[int, int, int] | None:
    if isinstance(order, str):
        try:
            order = [int(item.strip()) for item in order.split(",")]
        except ValueError:
            issues.append(ValidationIssue(path, "must be a permutation of 0,1,2"))
            return None
    if not isinstance(order, (list, tuple)):
        issues.append(ValidationIssue(path, "must be a list, tuple, or comma-separated string"))
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in order):
        issues.append(ValidationIssue(path, "must contain integer channel indices"))
        return None
    if sorted(order) != [0, 1, 2]:
        issues.append(ValidationIssue(path, f"must be a permutation of [0, 1, 2], got {list(order)!r}"))
        return None
    return tuple(order)


def _validate_unique_values(
    issues: list[ValidationIssue],
    path: str,
    indexed_values: list[tuple[int, Any]],
) -> None:
    first_index: dict[Any, int] = {}
    for index, value in indexed_values:
        if value in first_index:
            issues.append(ValidationIssue(
                path,
                "must contain unique values; "
                f"indices {first_index[value]} and {index} both resolve to {value!r}",
            ))
        else:
            first_index[value] = index


def _validate_test_condition_names(
    config: dict[str, Any],
    cue: dict[str, Any],
    issues: list[ValidationIssue],
    *,
    catalogue_parameters_valid: bool,
) -> None:
    requested = cue.get("condition_names", _ABSENT)
    if requested is _ABSENT:
        return
    path = "test_cue_suppression.condition_names"
    if not isinstance(requested, list) or not requested:
        issues.append(ValidationIssue(path, "must be a non-empty list"))
        return

    valid_names: list[str] = []
    for index, name in enumerate(requested):
        if not isinstance(name, str) or not name.strip():
            issues.append(ValidationIssue(
                f"{path}[{index}]", "must be a non-empty condition-name string"
            ))
        else:
            valid_names.append(name)
    duplicates = sorted({
        name for name in valid_names if valid_names.count(name) > 1
    })
    if duplicates:
        issues.append(ValidationIssue(path, f"contains duplicate names: {duplicates}"))
    if len(valid_names) != len(requested) or duplicates or not catalogue_parameters_valid:
        return

    # Validate the allow-list against the complete enabled catalogue even when
    # fixed-RGB evaluation is currently disabled. This keeps a preconfigured
    # allow-list safe to enable later without conflating it with matched training.
    catalogue_config = dict(config)
    catalogue_cue = dict(cue)
    catalogue_cue["enabled"] = True
    catalogue_config["test_cue_suppression"] = catalogue_cue
    try:
        from ..evaluation.cue_suppression import generate_test_cue_conditions

        generate_test_cue_conditions(catalogue_config)
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(ValidationIssue(path, str(exc)))


def _validate_transform_parameters(
    config: dict[str, Any], issues: list[ValidationIssue]
) -> None:
    cue = config.get("test_cue_suppression", {}) or {}
    if not isinstance(cue, dict):
        return
    initial_issue_count = len(issues)

    saturation = cue.get("saturation", {}) or {}
    if isinstance(saturation, dict):
        values = saturation.get("values", _ABSENT)
        if values is not _ABSENT:
            if not isinstance(values, list) or not values:
                issues.append(ValidationIssue(
                    "test_cue_suppression.saturation.values", "must be a non-empty list"
                ))
            else:
                valid_values = []
                for index, value in enumerate(values):
                    if _number(
                        issues,
                        f"test_cue_suppression.saturation.values[{index}]",
                        value,
                        minimum=0,
                        maximum=1,
                    ):
                        valid_values.append((index, float(value)))
                _validate_unique_values(
                    issues,
                    "test_cue_suppression.saturation.values",
                    valid_values,
                )
        for key, default in (("start", 1.0), ("stop", 0.0)):
            _number(
                issues,
                f"test_cue_suppression.saturation.{key}",
                saturation.get(key, default),
                minimum=0,
                maximum=1,
            )
        _number(
            issues,
            "test_cue_suppression.saturation.step",
            saturation.get("step", 0.01),
            minimum=0,
            exclusive_minimum=True,
        )

    channel = cue.get("channel_shuffle", {}) or {}
    if isinstance(channel, dict):
        orders = channel.get("orders", [[2, 0, 1]])
        if not isinstance(orders, list) or not orders:
            issues.append(ValidationIssue(
                "test_cue_suppression.channel_shuffle.orders", "must be a non-empty list"
            ))
        else:
            valid_orders = []
            for index, order in enumerate(orders):
                validated = _validate_channel_order(
                    issues,
                    f"test_cue_suppression.channel_shuffle.orders[{index}]",
                    order,
                )
                if validated is not None:
                    valid_orders.append((index, validated))
            _validate_unique_values(
                issues,
                "test_cue_suppression.channel_shuffle.orders",
                valid_orders,
            )

    gaussian = cue.get("gaussian_blur", {}) or {}
    if isinstance(gaussian, dict):
        sigmas = gaussian.get("sigmas", [0.5, 1.0, 2.0, 4.0])
        if not isinstance(sigmas, list) or not sigmas:
            issues.append(ValidationIssue(
                "test_cue_suppression.gaussian_blur.sigmas", "must be a non-empty list"
            ))
        else:
            valid_sigmas = []
            for index, sigma in enumerate(sigmas):
                if _number(
                    issues,
                    f"test_cue_suppression.gaussian_blur.sigmas[{index}]",
                    sigma,
                    minimum=0,
                    exclusive_minimum=True,
                ):
                    valid_sigmas.append((index, float(sigma)))
            _validate_unique_values(
                issues,
                "test_cue_suppression.gaussian_blur.sigmas",
                valid_sigmas,
            )

    bilateral = cue.get("bilateral_filter", {}) or {}
    if isinstance(bilateral, dict):
        settings = bilateral.get("settings", [
            {"diameter": 5, "sigma_colour": 25, "sigma_space": 25},
            {"diameter": 7, "sigma_colour": 50, "sigma_space": 50},
            {"diameter": 9, "sigma_colour": 100, "sigma_space": 100},
        ])
        if not isinstance(settings, list) or not settings:
            issues.append(ValidationIssue(
                "test_cue_suppression.bilateral_filter.settings", "must be a non-empty list"
            ))
        else:
            valid_settings = []
            for index, setting in enumerate(settings):
                path = f"test_cue_suppression.bilateral_filter.settings[{index}]"
                if not isinstance(setting, dict):
                    issues.append(ValidationIssue(path, "must be a mapping"))
                    continue
                valid_setting = True
                diameter = setting.get("diameter", _ABSENT)
                if isinstance(diameter, bool) or not isinstance(diameter, int):
                    issues.append(ValidationIssue(f"{path}.diameter", "must be an integer"))
                    valid_setting = False
                elif diameter <= 0 or diameter % 2 == 0:
                    issues.append(ValidationIssue(f"{path}.diameter", "must be a positive odd integer"))
                    valid_setting = False
                sigma_values = []
                for key in ("sigma_colour", "sigma_space"):
                    if key not in setting:
                        issues.append(ValidationIssue(f"{path}.{key}", "is required"))
                        valid_setting = False
                    else:
                        valid_number = _number(
                            issues,
                            f"{path}.{key}",
                            setting[key],
                            minimum=0,
                            exclusive_minimum=True,
                        )
                        valid_setting = valid_setting and valid_number
                        if valid_number:
                            sigma_values.append(float(setting[key]))
                if valid_setting:
                    valid_settings.append(
                        (index, (diameter, sigma_values[0], sigma_values[1]))
                    )
            _validate_unique_values(
                issues,
                "test_cue_suppression.bilateral_filter.settings",
                valid_settings,
            )

    patch = cue.get("patch_shuffle", {}) or {}
    if isinstance(patch, dict):
        grids = patch.get("grid_sizes", [2, 4, 8])
        image_size = _get(config, "data.image_size")
        if not isinstance(grids, list) or not grids:
            issues.append(ValidationIssue(
                "test_cue_suppression.patch_shuffle.grid_sizes", "must be a non-empty list"
            ))
        else:
            valid_grids = []
            for index, grid in enumerate(grids):
                path = f"test_cue_suppression.patch_shuffle.grid_sizes[{index}]"
                if isinstance(grid, bool) or not isinstance(grid, int):
                    issues.append(ValidationIssue(path, "must be an integer"))
                elif grid < 2:
                    issues.append(ValidationIssue(path, "must be >= 2"))
                elif isinstance(image_size, int) and image_size % grid != 0:
                    issues.append(ValidationIssue(path, f"must divide data.image_size={image_size}"))
                else:
                    valid_grids.append((index, grid))
            _validate_unique_values(
                issues,
                "test_cue_suppression.patch_shuffle.grid_sizes",
                valid_grids,
            )

    _validate_test_condition_names(
        config,
        cue,
        issues,
        catalogue_parameters_valid=len(issues) == initial_issue_count,
    )

    raw = config.get("input_condition", {}) or {}
    if isinstance(raw, dict):
        _validate_condition_object(
            config,
            raw,
            "input_condition",
            issues,
            enabled=bool(raw.get("enabled", False)),
        )


def _validate_condition_object(
    config: dict[str, Any],
    raw: dict[str, Any],
    path: str,
    issues: list[ValidationIssue],
    *,
    enabled: bool,
) -> None:
    transform = str(raw.get("transform", "original")).lower()
    if not enabled:
        if transform != "original":
            issues.append(ValidationIssue(
                f"{path}.transform",
                f"must be original when {path}.enabled=false",
            ))
        return
    if transform not in KNOWN_TRANSFORMS:
        issues.append(ValidationIssue(
            f"{path}.transform",
            f"unknown transformation {transform!r}; expected one of {sorted(KNOWN_TRANSFORMS)!r}",
        ))
        return
    parameters = raw.get("parameters", {}) or {}
    if not isinstance(parameters, dict):
        issues.append(ValidationIssue(f"{path}.parameters", "must be a mapping"))
        parameters = {}

    def parameter(key: str, default: Any = _ABSENT) -> tuple[Any, str]:
        if key in parameters:
            return parameters[key], f"{path}.parameters.{key}"
        if key in raw:
            return raw[key], f"{path}.{key}"
        return default, f"{path}.parameters.{key}"

    if transform == "saturation":
        retention, retention_path = parameter("retention")
        if retention is _ABSENT:
            issues.append(ValidationIssue(retention_path, "is required for saturation"))
        else:
            _number(issues, retention_path, retention, minimum=0, maximum=1)
    elif transform == "channel_shuffle":
        order, order_path = parameter("order", [2, 0, 1])
        _validate_channel_order(issues, order_path, order)
    elif transform == "gaussian_blur":
        sigma, sigma_path = parameter("sigma")
        if sigma is _ABSENT:
            issues.append(ValidationIssue(sigma_path, "is required for gaussian_blur"))
        else:
            _number(issues, sigma_path, sigma, minimum=0, exclusive_minimum=True)
    elif transform == "bilateral_filter":
        resolved: dict[str, tuple[Any, str]] = {}
        for key in ("diameter", "sigma_colour", "sigma_space"):
            resolved[key] = parameter(key)
            if resolved[key][0] is _ABSENT:
                issues.append(ValidationIssue(resolved[key][1], "is required for bilateral_filter"))
        diameter, diameter_path = resolved["diameter"]
        if diameter is not _ABSENT and (
            isinstance(diameter, bool) or not isinstance(diameter, int)
            or diameter <= 0 or diameter % 2 == 0
        ):
            issues.append(ValidationIssue(diameter_path, "must be a positive odd integer"))
        for key in ("sigma_colour", "sigma_space"):
            value, value_path = resolved[key]
            if value is not _ABSENT:
                _number(issues, value_path, value, minimum=0, exclusive_minimum=True)
    elif transform == "patch_shuffle":
        grid, grid_path = parameter("grid_size")
        if grid is _ABSENT:
            issues.append(ValidationIssue(grid_path, "is required for patch_shuffle"))
        elif isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
            issues.append(ValidationIssue(grid_path, "must be an integer >= 2"))
        else:
            image_size = _get(config, "preprocessing.image_size")
            image_size_path = "preprocessing.image_size"
            if image_size is _ABSENT:
                image_size = _get(config, "data.image_size")
                image_size_path = "data.image_size"
            if isinstance(image_size, int) and image_size % grid != 0:
                issues.append(ValidationIssue(
                    grid_path, f"must divide {image_size_path}={image_size}"
                ))


def _validate_sweeps(
    config: dict[str, Any], issues: list[ValidationIssue], workflow: str
) -> None:
    sweep = config.get("sweep", {}) or {}
    if isinstance(sweep, dict) and bool(sweep.get("enabled", False)):
        parameters = sweep.get("parameters", {})
        if not isinstance(parameters, dict):
            return
        for key, values in parameters.items():
            path = f"sweep.parameters.{key}"
            field = field_for_path(str(key))
            if field is None:
                issues.append(ValidationIssue(path, "references an unknown configuration path"))
            if not isinstance(values, list) or not values:
                issues.append(ValidationIssue(path, "must be a non-empty list"))
                continue
            if field is not None:
                for index, value in enumerate(values):
                    value_path = f"{path}[{index}]"
                    if not _is_type(value, field.expected_types):
                        issues.append(ValidationIssue(
                            value_path,
                            f"must be {_type_names(field.expected_types)}, "
                            f"got {type(value).__name__}",
                        ))
                    elif field.choices and value not in field.choices:
                        issues.append(ValidationIssue(
                            value_path, f"must be one of {list(field.choices)!r}"
                        ))
    if isinstance(sweep, dict) and "conditions" in sweep:
        try:
            conditions = normalize_conditions(sweep["conditions"])
        except ConfigNormalizationError as exc:
            issues.append(ValidationIssue("sweep.conditions", str(exc)))
        else:
            for index, condition in enumerate(conditions):
                _validate_condition_object(
                    config,
                    condition,
                    f"sweep.conditions[{index}]",
                    issues,
                    enabled=True,
                )

def _validate_condition_matrix(
    config: dict[str, Any], issues: list[ValidationIssue], workflow: str
) -> None:
    matrix = config.get("condition_matrix_evaluation", {}) or {}
    if not isinstance(matrix, dict) or not bool(matrix.get("enabled", False)):
        return
    path = "condition_matrix_evaluation.condition_names"
    try:
        from ..evaluation.condition_matrix import resolve_condition_matrix_conditions

        conditions = resolve_condition_matrix_conditions(config)
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(ValidationIssue(path, str(exc)))
        return

    if workflow != "training":
        return
    input_condition = config.get("input_condition", {}) or {}
    if not isinstance(input_condition, dict) or not bool(
        input_condition.get("enabled", False)
    ):
        issues.append(ValidationIssue(
            "input_condition.enabled",
            "must be true for a resolved condition-matrix training run",
        ))
        return
    training_name = str(
        input_condition.get("condition")
        or input_condition.get("name")
        or input_condition.get("transform", "original")
    )
    available = [str(condition["condition"]) for condition in conditions]
    if training_name not in available:
        issues.append(ValidationIssue(
            path,
            f"must include the resolved training condition {training_name!r}",
        ))


def _validate_tasks(config: dict[str, Any], issues: list[ValidationIssue]) -> None:
    target_cols = _get(config, "data.target_cols")
    if target_cols is _ABSENT:
        return
    if not isinstance(target_cols, dict) or not target_cols:
        issues.append(ValidationIssue("data.target_cols", "must be a non-empty mapping"))
        return
    for task, column in target_cols.items():
        if not isinstance(task, str) or not task.strip():
            issues.append(ValidationIssue("data.target_cols", "task names must be non-empty strings"))
        if not isinstance(column, str) or not column.strip():
            issues.append(ValidationIssue(f"data.target_cols.{task}", "column must be a non-empty string"))

    multi = config.get("multi_task", {}) or {}
    if not isinstance(multi, dict):
        return
    weights = multi.get("loss_weights", {}) or {}
    if isinstance(weights, dict):
        valid_selected_weights: dict[str, float] = {}
        for task, value in weights.items():
            if task != "hierarchy" and task not in target_cols:
                issues.append(ValidationIssue(
                    f"multi_task.loss_weights.{task}", "task is not present in data.target_cols"
                ))
            if _number(
                issues,
                f"multi_task.loss_weights.{task}",
                value,
                minimum=0,
            ) and task in target_cols:
                valid_selected_weights[task] = float(value)
        effective_weights = {
            task: valid_selected_weights.get(task, 1.0)
            for task in target_cols
            if task not in weights or task in valid_selected_weights
        }
        if (
            len(effective_weights) == len(target_cols)
            and not any(weight > 0 for weight in effective_weights.values())
        ):
            issues.append(ValidationIssue(
                "multi_task.loss_weights",
                "at least one selected task weight must be greater than zero",
            ))
    selection = multi.get("selection_metric", "mean_macro_f1")
    allowed_metrics = {
        "mean_macro_f1",
        *(f"{task}_macro_f1" for task in target_cols),
        *(f"{task}_accuracy" for task in target_cols),
        *(f"{task}_balanced_accuracy" for task in target_cols),
    }
    if isinstance(selection, str) and selection not in allowed_metrics:
        issues.append(ValidationIssue(
            "multi_task.selection_metric", f"must be one of {sorted(allowed_metrics)!r}"
        ))
    hierarchy = multi.get("hierarchy_loss", {}) or {}
    if isinstance(hierarchy, dict) and bool(hierarchy.get("enabled", False)):
        parent = hierarchy.get("parent_task", "genus")
        child = hierarchy.get("child_task", "species")
        if parent == child:
            issues.append(ValidationIssue("multi_task.hierarchy_loss", "parent_task and child_task must differ"))
        for key, task in (("parent_task", parent), ("child_task", child)):
            if task not in target_cols:
                issues.append(ValidationIssue(
                    f"multi_task.hierarchy_loss.{key}", "must name a task in data.target_cols"
                ))


def _validate_paths(config: dict[str, Any], issues: list[ValidationIssue]) -> None:
    data_root = _get(config, "data.root_dir")
    metadata = _get(config, "data.metadata_csv")
    if isinstance(data_root, str) and not Path(data_root).is_dir():
        issues.append(ValidationIssue("data.root_dir", f"directory does not exist: {data_root}"))
    dataset_format = _get(config, "data.dataset_format")
    if (
        dataset_format in {_ABSENT, "csv"}
        and isinstance(metadata, str)
        and not Path(metadata).is_file()
    ):
        issues.append(ValidationIssue("data.metadata_csv", f"file does not exist: {metadata}"))
    if dataset_format == "fish_pickle":
        data = config.get("data", {}) or {}
        metadata_dir = Path(str(data.get("metadata_dir", "../data")))
        split_dir = Path(str(data.get("split_dir", "splits")))
        if not split_dir.is_absolute():
            split_dir = metadata_dir / split_dir
        labels = Path(str(data.get("labels_json", "label_train.json")))
        if not labels.is_absolute():
            labels = metadata_dir / labels
        if not labels.is_file():
            issues.append(ValidationIssue("data.labels_json", f"file does not exist: {labels}"))
        split_files = data.get("split_files", {}) or {}
        train_path = split_dir / str(split_files.get("train", "train.pkl"))
        if not train_path.is_file():
            issues.append(ValidationIssue("data.split_files.train", f"file does not exist: {train_path}"))

    use_predefined = _get(config, "split.use_predefined_splits")
    split_root = _get(config, "split.predefined_split_dir")
    if use_predefined is True and isinstance(split_root, str):
        split_dir = Path(split_root) / "split_csv"
        for filename in ("train_split.csv", "val_split.csv", "test_split.csv"):
            if not (split_dir / filename).is_file():
                issues.append(ValidationIssue(
                    "split.predefined_split_dir",
                    f"missing required predefined split: {split_dir / filename}",
                ))

    output = _get(config, "output.out_dir")
    if isinstance(output, str):
        output_path = Path(output).resolve(strict=False)
        protected = {Path.cwd().resolve(strict=False)}
        if isinstance(data_root, str):
            protected.add(Path(data_root).resolve(strict=False))
        if isinstance(split_root, str):
            protected.add(Path(split_root).resolve(strict=False))
        if output_path in protected:
            issues.append(ValidationIssue(
                "output.out_dir", "must not be the repository, data, or predefined-split root"
            ))


def _validate_model_name(
    config: dict[str, Any], issues: list[ValidationIssue]
) -> None:
    names: list[tuple[str, Any]] = [("model.name", _get(config, "model.name"))]
    sweep = config.get("sweep", {}) or {}
    parameters = sweep.get("parameters", {}) if isinstance(sweep, dict) else {}
    sweep_models = parameters.get("model.name") if isinstance(parameters, dict) else None
    if isinstance(sweep_models, list):
        names.extend(
            (f"sweep.parameters.model.name[{index}]", name)
            for index, name in enumerate(sweep_models)
        )
    names = [(path, name) for path, name in names if isinstance(name, str) and name]
    if not names:
        return
    provider_value = _get(config, "model.provider")
    provider = (
        str(provider_value).lower()
        if provider_value is not _ABSENT
        else "auto"
    )
    dinov3_aliases = {
        "dinov3_vits16",
        "dinov3_vitb16",
        "dinov3_vitl16",
        "dinov3_convnext_tiny",
        "dinov3_convnext_small",
        "dinov3_convnext_base",
        "dinov3_convnext_large",
    }
    try:
        from torchvision import models
    except Exception:  # Cross-machine planning may not have the training stack.
        models = None
    known_torchvision = {
        "resnet18", "resnet50", "efficientnet_b0", "convnext_base", "vit_b_16"
    }
    for path, name in names:
        if name in dinov3_aliases or provider == "timm" or "dinov3" in name:
            continue
        if models is None and name in known_torchvision:
            continue
        if models is None or not callable(getattr(models, name, None)):
            issues.append(ValidationIssue(path, f"unknown torchvision model {name!r}"))

    image_size = _get(config, "preprocessing.image_size")
    for path, name in names:
        if (
            isinstance(image_size, int)
            and (name.startswith("dinov3_vit") or "patch16_dinov3" in name)
            and image_size % 16 != 0
        ):
            issues.append(ValidationIssue(
                "preprocessing.image_size",
                f"must be divisible by 16 for {path}={name!r}",
            ))


def _validate_canonical_training_switches(
    config: dict[str, Any], issues: list[ValidationIssue]
) -> None:
    """Validate config-driven training without changing legacy profiles."""
    legacy_profile = _get(config, "training.profile")
    if legacy_profile is not _ABSENT:
        issues.append(
            ValidationIssue(
                "training.profile",
                "is legacy-only; use explicit canonical trainer switches",
            )
        )
        return
    try:
        from ..training.modes import infer_experiment_type
        from ..training.modes import resolve_configured_profile
        from ..training.modes import validate_training_semantics

        profile = resolve_configured_profile(config)
        validate_training_semantics(
            config,
            profile,
            infer_experiment_type(config),
        )
    except ValueError as exc:
        issues.append(ValidationIssue("training", str(exc)))


def _validate_preprocessing_and_augmentation(
    config: dict[str, Any], issues: list[ValidationIssue], workflow: str
) -> None:
    image_size = _get(config, "preprocessing.image_size")
    legacy_image_size = _get(config, "data.image_size")
    if image_size is _ABSENT:
        image_size = legacy_image_size
    if workflow == "training" and image_size is _ABSENT:
        issues.append(ValidationIssue(
            "preprocessing.image_size",
            "is required for training (legacy data.image_size is also accepted)",
        ))

    preprocessing = config.get("preprocessing", {}) or {}
    if not isinstance(preprocessing, dict):
        return
    normalisation = preprocessing.get("normalisation", {}) or {}
    if not isinstance(normalisation, dict):
        issues.append(ValidationIssue(
            "preprocessing.normalisation", "must be a mapping"
        ))
    else:
        mean = normalisation.get("mean", [0.485, 0.456, 0.406])
        std = normalisation.get("std", [0.229, 0.224, 0.225])
        valid_vectors = True
        for name, values in (("mean", mean), ("std", std)):
            path = f"preprocessing.normalisation.{name}"
            if not isinstance(values, list) or not values:
                issues.append(ValidationIssue(path, "must be a non-empty list"))
                valid_vectors = False
                continue
            for index, value in enumerate(values):
                if not _number(issues, f"{path}[{index}]", value):
                    valid_vectors = False
                elif name == "std" and float(value) <= 0:
                    issues.append(ValidationIssue(
                        f"{path}[{index}]", "must be greater than zero"
                    ))
                    valid_vectors = False
        if valid_vectors and len(mean) != len(std):
            issues.append(ValidationIssue(
                "preprocessing.normalisation",
                "mean and std must have equal lengths",
            ))

    augmentation = config.get("augmentation", {}) or {}
    if not isinstance(augmentation, dict):
        return
    for flip in ("horizontal_flip", "vertical_flip"):
        operation = augmentation.get(flip, {}) or {}
        if not isinstance(operation, dict):
            issues.append(ValidationIssue(f"augmentation.{flip}", "must be a mapping"))
            continue
        probability = operation.get("probability", 0.5)
        _number(
            issues,
            f"augmentation.{flip}.probability",
            probability,
            minimum=0,
            maximum=1,
        )
    rotation = augmentation.get("rotation", {}) or {}
    if not isinstance(rotation, dict):
        issues.append(ValidationIssue("augmentation.rotation", "must be a mapping"))
    else:
        _number(
            issues,
            "augmentation.rotation.degrees",
            rotation.get("degrees", 270),
            minimum=0,
        )


def _condition_identifier(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name.strip():
            return name
    return None


def _validate_evaluation(config: dict[str, Any], issues: list[ValidationIssue]) -> None:
    evaluation = config.get("evaluation", {}) or {}
    if not isinstance(evaluation, dict):
        return
    for section in ("test_conditions", "condition_matrix"):
        raw = evaluation.get(section)
        if raw is None:
            continue
        path = f"evaluation.{section}"
        if not isinstance(raw, dict):
            issues.append(ValidationIssue(path, "must be a mapping"))
            continue
        conditions = raw.get("conditions", [])
        if not isinstance(conditions, list):
            issues.append(ValidationIssue(f"{path}.conditions", "must be a list"))
            continue
        try:
            from .normalization import normalize_condition_references

            conditions = normalize_condition_references(conditions)
        except (TypeError, ValueError) as exc:
            issues.append(ValidationIssue(f"{path}.conditions", str(exc)))
            continue
        if bool(raw.get("enabled", False)) and not conditions:
            issues.append(ValidationIssue(
                f"{path}.conditions", "must be non-empty when enabled"
            ))
        identifiers: list[str] = []
        for index, condition in enumerate(conditions):
            identifier = _condition_identifier(condition)
            if identifier is None:
                issues.append(ValidationIssue(
                    f"{path}.conditions[{index}]",
                    "must be a non-empty name or complete condition mapping",
                ))
                continue
            identifiers.append(identifier)
            if isinstance(condition, dict):
                _validate_condition_object(
                    config,
                    condition,
                    f"{path}.conditions[{index}]",
                    issues,
                    enabled=True,
                )
        duplicates = sorted({
            name for name in identifiers if identifiers.count(name) > 1
        })
        if duplicates:
            issues.append(ValidationIssue(
                f"{path}.conditions", f"contains duplicate names: {duplicates}"
            ))


def validate_config(
    config: dict[str, Any],
    *,
    workflow: Workflow = "auto",
    check_paths: bool = False,
    check_model_registry: bool = True,
) -> dict[str, Any]:
    """Validate without coercing, default-injecting, or mutating ``config``."""
    if not isinstance(config, dict):
        raise ConfigValidationError((ValidationIssue("config", "must be a mapping"),))
    resolved_workflow = resolve_workflow(config, workflow)
    issues: list[ValidationIssue] = []

    for section in (
        "wandb", "data", "multi_task", "multitask", "early_stopping", "split",
        "model", "training", "output", "cache", "experiment",
        "test_cue_suppression", "condition_matrix_evaluation",
        "matched_condition_training", "sweep", "input_condition",
        "preprocessing", "augmentation", "evaluation", "inference",
        "long_tail", "fine_tuning",
    ):
        value = config.get(section, _ABSENT)
        if value is not _ABSENT and value is not None and not isinstance(value, dict):
            issues.append(ValidationIssue(section, "must be a mapping"))

    for field in CONFIG_FIELDS:
        if field.path.endswith(".*"):
            continue
        value = _get(config, field.path)
        if value is _ABSENT:
            if resolved_workflow in field.required_in and field.default is MISSING_DEFAULT:
                issues.append(ValidationIssue(field.path, f"is required for {resolved_workflow}"))
            continue
        if not _is_type(value, field.expected_types):
            issues.append(ValidationIssue(
                field.path,
                f"must be {_type_names(field.expected_types)}, got {type(value).__name__}",
            ))
            continue
        if isinstance(value, float) and not math.isfinite(value):
            issues.append(ValidationIssue(field.path, "must be finite"))
        if field.choices and value not in field.choices:
            issues.append(ValidationIssue(field.path, f"must be one of {list(field.choices)!r}"))

    for path, minimum, maximum, exclusive in (
        ("data.crop_pad", 0, None, False),
        ("data.image_size", 0, None, True),
        ("preprocessing.image_size", 0, None, True),
        ("training.epochs", 0, None, True),
        ("training.batch_size", 0, None, True),
        ("training.lr", 0, None, True),
        ("training.weight_decay", 0, None, False),
        ("training.num_workers", 0, None, False),
        ("training.val_interval", 0, None, True),
        ("cache.num_workers", 1, None, False),
        ("early_stopping.patience", 0, None, False),
        ("early_stopping.min_delta", 0, None, False),
        ("multi_task.hierarchy_loss.weight", 0, None, False),
        ("training.class_weighting.beta", 0, 1, False),
        ("training.logit_adjustment.tau", 0, None, False),
        ("training.distributed.timeout_minutes", 0, None, True),
        ("long_tail.head_min_samples", 2, None, False),
        ("long_tail.staged_training.stage2_epochs", 0, None, True),
        ("long_tail.staged_training.head_replay_fraction", 0, 1, False),
        ("inference.hierarchy_genus_weight", 0, None, False),
    ):
        value = _get(config, path)
        if value is not _ABSENT and _is_type(value, (int, float)):
            _number(
                issues, path, value, minimum=minimum, maximum=maximum,
                exclusive_minimum=exclusive,
            )

    test_size = _get(config, "split.test_size")
    val_size = _get(config, "split.val_size")
    if test_size is not _ABSENT and _is_type(test_size, (int, float)):
        _number(issues, "split.test_size", test_size, minimum=0, maximum=1, exclusive_minimum=True)
        if float(test_size) >= 1:
            issues.append(ValidationIssue("split.test_size", "must be < 1"))
    if val_size is not _ABSENT and _is_type(val_size, (int, float)):
        _number(issues, "split.val_size", val_size, minimum=0, maximum=1, exclusive_minimum=True)
        if float(val_size) >= 1:
            issues.append(ValidationIssue("split.val_size", "must be < 1"))
    if all(value is not _ABSENT and _is_type(value, (int, float)) for value in (test_size, val_size)):
        if float(test_size) + float(val_size) >= 1:
            issues.append(ValidationIssue("split", "test_size + val_size must be < 1"))

    beta = _get(config, "training.class_weighting.beta")
    if (
        beta is not _ABSENT
        and _is_type(beta, (int, float))
        and float(beta) >= 1
    ):
        issues.append(ValidationIssue(
            "training.class_weighting.beta", "must be < 1"
        ))

    replay_fraction = _get(config, "long_tail.staged_training.head_replay_fraction")
    if (
        replay_fraction is not _ABSENT
        and _is_type(replay_fraction, (int, float))
        and float(replay_fraction) >= 1
    ):
        issues.append(ValidationIssue(
            "long_tail.staged_training.head_replay_fraction", "must be < 1"
        ))

    for path in ("training.mode", "experiment.type"):
        value = _get(config, path)
        if isinstance(value, str) and not value.strip():
            issues.append(ValidationIssue(path, "must be a non-empty string"))

    _validate_tasks(config, issues)
    dataset_format = _get(config, "data.dataset_format")
    if dataset_format in {_ABSENT, "csv"}:
        if _get(config, "data.metadata_csv") is _ABSENT:
            issues.append(ValidationIssue(
                "data.metadata_csv",
                "is required when data.dataset_format=csv",
            ))
    elif dataset_format == "fish_pickle":
        if _get(config, "data.metadata_dir") is _ABSENT:
            issues.append(ValidationIssue(
                "data.metadata_dir",
                "is required when data.dataset_format=fish_pickle",
            ))
    staged = _get(config, "long_tail.staged_training.enabled") is True
    if staged and _get(config, "split.strategy") != "long_tail":
        issues.append(ValidationIssue(
            "long_tail.staged_training.enabled",
            "requires split.strategy=long_tail",
        ))
    logit_task = _get(config, "training.logit_adjustment.task")
    target_cols = _get(config, "data.target_cols")
    if isinstance(logit_task, str) and isinstance(target_cols, dict) and logit_task not in target_cols:
        issues.append(ValidationIssue(
            "training.logit_adjustment.task", "must name a task in data.target_cols"
        ))
    if _get(config, "fine_tuning.enabled") is True:
        checkpoint = _get(config, "fine_tuning.checkpoint_path")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            issues.append(ValidationIssue(
                "fine_tuning.checkpoint_path",
                "must be a non-empty checkpoint path when fine_tuning.enabled=true",
            ))
        if _get(config, "fine_tuning.reset_optimizer") is False:
            issues.append(ValidationIssue(
                "fine_tuning.reset_optimizer",
                "must be true; checkpoints store model weights but no optimizer state",
            ))
    inference = config.get("inference", {}) or {}
    if isinstance(inference, dict):
        splits = inference.get("splits", [])
        if bool(inference.get("enabled", False)) and (
            not isinstance(splits, list) or not splits
        ):
            issues.append(ValidationIssue(
                "inference.splits", "must be a non-empty list when enabled"
            ))
        elif isinstance(splits, list):
            for index, name in enumerate(splits):
                if not isinstance(name, str) or not name.strip():
                    issues.append(ValidationIssue(
                        f"inference.splits[{index}]",
                        "must be a non-empty split name",
                    ))
    _validate_preprocessing_and_augmentation(config, issues, resolved_workflow)
    _validate_transform_parameters(config, issues)
    _validate_sweeps(config, issues, resolved_workflow)
    _validate_evaluation(config, issues)
    _validate_condition_matrix(config, issues, resolved_workflow)
    _validate_canonical_training_switches(config, issues)

    if resolved_workflow == "run_specs" and _get(config, "input_condition.enabled") is True:
        issues.append(ValidationIssue(
            "input_condition.enabled",
            "must be false while external matched-condition run specifications are expanded",
        ))

    if resolved_workflow == "training":
        raw = config.get("input_condition", {}) or {}
        if isinstance(raw, dict) and bool(raw.get("enabled", False)):
            for path in (
                "matched_condition_training.enabled",
                "sweep.enabled",
            ):
                if _get(config, path) is True:
                    issues.append(ValidationIssue(
                        path, "must be false when executing one externally expanded input_condition"
                    ))
            transform = str(raw.get("transform", "original")).lower()
            if _get(config, "test_cue_suppression.enabled") is True and transform != "original":
                issues.append(ValidationIssue(
                    "test_cue_suppression.enabled",
                    "fixed-RGB stress evaluation is only valid for an original/RGB training condition",
                ))

    if bool(_get(config, "matched_condition_training.enabled") is True):
        try:
            from ..experiments.conditions import generate_conditions

            generate_conditions(config)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(ValidationIssue("matched_condition_training", str(exc)))

    if check_model_registry:
        _validate_model_name(config, issues)
    if check_paths:
        _validate_paths(config, issues)
    if issues:
        raise ConfigValidationError(issues)
    return config


__all__ = [
    "ConfigValidationError",
    "KNOWN_TRANSFORMS",
    "ValidationIssue",
    "resolve_workflow",
    "validate_config",
    "validate_override_items",
]
