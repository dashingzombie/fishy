from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from torchvision import transforms

from .transform_ops import build_condition_operations


SplitName = Literal["train", "validation", "test"]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_NORMALISATION_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALISATION_STD = (0.229, 0.224, 0.225)


def _mapping(value, path: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _enabled_operation(config: dict, key: str, defaults: dict) -> dict | None:
    operation = {**defaults, **_mapping(config.get(key), f"augmentation.{key}")}
    return operation if bool(operation.get("enabled", True)) else None


def _augmentation_operations(augmentation: Mapping | None) -> list:
    config = _mapping(augmentation, "augmentation")
    if not bool(config.get("enabled", True)):
        return []

    operations = []
    horizontal = _enabled_operation(
        config,
        "horizontal_flip",
        {"enabled": True, "probability": 0.5},
    )
    if horizontal is not None:
        operations.append(
            transforms.RandomHorizontalFlip(p=float(horizontal["probability"]))
        )

    vertical = _enabled_operation(
        config,
        "vertical_flip",
        {"enabled": True, "probability": 0.5},
    )
    if vertical is not None:
        operations.append(
            transforms.RandomVerticalFlip(p=float(vertical["probability"]))
        )

    rotation = _enabled_operation(
        config,
        "rotation",
        {"enabled": True, "degrees": 270},
    )
    if rotation is not None:
        # Keep RandomRotation(0) when enabled: removing it would change RNG
        # consumption and therefore later stochastic samples.
        operations.append(
            transforms.RandomRotation(degrees=float(rotation["degrees"]))
        )
    return operations


def _normalisation_operation(preprocessing: dict):
    config = _mapping(
        preprocessing.get("normalisation"),
        "preprocessing.normalisation",
    )
    if not bool(config.get("enabled", True)):
        return None
    return transforms.Normalize(
        mean=config.get("mean", list(DEFAULT_NORMALISATION_MEAN)),
        std=config.get("std", list(DEFAULT_NORMALISATION_STD)),
    )


def build_split_transform(
    *,
    split: SplitName,
    preprocessing: Mapping | None = None,
    augmentation: Mapping | None = None,
    condition: Mapping | None = None,
    original_colour_retention: float = 1.0,
    apply_augmentation: bool | None = None,
) -> transforms.Compose:
    """Compose deterministic preprocessing, augmentation, and input condition.

    Augmentation is train-only unless a diagnostic caller explicitly overrides
    ``apply_augmentation``. Experimental conditions are tensor operations and
    always precede normalisation.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(
            f"split must be 'train', 'validation', or 'test', got {split!r}"
        )
    preprocessing_config = _mapping(preprocessing, "preprocessing")
    image_size = preprocessing_config.get("image_size", DEFAULT_IMAGE_SIZE)
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size <= 0
    ):
        raise ValueError("preprocessing.image_size must be a positive integer")

    operations = [transforms.Resize((image_size, image_size))]
    augment = (
        split == "train"
        if apply_augmentation is None
        else apply_augmentation
    )
    if augment:
        operations.extend(_augmentation_operations(augmentation))
    operations.append(transforms.ToTensor())
    operations.extend(
        build_condition_operations(
            _mapping(condition, "input_condition"),
            original_colour_retention=original_colour_retention,
        )
    )
    normalisation = _normalisation_operation(preprocessing_config)
    if normalisation is not None:
        operations.append(normalisation)
    return transforms.Compose(operations)
