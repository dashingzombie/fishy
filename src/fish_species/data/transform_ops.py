from __future__ import annotations

import math

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as transform_functional

try:
    import cv2
except ImportError:
    cv2 = None
else:
    cv2.setNumThreads(0)


class SaturationRetention:
    """Retain a configured fraction of image chroma."""

    def __init__(self, retention: float = 1.0):
        retention = float(retention)
        if not 0.0 <= retention <= 1.0:
            raise ValueError(
                f"saturation retention must be between 0 and 1, got {retention}."
            )
        self.retention = retention

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise TypeError(
                "SaturationRetention must be applied after transforms.ToTensor()."
            )
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected an RGB tensor with shape [3, H, W], got {tuple(image.shape)}."
            )
        if self.retention == 1.0:
            return image
        greyscale = transform_functional.rgb_to_grayscale(
            image, num_output_channels=3
        )
        return torch.lerp(greyscale, image, self.retention).clamp_(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(retention={self.retention:.3f})"


class ChannelShuffle:
    """Apply a fixed RGB-channel permutation."""

    def __init__(self, order: tuple[int, int, int] | list[int]):
        self.order = tuple(int(index) for index in order)
        if sorted(self.order) != [0, 1, 2]:
            raise ValueError(
                f"Channel order must be a permutation of [0, 1, 2], got {self.order}."
            )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected an RGB tensor with shape [3, H, W], got {tuple(image.shape)}."
            )
        return image[list(self.order), :, :]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(order={self.order})"


class TensorGaussianBlur:
    """Gaussian smoothing applied before ImageNet normalisation."""

    def __init__(self, sigma: float):
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise ValueError(f"Gaussian sigma must be > 0, got {self.sigma}.")
        self.kernel_size = max(3, 2 * int(math.ceil(3.0 * self.sigma)) + 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return transform_functional.gaussian_blur(
            image,
            kernel_size=[self.kernel_size, self.kernel_size],
            sigma=[self.sigma, self.sigma],
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sigma={self.sigma:.3f}, "
            f"kernel_size={self.kernel_size})"
        )


class TensorBilateralFilter:
    """Edge-preserving bilateral smoothing using OpenCV."""

    def __init__(self, diameter: int, sigma_colour: float, sigma_space: float):
        self.diameter = int(diameter)
        self.sigma_colour = float(sigma_colour)
        self.sigma_space = float(sigma_space)
        if self.diameter <= 0 or self.diameter % 2 == 0:
            raise ValueError(
                "Bilateral diameter must be a positive odd integer, "
                f"got {self.diameter}."
            )
        if self.sigma_colour <= 0 or self.sigma_space <= 0:
            raise ValueError("Bilateral sigma values must be greater than zero.")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if cv2 is None:
            raise ImportError(
                "OpenCV is required for bilateral filtering. Install opencv-python "
                "or opencv-python-headless in the training environment."
            )
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected an RGB tensor with shape [3, H, W], got {tuple(image.shape)}."
            )
        array = image.detach().cpu().permute(1, 2, 0).numpy()
        array_u8 = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        filtered = cv2.bilateralFilter(
            array_u8,
            d=self.diameter,
            sigmaColor=self.sigma_colour,
            sigmaSpace=self.sigma_space,
        )
        output = torch.from_numpy(filtered).permute(2, 0, 1).to(dtype=image.dtype)
        return output.div_(255.0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(diameter={self.diameter}, "
            f"sigma_colour={self.sigma_colour:.3f}, "
            f"sigma_space={self.sigma_space:.3f})"
        )


class PatchShuffle:
    """Deterministically shuffle a regular image-patch grid."""

    def __init__(self, grid_size: int, seed: int = 0):
        self.grid_size = int(grid_size)
        self.seed = int(seed)
        if self.grid_size < 2:
            raise ValueError("Patch-shuffle grid_size must be at least 2.")
        generator = torch.Generator().manual_seed(
            self.seed + 1009 * self.grid_size
        )
        patch_count = self.grid_size * self.grid_size
        permutation = torch.randperm(patch_count, generator=generator)
        if torch.equal(permutation, torch.arange(patch_count)):
            permutation = permutation.roll(1)
        self.permutation = permutation.tolist()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(f"Expected [C, H, W], got {tuple(image.shape)}.")
        _, height, width = image.shape
        if height % self.grid_size != 0 or width % self.grid_size != 0:
            raise ValueError(
                f"Image size {(height, width)} must be divisible by patch grid "
                f"{self.grid_size}. Choose a compatible grid size."
            )
        patch_height = height // self.grid_size
        patch_width = width // self.grid_size
        output = torch.empty_like(image)
        for destination, source in enumerate(self.permutation):
            destination_row, destination_column = divmod(
                destination, self.grid_size
            )
            source_row, source_column = divmod(source, self.grid_size)
            output[
                :,
                destination_row * patch_height:(destination_row + 1) * patch_height,
                destination_column * patch_width:(destination_column + 1) * patch_width,
            ] = image[
                :,
                source_row * patch_height:(source_row + 1) * patch_height,
                source_column * patch_width:(source_column + 1) * patch_width,
            ]
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grid_size={self.grid_size}, seed={self.seed})"


def _condition_parameters(condition: dict) -> dict:
    """Resolve canonical nested parameters plus legacy flat parameters once."""
    raw_parameters = condition.get("parameters", {}) or {}
    if not isinstance(raw_parameters, dict):
        raise TypeError("input condition parameters must be a mapping")
    parameters = dict(raw_parameters)
    for key in (
        "retention",
        "order",
        "diameter",
        "sigma_colour",
        "sigma_space",
        "sigma",
        "grid_size",
        "seed",
    ):
        if key in condition:
            parameters[key] = condition[key]
    return parameters


def build_condition_operations(
    condition: dict | None = None,
    original_colour_retention: float = 1.0,
) -> list:
    """Build the tensor-stage operations for one experimental condition."""
    condition = dict(condition or {})
    transform_name = str(condition.get("transform", "original")).lower()
    parameters = _condition_parameters(condition)

    if transform_name == "saturation":
        retention = float(parameters["retention"])
    elif transform_name == "grayscale":
        retention = 0.0
    else:
        retention = float(original_colour_retention)
    operations = [SaturationRetention(retention)]

    if transform_name in {"original", "saturation", "grayscale"}:
        pass
    elif transform_name == "channel_shuffle":
        operations.append(ChannelShuffle(parameters.get("order", [2, 0, 1])))
    elif transform_name == "gaussian_blur":
        operations.append(TensorGaussianBlur(float(parameters["sigma"])))
    elif transform_name == "bilateral_filter":
        operations.append(TensorBilateralFilter(
            diameter=int(parameters["diameter"]),
            sigma_colour=float(parameters["sigma_colour"]),
            sigma_space=float(parameters["sigma_space"]),
        ))
    elif transform_name == "patch_shuffle":
        operations.append(PatchShuffle(
            grid_size=int(parameters["grid_size"]),
            seed=int(parameters.get("seed", 0)),
        ))
    else:
        raise ValueError(f"Unsupported input transform: {transform_name!r}.")
    return operations


def build_condition_transform(
    image_size: int,
    train: bool,
    condition: dict | None = None,
    original_colour_retention: float = 1.0,
) -> transforms.Compose:
    """Build one matched train/evaluation condition without changing ordering."""
    from .transforms import build_split_transform

    return build_split_transform(
        split="train" if train else "validation",
        preprocessing={"image_size": image_size},
        condition=condition,
        original_colour_retention=original_colour_retention,
    )


def build_test_condition_transform(
    image_size: int,
    condition: dict,
    original_colour_retention: float = 1.0,
) -> transforms.Compose:
    """Build a deterministic fixed-checkpoint evaluation transform."""
    from .transforms import build_split_transform

    return build_split_transform(
        split="test",
        preprocessing={"image_size": image_size},
        condition=condition,
        original_colour_retention=original_colour_retention,
    )
