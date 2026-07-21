"""Lightweight data-package exports.

Transform and dataset implementations remain available from their named
modules.  Keeping this initializer small preserves the historical behavior of
``import fish_species.data.labels`` without eagerly importing torchvision or
OpenCV.
"""

from .labels import (
    build_label_maps,
    read_csvs_from_dir,
)

__all__ = [
    "build_label_maps",
    "read_csvs_from_dir",
]
