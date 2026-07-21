from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from .cropping import (
    foreground_bbox_from_image,
    foreground_bbox_from_mask,
    pad_square_bbox,
)
from .image_validation import resolve_path
class MultiTaskImageDataset(Dataset):
    """Image dataset whose supervised rows have complete task labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str | Path,
        image_col: str,
        target_col: str | None = None,
        label_to_index: dict[str, int] | None = None,
        transform=None,
        mask_col: str | None = None,
        crop_to_foreground: bool = True,
        crop_pad: float = 0.15,
        target_cols: dict[str, str] | None = None,
        label_to_index_by_task: dict[str, dict[str, int]] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.image_col = image_col
        self.mask_col = mask_col
        self.target_col = target_col
        self.label_to_index = label_to_index
        self.target_cols = target_cols
        self.label_to_index_by_task = label_to_index_by_task
        self.transform = transform
        self.crop_to_foreground = crop_to_foreground
        self.crop_pad = crop_pad
        self.multi_task = target_cols is not None
        if self.multi_task:
            if label_to_index_by_task is None:
                raise ValueError(
                    "label_to_index_by_task must be provided for multi-task training."
                )
        elif target_col is None or label_to_index is None:
            raise ValueError(
                "target_col and label_to_index must be provided for single-task training."
            )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, index: int):
        row = self.df.iloc[index]
        image_path = resolve_path(self.root_dir, row[self.image_col])
        image = Image.open(image_path).convert("RGB")

        if self.crop_to_foreground:
            bbox = None
            if (
                self.mask_col is not None
                and self.mask_col in row
                and pd.notna(row[self.mask_col])
            ):
                mask_path = resolve_path(self.root_dir, row[self.mask_col])
                bbox = foreground_bbox_from_mask(mask_path)
            if bbox is None:
                bbox = foreground_bbox_from_image(image)
            if bbox is not None:
                width, height = image.size
                bbox = pad_square_bbox(
                    bbox, width, height, self.crop_pad
                )
                image = image.crop(bbox)

        if self.transform is not None:
            image = self.transform(image)
        return row, image, image_path

    def __getitem__(self, index: int):
        row, image, image_path = self._load_image(index)

        if self.multi_task:
            labels = {}
            label_names = {}
            for task, column in self.target_cols.items():
                if column not in row or pd.isna(row[column]):
                    raise ValueError(f"Sample {index} has no {task!r} label")
                label_name = str(row[column])
                if label_name not in self.label_to_index_by_task[task]:
                    raise ValueError(
                        f"Sample {index} has unknown {task!r} label {label_name!r}"
                    )
                encoded = self.label_to_index_by_task[task][label_name]
                label_names[task] = label_name
                labels[task] = torch.tensor(encoded, dtype=torch.long)
            return {
                "image": image,
                "labels": labels,
                "label_names": label_names,
                "path": str(image_path),
                "sample_id": str(row.get("image_id", Path(image_path).name)),
            }

        label_name = row[self.target_col]
        encoded = self.label_to_index[label_name]
        return {
            "image": image,
            "label": torch.tensor(encoded, dtype=torch.long),
            "label_name": label_name,
            "path": str(image_path),
        }


# Backwards-compatible import for older fish experiment modules.
MultiTaskFishImageDataset = MultiTaskImageDataset


class InferenceImageDataset(MultiTaskImageDataset):
    """Image-only dataset for the official unlabeled test/unseen files."""

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(
            df,
            target_col="__inference__",
            label_to_index={"__inference__": 0},
            target_cols=None,
            label_to_index_by_task=None,
            **kwargs,
        )

    def __getitem__(self, index: int):
        row, image, image_path = self._load_image(index)
        return {
            "image": image,
            "path": str(image_path),
            "sample_id": str(row.get("image_id", Path(image_path).name)),
        }
