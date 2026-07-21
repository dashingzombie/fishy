from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def foreground_bbox_from_image(
    image: Image.Image,
) -> tuple[int, int, int, int] | None:
    array = np.asarray(image.convert("RGB"))
    mask = array.mean(axis=2) > 5
    if mask.sum() < 20:
        return None
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def foreground_bbox_from_mask(
    mask_path: Path,
) -> tuple[int, int, int, int] | None:
    if not mask_path.exists():
        return None
    array = np.asarray(Image.open(mask_path).convert("L"))
    foreground = array > 0
    if foreground.sum() < 20:
        return None
    ys, xs = np.where(foreground)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def pad_square_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    side = int(max(x2 - x1, y2 - y1) * (1.0 + pad))
    centre_x = (x1 + x2) // 2
    centre_y = (y1 + y2) // 2
    new_x1 = max(0, centre_x - side // 2)
    new_y1 = max(0, centre_y - side // 2)
    new_x2 = min(img_w, new_x1 + side)
    new_y2 = min(img_h, new_y1 + side)
    new_x1 = max(0, new_x2 - side)
    new_y1 = max(0, new_y2 - side)
    return new_x1, new_y1, new_x2, new_y2
