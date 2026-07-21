from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from src.fish_species.config.loading import load_config
from src.fish_species.config.overrides import apply_overrides, parse_scalar, set_nested


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
        handle.write("\n")


def short_hash(obj: Any, length: int = 8) -> str:
    if length < 1:
        raise ValueError("Hash length must be positive")
    text = json.dumps(obj, sort_keys=True)
    # MD5 is used only as a stable, compact identifier; it is not a security hash.
    return hashlib.md5(text.encode()).hexdigest()[:length]


def make_run_name(cfg: dict[str, Any]) -> str:
    parts = [
        cfg["model"]["name"],
        cfg["data"]["image_col"],
        cfg["data"]["target_col"],
        short_hash(cfg),
    ]
    return "__".join(str(p) for p in parts)
