"""Deterministic random-state setup shared by training and analysis."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and every available CUDA generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


__all__ = ["set_seed"]
