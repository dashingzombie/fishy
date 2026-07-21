#!/usr/bin/env python3
"""
Download all pretrained torchvision or timm/DINOv3 weights required by a config.

Default cache target: ${TORCH_HOME:-~/.cache/torch}.

Usage:
    cd ~/fish-species
    conda activate fishspecies
    python download_pretrained_from_config.py --config config.yaml
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
from pathlib import Path
from typing import Any

from fish_species.config.loading import load_config
from fish_species.models.factory import build_model


def set_nested(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = cfg

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def generate_sweep_configs(base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sweep_cfg = base_cfg.get("sweep", {}) or {}

    if not sweep_cfg.get("enabled", False):
        return [base_cfg]

    params = sweep_cfg.get("parameters", {}) or {}

    if len(params) == 0:
        return [base_cfg]

    if not isinstance(params, dict):
        raise TypeError("sweep.parameters must be a dictionary.")

    keys = list(params.keys())
    values = []

    for key in keys:
        vals = params[key]
        if not isinstance(vals, list):
            raise TypeError(f"sweep.parameters.{key} must be a list.")
        if len(vals) == 0:
            raise ValueError(f"sweep.parameters.{key} is empty.")
        values.append(vals)

    configs: list[dict[str, Any]] = []

    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base_cfg)

        for key, value in zip(keys, combo):
            set_nested(cfg, key, value)

        configs.append(cfg)

    return configs


def collect_pretrained_model_names(configs: list[dict[str, Any]]) -> list[str]:
    """
    Collect model names that require pretrained weights.

    Missing model.pretrained defaults to True because most training scripts
    use pretrained=True as the default unless explicitly disabled.
    """
    names = set()

    for cfg in configs:
        model_cfg = cfg.get("model", {}) or {}

        name = model_cfg.get("name")
        pretrained = model_cfg.get("pretrained", True)

        if name is None:
            continue

        if bool(pretrained):
            names.add(str(name))

    return sorted(names)


def download_pretrained_model(name: str) -> None:
    print(f"\n[DOWNLOAD/CHECK] {name}")
    _ = build_model(name=name, num_classes=1, pretrained=True, provider="auto")
    print(f"[OK] {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch")),
        help="Torch cache root. Checkpoints go under <torch-home>/hub/checkpoints.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    args.torch_home.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.torch_home / "hub" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_HOME"] = str(args.torch_home)

    print("Config:", args.config.resolve())
    print("TORCH_HOME:", os.environ["TORCH_HOME"])
    print("Checkpoint directory:", checkpoints_dir)

    base_cfg = load_config(args.config)

    configs = generate_sweep_configs(base_cfg)
    model_names = collect_pretrained_model_names(configs)

    print(f"\nNumber of expanded configs: {len(configs)}")

    if len(model_names) == 0:
        print("No pretrained models required by this config.")
        return

    print("Pretrained models required:")
    for name in model_names:
        print(f"  - {name}")

    for name in model_names:
        download_pretrained_model(name)

    print("\nDone.")
    print("Cached checkpoint files:")
    for path in sorted(checkpoints_dir.glob("*")):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name:45s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
