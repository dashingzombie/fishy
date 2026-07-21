"""Print a canonical, resolved configuration without modifying its source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .loading import load_config
from .normalization import ConfigNormalizationError, normalize_config_with_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print an equivalent canonical configuration to stdout."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--format", choices=("yaml", "json"), default="yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = normalize_config_with_report(load_config(args.config))
    except (ConfigNormalizationError, OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"Compatibility: {warning.message}", file=sys.stderr)
    if args.format == "json":
        print(json.dumps(result.config, indent=2))
    else:
        print(yaml.safe_dump(result.config, sort_keys=False).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
