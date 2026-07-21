"""Command line interface for persistent image-cache maintenance."""

from __future__ import annotations

import argparse
import json
import sys

from .maintenance import CacheMaintenanceError
from .maintenance import build_persistent_cache, verify_persistent_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fish_species.cache")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build or reuse a persistent cache")
    build.add_argument("--config", default="config.yaml")
    build.add_argument("--data-root", required=True)
    build.add_argument("--metadata-csv", required=True)
    build.add_argument("--cache-dir", required=True)
    build.add_argument("--image-col", default="rel_path_seg")
    build.add_argument("--force", "--force-rebuild", action="store_true")
    build.add_argument("--json", action="store_true", dest="json_output")
    verify = subparsers.add_parser("verify", help="verify marker and manifest")
    verify.add_argument("--cache-dir", required=True)
    verify.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_persistent_cache(
                args.config,
                data_root=args.data_root,
                metadata_csv=args.metadata_csv,
                cache_dir=args.cache_dir,
                image_col=args.image_col,
                force=args.force,
            )
        else:
            result = verify_persistent_cache(args.cache_dir)
    except (OSError, ValueError) as exc:
        print(f"configuration/path error: {exc}", file=sys.stderr)
        return 2
    except CacheMaintenanceError as exc:
        print(f"cache error: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"{result.status}: {result.cache_dir}")
    return 0
