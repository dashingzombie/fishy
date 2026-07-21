from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .loading import load_config
from .overrides import apply_overrides
from .validation import (
    ConfigValidationError,
    resolve_workflow,
    validate_config,
    validate_override_items,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a fish-species configuration.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Apply existing dotted key=value overrides before validation.",
    )
    parser.add_argument(
        "--workflow",
        choices=("auto", "training", "run_specs", "saved"),
        default="auto",
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Check local data and predefined-split paths (off for dry runs by default).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_override_items(args.override)
        config = apply_overrides(load_config(args.config), args.override)
        workflow = resolve_workflow(config, args.workflow)
        validate_config(
            config,
            workflow=args.workflow,
            check_paths=args.check_paths,
            check_model_registry=True,
        )
    except (ConfigValidationError, OSError, TypeError, ValueError) as exc:
        if args.json:
            print(json.dumps({
                "valid": False,
                "config": str(args.config),
                "error": str(exc),
            }, indent=2))
        else:
            print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "valid": True,
            "config": str(args.config),
            "workflow": workflow,
            "paths_checked": bool(args.check_paths),
        }, indent=2))
    else:
        print(f"Configuration valid: {args.config}")
        print(f"Workflow: {workflow}")
        print(f"Paths checked: {'yes' if args.check_paths else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
