"""Unified command dispatcher for configuration utilities."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print("usage: python -m fish_species.config {inspect,validate,migrate,tui} ...")
        return 0
    command, *remainder = arguments
    if command == "inspect":
        from .inspect import main as command_main
    elif command == "validate":
        from .validate import main as command_main
    elif command == "migrate":
        from .migrate import main as command_main
    elif command in {"tui", "wizard"}:
        from .tui import main as command_main
    else:
        print(f"Unknown config command: {command!r}", file=sys.stderr)
        return 2
    return command_main(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
