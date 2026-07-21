"""Module entry point for canonical SLURM rendering and submission."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
