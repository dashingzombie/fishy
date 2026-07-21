"""Validated, side-effect-free SLURM configuration and planning."""

from .config import (
    SlurmConfigError,
    deep_merge,
    load_submission_config,
    parse_memory,
    parse_time_limit,
    validate_slurm_config,
)
from .planning import RunSpec, SubmissionPlan, plan_submission

__all__ = [
    "RunSpec",
    "SlurmConfigError",
    "SubmissionPlan",
    "deep_merge",
    "load_submission_config",
    "parse_memory",
    "parse_time_limit",
    "plan_submission",
    "validate_slurm_config",
]
