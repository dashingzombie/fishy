"""Evaluation components, imported lazily to keep metric-only use lightweight."""

from __future__ import annotations

from typing import Any

__all__ = [
    "evaluate_condition_matrix",
    "evaluation_relation",
    "generate_test_cue_conditions",
    "make_test_condition_loader",
    "resolve_condition_matrix_conditions",
]


def __getattr__(name: str) -> Any:
    if name in {"generate_test_cue_conditions", "make_test_condition_loader"}:
        from . import cue_suppression
        return getattr(cue_suppression, name)
    if name in {
        "evaluate_condition_matrix", "evaluation_relation",
        "resolve_condition_matrix_conditions",
    }:
        from . import condition_matrix
        return getattr(condition_matrix, name)
    raise AttributeError(name)
