"""Runtime checkpoint schema contracts for every training profile."""

from __future__ import annotations

from pathlib import Path

import torch

from .modes import TrainingProfile


BASE_CHECKPOINT_KEYS = {
    "model_state",
    "cfg",
    "label_to_index_by_task",
    "index_to_label_by_task",
    "best_val_score",
    "selection_metric",
    "best_epoch",
    "long_tail_metadata",
}


def checkpoint_keys(profile: TrainingProfile) -> set[str]:
    keys = set(BASE_CHECKPOINT_KEYS)
    if profile.loader_mode == "condition":
        keys.add("training_condition")
    return keys


def build_checkpoint_payload(
    *,
    profile: TrainingProfile,
    model_state,
    cfg: dict,
    label_to_index_by_task: dict,
    index_to_label_by_task: dict,
    best_val_score: float,
    selection_metric: str,
    best_epoch: int,
    training_condition: dict | None = None,
    long_tail_metadata: dict | None = None,
) -> dict:
    payload = {
        "model_state": model_state,
        "cfg": cfg,
        "label_to_index_by_task": label_to_index_by_task,
        "index_to_label_by_task": index_to_label_by_task,
        "best_val_score": best_val_score,
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "long_tail_metadata": long_tail_metadata or {},
    }
    if profile.loader_mode == "condition":
        payload["training_condition"] = training_condition
    assert set(payload) == checkpoint_keys(profile)
    return payload


def save_checkpoint(payload: dict, path: str | Path) -> None:
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location=None) -> dict:
    payload = torch.load(path, map_location=map_location)
    payload.setdefault("long_tail_metadata", {})
    return payload


def load_model_state_compat(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    initialize_dual_from_species: bool = True,
) -> tuple[list[str], list[str]]:
    """Load legacy matching tensors and report newly initialized modules.

    Shape mismatches are errors. When explicitly allowed, a legacy
    ``species_head`` is copied into both dual heads.
    """
    current = model.state_dict()
    adapted = dict(state)
    if initialize_dual_from_species:
        for suffix in ("weight", "bias", "log_scale"):
            old_key = next((key for key in (
                f"species_head.{suffix}", f"heads.species.{suffix}"
            ) if key in state), None)
            if old_key is not None:
                for name in ("natural_head", "balanced_head"):
                    new_key = f"{name}.{suffix}"
                    if new_key in current and new_key not in adapted:
                        adapted[new_key] = state[old_key]
                single_key = f"species_head.{suffix}"
                if single_key in current and single_key not in adapted:
                    adapted[single_key] = state[old_key]
    incompatible_shapes = [
        key for key, value in adapted.items()
        if key in current and tuple(value.shape) != tuple(current[key].shape)
    ]
    if incompatible_shapes:
        raise RuntimeError(
            "Checkpoint tensor shapes are incompatible: "
            + ", ".join(incompatible_shapes)
        )
    matching = {key: value for key, value in adapted.items() if key in current}
    result = model.load_state_dict(matching, strict=False)
    ignored = [key for key in adapted if key not in current]
    return list(result.missing_keys), sorted(set(result.unexpected_keys) | set(ignored))
