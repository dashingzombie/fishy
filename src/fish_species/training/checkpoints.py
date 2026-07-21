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
) -> dict:
    payload = {
        "model_state": model_state,
        "cfg": cfg,
        "label_to_index_by_task": label_to_index_by_task,
        "index_to_label_by_task": index_to_label_by_task,
        "best_val_score": best_val_score,
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
    }
    if profile.loader_mode == "condition":
        payload["training_condition"] = training_condition
    assert set(payload) == checkpoint_keys(profile)
    return payload


def save_checkpoint(payload: dict, path: str | Path) -> None:
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location=None) -> dict:
    return torch.load(path, map_location=map_location)
