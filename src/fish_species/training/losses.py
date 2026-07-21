"""Class-balanced task losses and taxonomy-hierarchy consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


class LogitAdjustedCrossEntropy(nn.Module):
    """Cross-entropy with training-prior logit adjustment."""

    def __init__(
        self,
        prior: torch.Tensor,
        tau: float = 1.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("adjustment", float(tau) * prior.clamp_min(1e-12).log())
        self.register_buffer("class_weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits + self.adjustment.to(dtype=logits.dtype),
            target,
            weight=self.class_weight,
        )


def compute_class_weights(
    train_df: pd.DataFrame,
    target_col: str,
    label_to_index: dict[str, int],
    *,
    basis: str = "samples",
    group_col: str | None = None,
    method: str = "inverse_frequency",
    beta: float = 0.9999,
) -> torch.Tensor:
    """Compute normalized per-class weights from the labeled training split."""
    labelled = train_df[train_df[target_col].notna()]
    if basis == "samples":
        counts = labelled[target_col].value_counts().to_dict()
    elif basis == "groups":
        if not group_col:
            raise ValueError("group_col is required when class-weight basis is 'groups'")
        counts = labelled.groupby(target_col)[group_col].nunique().to_dict()
    else:
        raise ValueError("Class-weight basis must be 'samples' or 'groups'")

    weights = []
    for label in label_to_index:
        count = max(float(counts.get(label, 1.0)), 1.0)
        if method == "inverse_frequency":
            weight = 1.0 / count
        elif method == "sqrt_inverse_frequency":
            weight = 1.0 / np.sqrt(count)
        elif method == "effective_number":
            if not 0 <= beta < 1:
                raise ValueError(
                    "training.class_weighting.beta must be in [0, 1)"
                )
            weight = (1.0 - beta) / max(1.0 - beta ** count, 1e-12)
        else:
            raise ValueError(
                "Class-weight method must be 'inverse_frequency', "
                "'sqrt_inverse_frequency', or 'effective_number'"
            )
        weights.append(weight)

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def compute_individual_class_weights(
    train_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    label_to_index: dict[str, int],
) -> torch.Tensor:
    """Compatibility wrapper for historical individual-weighted experiments."""
    return compute_class_weights(
        train_df,
        target_col,
        label_to_index,
        basis="groups",
        group_col=group_col,
    )


def build_criteria(
    train_df: pd.DataFrame,
    target_cols: dict[str, str],
    group_col: str,
    label_to_index_by_task: dict[str, dict[str, int]],
    device: torch.device,
    *,
    use_class_weights: bool = True,
    class_weighting: dict | None = None,
    logit_adjustment: dict | None = None,
) -> dict[str, nn.Module]:
    """Build per-task losses with configurable class-frequency weighting."""
    criteria = {}
    weighting = class_weighting or {}
    basis = str(weighting.get("basis", "samples"))
    method = str(weighting.get("method", "inverse_frequency"))
    beta = float(weighting.get("beta", 0.9999))
    adjustment_cfg = logit_adjustment or {}
    adjustment_enabled = bool(adjustment_cfg.get("enabled", False))
    adjustment_tau = float(adjustment_cfg.get("tau", 1.0))
    adjustment_task = str(adjustment_cfg.get("task", "species"))

    for task, col in target_cols.items():
        class_weights = None
        if use_class_weights:
            class_weights = compute_class_weights(
                train_df=train_df,
                target_col=col,
                label_to_index=label_to_index_by_task[task],
                basis=basis,
                group_col=group_col,
                method=method,
                beta=beta,
            ).to(device)

        if adjustment_enabled and task == adjustment_task:
            counts = train_df[col].value_counts()
            prior = torch.tensor(
                [float(counts.get(label, 0)) for label in label_to_index_by_task[task]],
                dtype=torch.float32,
                device=device,
            )
            prior = prior / prior.sum().clamp_min(1.0)
            criteria[task] = LogitAdjustedCrossEntropy(
                prior=prior,
                tau=adjustment_tau,
                weight=class_weights,
            )
        else:
            criteria[task] = nn.CrossEntropyLoss(weight=class_weights)

    return criteria


def build_dual_species_criteria(
    train_df: pd.DataFrame,
    species_col: str,
    label_to_index: dict[str, int],
    group_col: str,
    device: torch.device,
    config: dict,
) -> tuple[nn.Module, nn.Module]:
    """Build ordinary natural-head CE and exactly one balanced-head correction."""
    natural = nn.CrossEntropyLoss()
    method = str(config.get("balanced_method", "logit_adjustment"))
    if method == "none":
        return natural, nn.CrossEntropyLoss()
    if method == "class_weight":
        weights = compute_class_weights(
            train_df, species_col, label_to_index,
            basis="samples", group_col=group_col,
        ).to(device)
        return natural, nn.CrossEntropyLoss(weight=weights)
    if method == "logit_adjustment":
        counts = train_df[species_col].value_counts()
        prior = torch.tensor(
            [float(counts.get(label, 0)) for label in label_to_index],
            dtype=torch.float32, device=device,
        )
        prior /= prior.sum().clamp_min(1.0)
        return natural, LogitAdjustedCrossEntropy(
            prior, tau=float(config.get("tau", 1.0))
        )
    raise ValueError(
        "training.dual_species_classifier.balanced_method must be "
        "'logit_adjustment', 'class_weight', or 'none'"
    )


def infer_parent_label_from_child_label(child_label: str) -> str:
    """Infer a genus-like parent from a space- or underscore-delimited label."""
    child_label = str(child_label).strip()

    if " " in child_label:
        return child_label.split()[0]
    if "_" in child_label:
        return child_label.split("_")[0]

    return child_label


def build_child_to_parent_matrix(
    label_to_index_by_task: dict[str, dict[str, int]],
    parent_task: str,
    child_task: str,
    device: torch.device,
    child_to_parent: dict[str, str] | None = None,
) -> torch.Tensor:
    """Build the exact legacy child-class to parent-class mapping matrix."""
    if parent_task not in label_to_index_by_task:
        raise ValueError(f"Parent task {parent_task!r} is not in label_to_index_by_task.")
    if child_task not in label_to_index_by_task:
        raise ValueError(f"Child task {child_task!r} is not in label_to_index_by_task.")

    parent_to_index = label_to_index_by_task[parent_task]
    child_to_index = label_to_index_by_task[child_task]
    child_to_parent = child_to_parent or {}

    matrix = torch.zeros(
        len(child_to_index),
        len(parent_to_index),
        dtype=torch.float32,
        device=device,
    )

    missing_parent_labels = []

    for child_label, child_index in child_to_index.items():
        parent_label = child_to_parent.get(
            child_label,
            infer_parent_label_from_child_label(child_label),
        )

        if parent_label not in parent_to_index:
            missing_parent_labels.append((child_label, parent_label))
            continue

        parent_index = parent_to_index[parent_label]
        matrix[child_index, parent_index] = 1.0

    if missing_parent_labels:
        examples = ", ".join(
            f"{child!r}->{parent!r}" for child, parent in missing_parent_labels[:10]
        )
        raise ValueError(
            f"Could not map {len(missing_parent_labels)} {child_task!r} labels "
            f"to valid {parent_task!r} labels. Examples: {examples}. "
            "Either make sure species labels start with the genus name, "
            "or provide multi_task.hierarchy_loss.child_to_parent in the config."
        )

    if not torch.all(matrix.sum(dim=1) == 1):
        raise ValueError(
            f"Each {child_task!r} class must map to exactly one {parent_task!r} class."
        )

    return matrix


def hierarchy_consistency_loss(
    parent_logits: torch.Tensor,
    child_logits: torch.Tensor,
    child_to_parent_matrix: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor | None:
    """Penalise parent/child disagreement on jointly labelled samples.

    The hierarchy calculation is performed in float32 because probability-space
    aggregation and logarithms are numerically unstable under float16 autocast.
    """
    if not torch.any(valid_mask):
        return None

    device_type = parent_logits.device.type

    # Explicitly exclude this calculation from AMP.
    with torch.autocast(device_type=device_type, enabled=False):
        parent_logits = parent_logits[valid_mask].float()
        child_logits = child_logits[valid_mask].float()

        child_to_parent_matrix = child_to_parent_matrix.to(
            device=child_logits.device,
            dtype=torch.float32,
        )

        # log_softmax avoids calculating softmax followed by log for the
        # directly predicted parent distribution.
        parent_log_probs = F.log_softmax(parent_logits, dim=1)
        parent_probs = parent_log_probs.exp()

        # Probability-space aggregation is required for the child-derived
        # parent distribution.
        child_probs = F.softmax(child_logits, dim=1)
        implied_parent_probs = child_probs @ child_to_parent_matrix

        # Aggregation may still produce exact zeros, even in float32.
        implied_parent_probs = implied_parent_probs.clamp_min(eps)

        # Clamping slightly changes the row sums, so restore a valid
        # probability distribution.
        implied_parent_probs = implied_parent_probs / implied_parent_probs.sum(
            dim=1,
            keepdim=True,
        )
        implied_parent_log_probs = implied_parent_probs.log()

        # KL(implied parent || predicted parent):
        # gradients update the parent head only.
        parent_loss = F.kl_div(
            parent_log_probs,
            implied_parent_log_probs.detach(),
            reduction="batchmean",
            log_target=True,
        )

        # KL(predicted parent || implied parent):
        # gradients update the child head only.
        child_loss = F.kl_div(
            implied_parent_log_probs,
            parent_log_probs.detach(),
            reduction="batchmean",
            log_target=True,
        )

        return 0.5 * (parent_loss + child_loss)
