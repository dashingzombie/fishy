"""Taxonomic-distance metrics and memory-conscious minimum-risk inference."""

from __future__ import annotations

import numpy as np
import torch


def species_to_genus_indices(
    index_to_label_by_task: dict[str, dict[int, str]],
    child_to_parent: dict[str, str] | None = None,
) -> torch.Tensor:
    """Resolve every species index to an existing genus index."""
    from ..training.losses import infer_parent_label_from_child_label

    genus_labels = index_to_label_by_task.get("genus", {})
    species_labels = index_to_label_by_task.get("species", {})
    genus_to_index = {str(label): int(index) for index, label in genus_labels.items()}
    explicit = child_to_parent or {}
    mapping = torch.empty(len(species_labels), dtype=torch.long)
    missing: list[str] = []
    for index in range(len(species_labels)):
        species = str(species_labels[index])
        genus = explicit.get(species, infer_parent_label_from_child_label(species))
        if genus not in genus_to_index:
            missing.append(f"{species}->{genus}")
        else:
            mapping[index] = genus_to_index[genus]
    if missing:
        raise ValueError(
            "Invalid species-to-genus mapping; examples: " + ", ".join(missing[:10])
        )
    return mapping


def taxonomic_costs(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    species_to_genus: torch.Tensor,
    *,
    same_species_cost: float = 0.0,
    same_genus_cost: float = 1.0,
    different_genus_cost: float = 2.0,
) -> torch.Tensor:
    """Return per-example costs for the three-level species/genus taxonomy."""
    with torch.autocast(device_type=targets.device.type, enabled=False):
        mapping = species_to_genus.to(targets.device)
        correct = targets.eq(predictions)
        same_genus = mapping[targets].eq(mapping[predictions])
        costs = torch.full(
            targets.shape, float(different_genus_cost),
            device=targets.device, dtype=torch.float32,
        )
        costs[same_genus] = float(same_genus_cost)
        costs[correct] = float(same_species_cost)
        return costs


def minimum_taxonomic_risk_predictions(
    probabilities: torch.Tensor,
    species_to_genus: torch.Tensor,
    *,
    same_species_cost: float = 0.0,
    same_genus_cost: float = 1.0,
    different_genus_cost: float = 2.0,
) -> torch.Tensor:
    """Select minimum-risk species without allocating a species² cost matrix.

    For candidate ``k`` in genus ``g``, risk is
    ``C2 + (C1-C2) P(g|x) + (C0-C1) p(k|x)``.
    """
    with torch.autocast(device_type=probabilities.device.type, enabled=False):
        probs = probabilities.float()
        mapping = species_to_genus.to(probabilities.device)
        num_genera = int(mapping.max().item()) + 1
        genus_probs = probs.new_zeros((probs.shape[0], num_genera))
        genus_probs.scatter_add_(1, mapping[None, :].expand(probs.shape[0], -1), probs)
        risk = (
            float(different_genus_cost)
            + (float(same_genus_cost) - float(different_genus_cost))
            * genus_probs[:, mapping]
            + (float(same_species_cost) - float(same_genus_cost)) * probs
        )
        return risk.argmin(dim=1)


def taxonomic_metrics(
    targets: np.ndarray | torch.Tensor,
    predictions: np.ndarray | torch.Tensor,
    species_to_genus: torch.Tensor,
    *,
    same_species_cost: float = 0.0,
    same_genus_cost: float = 1.0,
    different_genus_cost: float = 2.0,
    prefix: str = "species_taxonomic",
) -> dict[str, float]:
    """Summarize raw or minimum-risk predictions using taxonomy-aware metrics."""
    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    prediction_tensor = torch.as_tensor(predictions, dtype=torch.long)
    if target_tensor.numel() == 0:
        return {
            f"{prefix}_mean_cost": float("nan"),
            f"{prefix}_median_cost": float("nan"),
            f"{prefix}_error_within_genus_fraction": float("nan"),
            f"{prefix}_species_accuracy": float("nan"),
            f"{prefix}_genus_accuracy": float("nan"),
        }
    costs = taxonomic_costs(
        target_tensor, prediction_tensor, species_to_genus,
        same_species_cost=same_species_cost,
        same_genus_cost=same_genus_cost,
        different_genus_cost=different_genus_cost,
    )
    mapping = species_to_genus.cpu()
    correct = target_tensor.eq(prediction_tensor)
    genus_correct = mapping[target_tensor].eq(mapping[prediction_tensor])
    errors = ~correct
    return {
        f"{prefix}_mean_cost": float(costs.mean().item()),
        f"{prefix}_median_cost": float(torch.quantile(costs, 0.5).item()),
        f"{prefix}_error_within_genus_fraction": (
            float(genus_correct[errors].float().mean().item())
            if errors.any() else 0.0
        ),
        f"{prefix}_species_accuracy": float(correct.float().mean().item()),
        f"{prefix}_genus_accuracy": float(genus_correct.float().mean().item()),
    }
