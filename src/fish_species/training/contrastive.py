"""Numerically stable contrastive objectives for long-tail representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class ContrastiveStats:
    """Counts used for diagnostics and W&B logging."""

    valid_anchors: int = 0
    positive_pairs: int = 0
    classes: int = 0
    image_positive_pairs: int = 0
    prototype_positive_pairs: int = 0


def gather_with_local_grad(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Gather equal-sized rank tensors while preserving local-rank gradients."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor, 0
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.detach())
    gathered[rank] = tensor
    return torch.cat(gathered, dim=0), rank * tensor.shape[0]


def _weighted_contrastive(
    local_embeddings: torch.Tensor,
    all_embeddings: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float,
    local_offset: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Compute a weighted multi-positive InfoNCE loss in float32."""
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    with torch.autocast(device_type=local_embeddings.device.type, enabled=False):
        local = F.normalize(local_embeddings.float(), dim=1)
        all_values = F.normalize(all_embeddings.float(), dim=1)
        similarities = local @ all_values.t() / float(temperature)
        denominator_mask = torch.ones_like(weights, dtype=torch.bool)
        rows = torch.arange(local.shape[0], device=local.device)
        denominator_mask[rows, rows + local_offset] = False
        positive_mask = weights > 0
        valid = positive_mask.any(dim=1)
        if not valid.any():
            return None, valid
        logits = similarities.masked_fill(~denominator_mask, -torch.inf)
        log_denominator = torch.logsumexp(logits, dim=1)
        normalized_weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        positive_log_prob = (normalized_weights * (
            similarities - log_denominator[:, None]
        ).masked_fill(~positive_mask, 0.0)).sum(dim=1)
        return -positive_log_prob[valid].mean(), valid


def hierarchical_contrastive_loss(
    embeddings: torch.Tensor,
    species_labels: torch.Tensor,
    genus_labels: torch.Tensor,
    *,
    temperature: float = 0.1,
    same_species_weight: float = 1.0,
    same_genus_weight: float = 0.25,
    source_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, ContrastiveStats]:
    """Contrast species strongly and genus relatives weakly without self-pairs."""
    all_embeddings, offset = gather_with_local_grad(embeddings)
    all_species, _ = gather_with_local_grad(species_labels)
    all_genus, _ = gather_with_local_grad(genus_labels)
    all_sources = None
    if source_ids is not None:
        all_sources, _ = gather_with_local_grad(source_ids)
    same_species = species_labels[:, None].eq(all_species[None, :])
    same_genus = genus_labels[:, None].eq(all_genus[None, :])
    weights = torch.zeros_like(same_species, dtype=torch.float32)
    weights[same_genus] = float(same_genus_weight)
    weights[same_species] = float(same_species_weight)
    if source_ids is not None and all_sources is not None:
        same_source = source_ids[:, None].eq(all_sources[None, :])
        weights[same_source] = max(1.0, float(same_species_weight)) + 1.0
    rows = torch.arange(embeddings.shape[0], device=embeddings.device)
    weights[rows, rows + offset] = 0
    loss, valid = _weighted_contrastive(
        embeddings, all_embeddings, weights,
        temperature=temperature, local_offset=offset,
    )
    return loss, ContrastiveStats(
        valid_anchors=int(valid.sum().item()),
        positive_pairs=int((weights > 0).sum().item()),
        classes=int(species_labels.unique().numel()),
    )


def balanced_contrastive_loss(
    embeddings: torch.Tensor,
    species_labels: torch.Tensor,
    *,
    temperature: float = 0.1,
    class_average: bool = True,
    source_ids: torch.Tensor | None = None,
    prototype_embeddings: torch.Tensor | None = None,
    prototype_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, ContrastiveStats]:
    """Species contrastive loss with per-anchor positive and per-class balancing."""
    all_embeddings, offset = gather_with_local_grad(embeddings)
    all_labels, _ = gather_with_local_grad(species_labels)
    image_positive = species_labels[:, None].eq(all_labels[None, :])
    rows = torch.arange(embeddings.shape[0], device=embeddings.device)
    image_positive[rows, rows + offset] = False

    candidates = all_embeddings
    weights = image_positive.float()
    prototype_pairs = 0
    if prototype_embeddings is not None:
        prototype_embeddings = prototype_embeddings.to(embeddings.device)
        if prototype_counts is None:
            available = torch.ones(
                prototype_embeddings.shape[0], dtype=torch.bool,
                device=embeddings.device,
            )
        else:
            available = prototype_counts.to(embeddings.device) > 0
        prototype_mask = species_labels[:, None].eq(
            torch.arange(prototype_embeddings.shape[0], device=embeddings.device)[None, :]
        ) & available[None, :]
        prototype_pairs = int(prototype_mask.sum().item())
        candidates = torch.cat([all_embeddings, prototype_embeddings], dim=0)
        weights = torch.cat([weights, prototype_mask.float()], dim=1)

    if class_average:
        # Equalize anchors by inverse local class frequency, then renormalize.
        class_counts = torch.bincount(all_labels).clamp_min(1)
        anchor_weights = class_counts[species_labels].float().reciprocal()
    else:
        anchor_weights = torch.ones_like(species_labels, dtype=torch.float32)

    loss, valid = _weighted_contrastive(
        embeddings, candidates, weights,
        temperature=temperature, local_offset=offset,
    )
    if loss is not None and class_average:
        # Recompute the reduction explicitly so class-balanced anchor weighting
        # does not change positive normalization within an anchor.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            local = F.normalize(embeddings.float(), dim=1)
            cand = F.normalize(candidates.float(), dim=1)
            sim = local @ cand.t() / float(temperature)
            denominator = torch.ones_like(weights, dtype=torch.bool)
            denominator[rows, rows + offset] = False
            log_den = torch.logsumexp(sim.masked_fill(~denominator, -torch.inf), dim=1)
            positive = weights > 0
            normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            per_anchor = -(normalized * (sim - log_den[:, None]).masked_fill(
                ~positive, 0.0
            )).sum(dim=1)
            selected_weights = anchor_weights[valid]
            loss = (per_anchor[valid] * selected_weights).sum() / selected_weights.sum()
    return loss, ContrastiveStats(
        valid_anchors=int(valid.sum().item()),
        positive_pairs=int((weights > 0).sum().item()),
        classes=int(species_labels.unique().numel()),
        image_positive_pairs=int(image_positive.sum().item()),
        prototype_positive_pairs=prototype_pairs,
    )
