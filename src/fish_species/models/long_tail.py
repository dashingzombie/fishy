"""Long-tail classifier modules operating on shared backbone embeddings."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class CosineClassifier(nn.Module):
    """A cosine-similarity classifier with a bounded positive scale."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        initial_scale: float = 20.0,
        learnable_scale: bool = True,
        maximum_scale: float = 100.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_classes <= 0:
            raise ValueError("feature_dim and num_classes must be positive")
        if initial_scale <= 0 or maximum_scale <= 0:
            raise ValueError("cosine classifier scales must be positive")
        if initial_scale > maximum_scale:
            raise ValueError("initial_scale must not exceed maximum_scale")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.maximum_scale = float(maximum_scale)
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        log_scale = torch.tensor(math.log(float(initial_scale)), dtype=torch.float32)
        if learnable_scale:
            self.log_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("log_scale", log_scale)
        self.reset_parameters()

    @property
    def scale(self) -> torch.Tensor:
        """Return the positive, maximum-bounded classifier scale."""
        minimum_log = math.log(torch.finfo(torch.float32).tiny)
        maximum_log = math.log(self.maximum_scale)
        return self.log_scale.float().clamp(
            min=minimum_log, max=maximum_log
        ).exp()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.01)

    def reset_scale(self, initial_scale: float) -> None:
        """Restore a configured scale without replacing its parameter object."""
        if not 0 < initial_scale <= self.maximum_scale:
            raise ValueError("initial_scale must be in (0, maximum_scale]")
        with torch.no_grad():
            self.log_scale.fill_(math.log(float(initial_scale)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        device_type = features.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            normalized_features = F.normalize(features.float(), dim=1)
            normalized_weights = F.normalize(self.weight.float(), dim=1)
            return (normalized_features @ normalized_weights.t()) * self.scale


class ProjectionHead(nn.Module):
    """Two-layer normalized projection used by contrastive objectives."""

    def __init__(self, feature_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(features).float(), dim=1)


class PrototypeClassifier(nn.Module):
    """Checkpoint-persistent normalized class prototypes and logit fusion."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        *,
        update: str = "static",
        momentum: float = 0.99,
        scale: float = 20.0,
        fusion_enabled: bool = True,
        fusion_mode: str = "fixed",
        learned_weight: float = 0.5,
        prototype_strength: float = 10.0,
    ) -> None:
        super().__init__()
        if update not in {"static", "ema"}:
            raise ValueError("prototype update must be 'static' or 'ema'")
        if fusion_mode not in {"fixed", "frequency_dependent"}:
            raise ValueError("prototype fusion mode must be 'fixed' or 'frequency_dependent'")
        if not 0 <= momentum < 1:
            raise ValueError("prototype momentum must be in [0, 1)")
        if scale <= 0 or not 0 <= learned_weight <= 1 or prototype_strength < 0:
            raise ValueError("invalid prototype scale or fusion parameters")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.update_mode = update
        self.momentum = float(momentum)
        self.scale = float(scale)
        self.fusion_enabled = bool(fusion_enabled)
        self.fusion_mode = fusion_mode
        self.learned_weight = float(learned_weight)
        self.prototype_strength = float(prototype_strength)
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("counts", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def set_from_sums(self, sums: torch.Tensor, counts: torch.Tensor) -> None:
        """Install prototypes from class-wise normalized-feature sums."""
        sums = sums.to(device=self.prototypes.device, dtype=torch.float32)
        counts = counts.to(device=self.counts.device, dtype=torch.long)
        if sums.shape != self.prototypes.shape or counts.shape != self.counts.shape:
            raise ValueError("prototype sums/counts have incompatible shapes")
        valid = counts > 0
        updated = torch.zeros_like(self.prototypes)
        if valid.any():
            updated[valid] = F.normalize(
                sums[valid] / counts[valid, None].float(), dim=1
            )
        self.prototypes.copy_(updated)
        self.counts.copy_(counts)

    @torch.no_grad()
    def accumulate(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return distributed class-wise sums and counts for one collection."""
        normalized = F.normalize(features.float(), dim=1)
        sums = torch.zeros_like(self.prototypes, dtype=torch.float32)
        counts = torch.zeros_like(self.counts)
        sums.index_add_(0, labels, normalized)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
            dist.all_reduce(counts)
        return sums, counts

    @torch.no_grad()
    def rebuild(self, feature_batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Rebuild from an iterable of ``(features, species_labels)`` batches."""
        total_sums = torch.zeros_like(self.prototypes, dtype=torch.float32)
        total_counts = torch.zeros_like(self.counts)
        for features, labels in feature_batches:
            normalized = F.normalize(features.to(self.prototypes.device).float(), dim=1)
            labels = labels.to(self.prototypes.device)
            total_sums.index_add_(0, labels, normalized)
            total_counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_sums)
            dist.all_reduce(total_counts)
        self.set_from_sums(total_sums, total_counts)

    @torch.no_grad()
    def ema_update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Synchronously update only classes represented by the current batch."""
        sums, batch_counts = self.accumulate(features, labels)
        present = batch_counts > 0
        if not present.any():
            return
        means = torch.zeros_like(sums)
        means[present] = F.normalize(
            sums[present] / batch_counts[present, None].float(), dim=1
        )
        initialized = present & (self.counts > 0)
        new = present & ~initialized
        self.prototypes[new] = means[new]
        self.prototypes[initialized] = F.normalize(
            self.momentum * self.prototypes[initialized]
            + (1.0 - self.momentum) * means[initialized],
            dim=1,
        )
        self.counts.add_(batch_counts)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=features.device.type, enabled=False):
            normalized = F.normalize(features.float(), dim=1)
            prototypes = F.normalize(self.prototypes.float(), dim=1)
            return self.scale * (normalized @ prototypes.t())

    def fuse(
        self, learned_logits: torch.Tensor, prototype_logits: torch.Tensor
    ) -> torch.Tensor:
        """Fuse learned and prototype logits according to configured frequency policy."""
        if not self.fusion_enabled:
            return learned_logits
        if self.fusion_mode == "fixed":
            alpha: torch.Tensor | float = self.learned_weight
        else:
            counts = self.counts.to(device=learned_logits.device, dtype=torch.float32)
            alpha = counts / (counts + self.prototype_strength).clamp_min(1e-12)
        return alpha * learned_logits + (1.0 - alpha) * prototype_logits


def build_classifier(
    classifier_type: str,
    feature_dim: int,
    num_classes: int,
    config: dict | None = None,
) -> nn.Module:
    """Build a linear or cosine classifier from one normalized config block."""
    config = config or {}
    if classifier_type == "linear":
        return nn.Linear(feature_dim, num_classes)
    if classifier_type == "cosine":
        return CosineClassifier(
            feature_dim,
            num_classes,
            initial_scale=float(config.get("initial_scale", 20.0)),
            learnable_scale=bool(config.get("learnable_scale", True)),
            maximum_scale=float(config.get("maximum_scale", 100.0)),
        )
    raise ValueError("species classifier type must be 'linear' or 'cosine'")
