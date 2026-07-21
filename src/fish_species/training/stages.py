"""Stage-2 classifier initialization and architecture-aware freezing."""

from __future__ import annotations

import torch
from torch import nn

from ..models.long_tail import CosineClassifier
from ..models.multitask import MultiTaskClassifier


def initialise_species_classifier(
    model: MultiTaskClassifier, method: str, *, initial_scale: float = 20.0
) -> None:
    """Apply the configured fresh Stage-2 classifier initialization."""
    if method == "keep":
        return
    heads = (
        [model.natural_head, model.balanced_head]
        if model.dual_enabled
        else [model.species_head]
    )
    heads = [head for head in heads if head is not None]
    if method == "random":
        for head in heads:
            if hasattr(head, "reset_parameters"):
                head.reset_parameters()
            if isinstance(head, CosineClassifier):
                head.reset_scale(initial_scale)
        return
    if method != "prototype":
        raise ValueError("classifier_initialisation must be 'keep', 'random', or 'prototype'")
    if model.prototype_classifier is None:
        raise ValueError("prototype Stage-2 initialization requires available prototypes")
    prototypes = model.prototype_classifier.prototypes
    if not torch.any(model.prototype_classifier.counts > 0):
        raise ValueError("prototype Stage-2 initialization requires non-empty prototypes")
    for head in heads:
        weight = getattr(head, "weight", None)
        if weight is None or tuple(weight.shape) != tuple(prototypes.shape):
            raise ValueError(
                "prototype and species-classifier weight dimensions are incompatible"
            )
        with torch.no_grad():
            weight.copy_(prototypes.to(weight.device, weight.dtype))
        if isinstance(head, CosineClassifier):
            head.reset_scale(initial_scale)


def _final_backbone_block(backbone: nn.Module) -> nn.Module:
    """Resolve the final trainable stage for supported torchvision/timm families."""
    # torchvision ConvNeXt and timm ConvNeXt expose stages differently.
    if hasattr(backbone, "stages") and len(backbone.stages):
        return backbone.stages[-1]
    if hasattr(backbone, "features") and len(backbone.features):
        return backbone.features[-1]
    # timm ViT (including DINOv3).
    if hasattr(backbone, "blocks") and len(backbone.blocks):
        return backbone.blocks[-1]
    # torchvision ViT.
    encoder = getattr(backbone, "encoder", None)
    if encoder is not None and hasattr(encoder, "layers") and len(encoder.layers):
        return encoder.layers[-1]
    raise ValueError(
        "heads_and_last_block is unsupported for this backbone; supported "
        "families are torchvision ConvNeXt/ViT and timm DINOv3 ViT/ConvNeXt"
    )


def apply_stage2_trainable_scope(
    model: MultiTaskClassifier, scope: str
) -> tuple[int, int]:
    """Freeze parameters for Stage 2 and return trainable/frozen counts."""
    if scope not in {"heads", "heads_and_last_block", "full_model"}:
        raise ValueError(
            "trainable_scope must be 'heads', 'heads_and_last_block', or 'full_model'"
        )
    for parameter in model.parameters():
        parameter.requires_grad = scope == "full_model"
    if scope != "full_model":
        modules: list[nn.Module] = [model.heads, model.projection_heads]
        for module in (
            model.species_head,
            model.natural_head,
            model.balanced_head,
        ):
            if module is not None:
                modules.append(module)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        if scope == "heads_and_last_block":
            for parameter in _final_backbone_block(model.backbone).parameters():
                parameter.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def keep_frozen_backbone_eval(model: MultiTaskClassifier) -> None:
    """Prevent running-stat updates in every fully frozen backbone subtree."""
    for module in model.backbone.modules():
        parameters = tuple(module.parameters(recurse=True))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            module.eval()
