from __future__ import annotations

import torch.nn as nn


DINOV3_ALIASES = {
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
    "dinov3_convnext_tiny": "convnext_tiny.eupe_lvd1689m",
    "dinov3_convnext_small": "convnext_small.eupe_lvd1689m",
    "dinov3_convnext_base": "convnext_base.eupe_lvd1689m",
    "dinov3_convnext_large": "convnext_large.eupe_lvd1689m",
}


def _load_model(name: str, pretrained: bool) -> nn.Module:
    try:
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError(
            "Torchvision models require torchvision; use a timm/DINOv3 model "
            "or install the training requirements."
        ) from exc
    try:
        constructor = getattr(models, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torchvision model: {name}") from exc

    if pretrained:
        try:
            return constructor(weights="DEFAULT")
        except TypeError:
            return constructor(pretrained=True)
    try:
        return constructor(weights=None)
    except TypeError:
        return constructor(pretrained=False)


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    provider: str = "auto",
    checkpoint_path: str | None = None,
) -> nn.Module:
    resolved_name = DINOV3_ALIASES.get(name, name)
    resolved_provider = provider.lower()
    if resolved_provider == "auto":
        resolved_provider = "timm" if "dinov3" in resolved_name else "torchvision"
    if resolved_provider == "timm":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "DINOv3/timm models require timm>=1.0.20"
            ) from exc
        kwargs = {
            "pretrained": pretrained and checkpoint_path is None,
            "num_classes": 0,
        }
        if checkpoint_path:
            kwargs["checkpoint_path"] = checkpoint_path
        model = timm.create_model(resolved_name, **kwargs)
        feature_dim = int(getattr(model, "num_features", 0))
        if feature_dim <= 0:
            raise ValueError(f"Could not determine timm feature size for {resolved_name!r}")
        model._fish_feature_dim = feature_dim
        model._fish_is_feature_backbone = True
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        return model
    if resolved_provider != "torchvision":
        raise ValueError("model.provider must be 'auto', 'torchvision', or 'timm'")
    model = _load_model(resolved_name, pretrained)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            for index in range(len(model.classifier) - 1, -1, -1):
                layer = model.classifier[index]
                if isinstance(layer, nn.Linear):
                    model.classifier[index] = nn.Linear(layer.in_features, num_classes)
                    return model
        if isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model

    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise ValueError(f"Do not know how to replace classification head for model: {name}")
