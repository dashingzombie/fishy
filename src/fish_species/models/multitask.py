from __future__ import annotations

import torch
from torch import nn

from .factory import build_model
from .long_tail import CosineClassifier, ProjectionHead, PrototypeClassifier
from .long_tail import build_classifier


class MultiTaskClassifier(nn.Module):
    """Shared image backbone with one classification head per task."""

    def __init__(
        self,
        base_model: nn.Module,
        num_classes_by_task: dict[str, int],
        config: dict | None = None,
    ) -> None:
        super().__init__()
        config = config or {}
        self.backbone = base_model
        self.feature_dim = self._remove_classifier_and_get_feature_dim()
        model_cfg = config.get("model", {}) or {}
        species_cfg = model_cfg.get("species_classifier", {}) or {}
        species_type = str(species_cfg.get("type", "linear"))
        dual_cfg = model_cfg.get("dual_species_classifier", {}) or {}
        self.dual_enabled = bool(dual_cfg.get("enabled", False))
        self.heads = nn.ModuleDict()
        for task, num_classes in num_classes_by_task.items():
            if task != "species":
                self.heads[task] = nn.Linear(self.feature_dim, num_classes)

        self.species_head: nn.Module | None = None
        self.natural_head: nn.Module | None = None
        self.balanced_head: nn.Module | None = None
        if "species" in num_classes_by_task:
            species_classes = num_classes_by_task["species"]
            if self.dual_enabled:
                dual_type = str(dual_cfg.get("classifier_type", "cosine"))
                head_cfg = species_cfg if dual_type == "cosine" else {}
                self.natural_head = build_classifier(
                    dual_type, self.feature_dim, species_classes, head_cfg
                )
                self.balanced_head = build_classifier(
                    dual_type, self.feature_dim, species_classes, head_cfg
                )
            else:
                self.species_head = build_classifier(
                    species_type, self.feature_dim, species_classes, species_cfg
                )

        prototype_cfg = model_cfg.get("prototype_classifier", {}) or {}
        self.prototype_classifier: PrototypeClassifier | None = None
        if bool(prototype_cfg.get("enabled", False)):
            fusion = prototype_cfg.get("fusion", {}) or {}
            self.prototype_classifier = PrototypeClassifier(
                self.feature_dim,
                num_classes_by_task["species"],
                update=str(prototype_cfg.get("update", "static")),
                momentum=float(prototype_cfg.get("momentum", 0.99)),
                scale=float(prototype_cfg.get("scale", 20.0)),
                fusion_enabled=bool(fusion.get("enabled", True)),
                fusion_mode=str(fusion.get("mode", "fixed")),
                learned_weight=float(fusion.get("learned_weight", 0.5)),
                prototype_strength=float(fusion.get("prototype_strength", 10.0)),
            )

        hierarchical = (config.get("multi_task", {}) or {}).get(
            "hierarchical_contrastive", {}
        ) or {}
        balanced = (config.get("multi_task", {}) or {}).get(
            "balanced_contrastive", {}
        ) or {}
        projection_dims = {
            int(block.get("projection_dim", 256))
            for block in (hierarchical, balanced)
            if bool(block.get("enabled", False))
        }
        self.projection_heads = nn.ModuleDict({
            str(dim): ProjectionHead(self.feature_dim, dim)
            for dim in sorted(projection_dims)
        })
        inference_cfg = dual_cfg.get("inference", {}) or {}
        self.dual_inference_mode = str(inference_cfg.get("mode", "fused"))
        self.dual_natural_weight = float(inference_cfg.get("natural_weight", 0.5))
        self.dual_frequency_dependent = bool(
            inference_cfg.get("frequency_dependent", False)
        )
        self.register_buffer(
            "species_class_counts",
            torch.zeros(num_classes_by_task.get("species", 0), dtype=torch.long),
        )

    def _remove_classifier_and_get_feature_dim(self) -> int:
        if bool(getattr(self.backbone, "_fish_is_feature_backbone", False)):
            return int(self.backbone._fish_feature_dim)
        if hasattr(self.backbone, "fc") and isinstance(self.backbone.fc, nn.Linear):
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            return feature_dim

        if hasattr(self.backbone, "classifier"):
            classifier = self.backbone.classifier
            if isinstance(classifier, nn.Linear):
                feature_dim = classifier.in_features
                self.backbone.classifier = nn.Identity()
                return feature_dim
            if isinstance(classifier, nn.Sequential):
                for index in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[index], nn.Linear):
                        feature_dim = classifier[index].in_features
                        classifier[index] = nn.Identity()
                        return feature_dim

        if hasattr(self.backbone, "heads") and hasattr(self.backbone.heads, "head"):
            head = self.backbone.heads.head
            if isinstance(head, nn.Linear):
                feature_dim = head.in_features
                self.backbone.heads.head = nn.Identity()
                return feature_dim

        raise ValueError(
            "Could not identify the final classifier layer. "
            "Add a case in MultiTaskClassifier._remove_classifier_and_get_feature_dim "
            "for your model."
        )

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the shared, unprojected backbone embedding."""
        features = self.backbone(inputs)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(
                torch.nn.functional.adaptive_avg_pool2d(features, 1), 1
            )
        return features

    def _dual_fusion(
        self, natural: torch.Tensor, balanced: torch.Tensor
    ) -> torch.Tensor:
        if self.dual_inference_mode == "natural":
            return natural
        if self.dual_inference_mode == "balanced":
            return balanced
        if self.dual_frequency_dependent:
            counts = self.species_class_counts.to(natural.device, torch.float32)
            positive = counts[counts > 0]
            pivot = positive.median() if positive.numel() else counts.new_tensor(1.0)
            # Monotonic in frequency: rare classes receive less natural-head weight.
            alpha = counts / (counts + pivot.clamp_min(1.0))
        else:
            alpha = self.dual_natural_weight
        return alpha * natural + (1.0 - alpha) * balanced

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(inputs)
        outputs = {task: head(features) for task, head in self.heads.items()}
        outputs["features"] = features
        learned: torch.Tensor | None = None
        if self.dual_enabled:
            assert self.natural_head is not None and self.balanced_head is not None
            natural = self.natural_head(features)
            balanced = self.balanced_head(features)
            learned = self._dual_fusion(natural, balanced)
            outputs.update(
                species_natural=natural,
                species_balanced=balanced,
                species_dual_fused=learned,
            )
        elif self.species_head is not None:
            learned = self.species_head(features)
        if learned is not None:
            outputs["species_learned"] = learned
            species_logits = learned
            if self.prototype_classifier is not None:
                prototype = self.prototype_classifier(features)
                outputs["species_prototype"] = prototype
                species_logits = self.prototype_classifier.fuse(learned, prototype)
                outputs["species_prototype_fused"] = species_logits
            outputs["species"] = species_logits
        for dim, head in self.projection_heads.items():
            outputs[f"projection_{dim}"] = head(features)
        return outputs

    @torch.no_grad()
    def rebuild_prototypes(
        self, loader, device: torch.device, *, use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Rebuild prototypes from a labelled training DataLoader only."""
        if self.prototype_classifier is None:
            raise ValueError("prototype classifier is not enabled")
        was_training = self.training
        self.eval()
        sums = torch.zeros_like(self.prototype_classifier.prototypes)
        counts = torch.zeros_like(self.prototype_classifier.counts)
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"]["species"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
                dtype=amp_dtype,
            ):
                features = self.forward_features(images)
            normalized = torch.nn.functional.normalize(features.float(), dim=1)
            sums.index_add_(0, labels, normalized)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(counts)
        self.prototype_classifier.set_from_sums(sums, counts)
        self.train(was_training)

    def learned_species_head(self) -> nn.Module:
        """Return the primary learned species head for Stage 2 initialization."""
        if self.dual_enabled:
            assert self.natural_head is not None
            return self.natural_head
        if self.species_head is None:
            raise ValueError("model has no species head")
        return self.species_head


def build_multitask_model(
    cfg: dict,
    num_classes_by_task: dict[str, int],
) -> nn.Module:
    temporary_num_classes = max(num_classes_by_task.values())
    base_model = build_model(
        name=cfg["model"]["name"],
        num_classes=temporary_num_classes,
        pretrained=cfg["model"].get("pretrained", True),
        freeze_backbone=cfg["model"].get("freeze_backbone", False),
        provider=cfg["model"].get("provider", "auto"),
        checkpoint_path=cfg["model"].get("checkpoint_path"),
    )
    return MultiTaskClassifier(
        base_model=base_model,
        num_classes_by_task=num_classes_by_task,
        config=cfg,
    )
