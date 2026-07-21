"""Hierarchy-aware epoch loop with opt-in long-tail objectives."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from ..evaluation.taxonomic import minimum_taxonomic_risk_predictions
from ..evaluation.taxonomic import taxonomic_metrics
from .contrastive import balanced_contrastive_loss
from .contrastive import hierarchical_contrastive_loss
from .distributed import unwrap_model
from .losses import hierarchy_consistency_loss
from .losses import infer_parent_label_from_child_label
from .metrics import safe_metric
from .stages import keep_frozen_backbone_eval


def _gather_prediction_dict(values: dict[str, list[int]]) -> dict[str, list[int]]:
    if not (dist.is_available() and dist.is_initialized()):
        return values
    gathered: list[dict[str, list[int]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, values)
    keys = set().union(*(rank_values.keys() for rank_values in gathered if rank_values))
    return {
        key: [
            item for rank_values in gathered if rank_values
            for item in rank_values.get(key, [])
        ]
        for key in keys
    }


def _amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def run_hierarchy_epoch(
    model: nn.Module,
    loader: DataLoader,
    criteria: dict[str, nn.Module],
    optimizer,
    device: torch.device,
    train: bool,
    scaler=None,
    use_amp: bool = True,
    task_loss_weights: dict[str, float] | None = None,
    normalize_loss_by_active_tasks: bool = True,
    hierarchy_cfg: dict | None = None,
    child_to_parent_matrix: torch.Tensor | None = None,
    metric_context: dict | None = None,
    long_tail_config: dict | None = None,
):
    """Run one epoch while preserving the legacy return and metric surface."""
    model.train(train)
    base_model = unwrap_model(model)
    if train and hasattr(base_model, "backbone"):
        keep_frozen_backbone_eval(base_model)
    tasks = list(criteria)
    task_loss_weights = task_loss_weights or {task: 1.0 for task in tasks}
    hierarchy_cfg = hierarchy_cfg or {}
    metric_context = metric_context or {}
    config = long_tail_config or {}
    multi_cfg = config.get("multi_task", {}) or {}
    dual_cfg = (config.get("training", {}) or {}).get("dual_species_classifier", {}) or {}
    hierarchical_cfg = multi_cfg.get("hierarchical_contrastive", {}) or {}
    balanced_cfg = multi_cfg.get("balanced_contrastive", {}) or {}
    amp_name = str((config.get("training", {}) or {}).get("amp_dtype", "float16"))

    hierarchy_parent_task = hierarchy_cfg.get("parent_task", "genus")
    hierarchy_child_task = hierarchy_cfg.get("child_task", "species")
    hierarchy_weight = float(hierarchy_cfg.get("weight", task_loss_weights.get("hierarchy", 0.1)))
    use_hierarchy_loss = (
        bool(hierarchy_cfg.get("enabled", False)) and hierarchy_weight > 0
        and child_to_parent_matrix is not None
        and hierarchy_parent_task in tasks and hierarchy_child_task in tasks
    )

    loss_values: list[float] = []
    component_values: dict[str, list[float]] = {task: [] for task in tasks}
    all_true = {task: [] for task in tasks}
    all_pred = {task: [] for task in tasks}
    head_predictions: dict[str, list[int]] = {}
    min_risk_predictions: list[int] = []
    top5_flags: list[int] = []
    top5_correct = top5_total = 0
    complete_correct_total = complete_total = 0
    hierarchy_prediction_correct = hierarchy_prediction_total = 0
    contrastive_counts = {
        "hierarchical_contrastive_valid_anchors": 0,
        "hierarchical_contrastive_positive_pairs": 0,
        "balanced_contrastive_valid_anchors": 0,
        "balanced_contrastive_classes": 0,
        "balanced_contrastive_image_positive_pairs": 0,
        "balanced_contrastive_prototype_positive_pairs": 0,
    }
    started = time.perf_counter()
    processed = 0
    index_to_label = metric_context.get("index_to_label_by_task", {}) or {}
    taxonomy_mapping = metric_context.get("species_to_genus")

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = {task: batch["labels"][task].to(device, non_blocking=True) for task in tasks}
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(
                enabled=use_amp and device.type == "cuda",
                device_type=device.type,
                dtype=_amp_dtype(amp_name),
            ):
                outputs = model(x)
                total_loss = torch.zeros((), device=device)
                active_weight_sum = 0.0
                batch_components: dict[str, torch.Tensor | None] = {}

                dual_enabled = "species_natural" in outputs
                for task in tasks:
                    if task == "species" and dual_enabled:
                        natural_criterion, balanced_criterion = metric_context["dual_criteria"]
                        natural_loss = natural_criterion(outputs["species_natural"], y[task])
                        balanced_loss = balanced_criterion(outputs["species_balanced"], y[task])
                        nw = float(dual_cfg.get("natural_loss_weight", 1.0))
                        bw = float(dual_cfg.get("balanced_loss_weight", 1.0))
                        total_loss = total_loss + nw * natural_loss + bw * balanced_loss
                        active_weight_sum += nw + bw
                        batch_components["species_natural"] = natural_loss
                        batch_components["species_balanced"] = balanced_loss
                        batch_components[task] = 0.5 * (natural_loss + balanced_loss)
                    else:
                        task_loss = criteria[task](outputs[task], y[task])
                        weight = float(task_loss_weights.get(task, 1.0))
                        total_loss = total_loss + weight * task_loss
                        active_weight_sum += weight
                        batch_components[task] = task_loss

                if use_hierarchy_loss:
                    value = hierarchy_consistency_loss(
                        outputs[hierarchy_parent_task], outputs[hierarchy_child_task],
                        child_to_parent_matrix, torch.ones_like(y[hierarchy_parent_task], dtype=torch.bool),
                    )
                    batch_components["hierarchy"] = value
                    if value is not None:
                        total_loss += hierarchy_weight * value
                        active_weight_sum += hierarchy_weight

                projected_by_dim: dict[int, torch.Tensor] = {}
                view2_outputs = None
                needs_contrast = bool(hierarchical_cfg.get("enabled", False) or balanced_cfg.get("enabled", False))
                if needs_contrast and "image_view2" in batch:
                    view2 = batch["image_view2"].to(device, non_blocking=True)
                    view2_outputs = model(view2)
                source_ids = torch.arange(x.shape[0], device=device)
                if dist.is_available() and dist.is_initialized():
                    source_ids += dist.get_rank() * max(1, x.shape[0])

                if bool(hierarchical_cfg.get("enabled", False)):
                    dim = int(hierarchical_cfg.get("projection_dim", 256))
                    embeddings = outputs[f"projection_{dim}"]
                    species_labels, genus_labels, sources = y["species"], y["genus"], source_ids
                    if view2_outputs is not None:
                        embeddings = torch.cat([embeddings, view2_outputs[f"projection_{dim}"]])
                        species_labels = torch.cat([species_labels, species_labels])
                        genus_labels = torch.cat([genus_labels, genus_labels])
                        sources = torch.cat([sources, sources])
                    value, stats = hierarchical_contrastive_loss(
                        embeddings, species_labels, genus_labels,
                        temperature=float(hierarchical_cfg.get("temperature", 0.1)),
                        same_species_weight=float(hierarchical_cfg.get("same_species_weight", 1.0)),
                        same_genus_weight=float(hierarchical_cfg.get("same_genus_weight", 0.25)),
                        source_ids=sources,
                    )
                    batch_components["hierarchical_contrastive"] = value
                    if value is not None:
                        weight = float(hierarchical_cfg.get("weight", 0.05))
                        total_loss += weight * value
                        active_weight_sum += weight
                    contrastive_counts["hierarchical_contrastive_valid_anchors"] += stats.valid_anchors
                    contrastive_counts["hierarchical_contrastive_positive_pairs"] += stats.positive_pairs

                if bool(balanced_cfg.get("enabled", False)):
                    dim = int(balanced_cfg.get("projection_dim", 256))
                    embeddings = outputs[f"projection_{dim}"]
                    species_labels, sources = y["species"], source_ids
                    if view2_outputs is not None:
                        embeddings = torch.cat([embeddings, view2_outputs[f"projection_{dim}"]])
                        species_labels = torch.cat([species_labels, species_labels])
                        sources = torch.cat([sources, sources])
                    prototype_embeddings = prototype_counts = None
                    prototype_module = getattr(base_model, "prototype_classifier", None)
                    if bool(balanced_cfg.get("include_class_prototypes", True)) and prototype_module is not None:
                        projection = base_model.projection_heads[str(dim)]
                        prototype_embeddings = projection(prototype_module.prototypes)
                        prototype_counts = prototype_module.counts
                    value, stats = balanced_contrastive_loss(
                        embeddings, species_labels,
                        temperature=float(balanced_cfg.get("temperature", 0.1)),
                        class_average=bool(balanced_cfg.get("class_average", True)),
                        source_ids=sources,
                        prototype_embeddings=prototype_embeddings,
                        prototype_counts=prototype_counts,
                    )
                    batch_components["balanced_contrastive"] = value
                    if value is not None:
                        weight = float(balanced_cfg.get("weight", 0.1))
                        total_loss += weight * value
                        active_weight_sum += weight
                    contrastive_counts["balanced_contrastive_valid_anchors"] += stats.valid_anchors
                    contrastive_counts["balanced_contrastive_classes"] += stats.classes
                    contrastive_counts["balanced_contrastive_image_positive_pairs"] += stats.image_positive_pairs
                    contrastive_counts["balanced_contrastive_prototype_positive_pairs"] += stats.prototype_positive_pairs

                if active_weight_sum == 0:
                    continue
                if normalize_loss_by_active_tasks:
                    total_loss = total_loss / active_weight_sum

            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
                prototype_module = getattr(base_model, "prototype_classifier", None)
                if prototype_module is not None and prototype_module.update_mode == "ema":
                    prototype_module.ema_update(outputs["features"].detach(), y["species"])

        processed += int(x.shape[0])
        loss_values.append(float(total_loss.detach().item()))
        for name, value in batch_components.items():
            if value is not None:
                component_values.setdefault(name, []).append(float(value.detach().item()))
        if train:
            continue  # Avoid training-time predictions and device-to-CPU copies.

        complete = torch.ones(x.shape[0], dtype=torch.bool, device=device)
        for task in tasks:
            pred = outputs[task].argmax(1)
            complete &= pred.eq(y[task])
            all_true[task].extend(y[task].cpu().tolist())
            all_pred[task].extend(pred.cpu().tolist())
        complete_correct_total += int(complete.sum().item())
        complete_total += int(x.shape[0])
        if "species" in tasks:
            species_logits = outputs["species"]
            top = species_logits.topk(min(5, species_logits.shape[1]), dim=1).indices
            top5_correct += int(top.eq(y["species"][:, None]).any(1).sum().item())
            top5_total += int(y["species"].numel())
            top5_flags.extend(top.eq(y["species"][:, None]).any(1).cpu().long().tolist())
            for key in (
                "species_learned", "species_prototype", "species_prototype_fused",
                "species_natural", "species_balanced", "species_dual_fused",
            ):
                if key in outputs:
                    head_predictions.setdefault(key, []).extend(outputs[key].argmax(1).cpu().tolist())
            if taxonomy_mapping is not None and bool(metric_context.get("minimum_risk", False)):
                probabilities = outputs["species"].float().softmax(1)
                min_risk_predictions.extend(minimum_taxonomic_risk_predictions(
                    probabilities, taxonomy_mapping,
                    **metric_context.get("taxonomy_costs", {}),
                ).cpu().tolist())
        if "species" in tasks and "genus" in tasks and index_to_label:
            genus_pred = outputs["genus"].argmax(1).cpu().tolist()
            species_pred = outputs["species"].argmax(1).cpu().tolist()
            for genus_index, species_index in zip(genus_pred, species_pred):
                implied = infer_parent_label_from_child_label(str(index_to_label["species"][species_index]))
                hierarchy_prediction_correct += int(str(index_to_label["genus"][genus_index]) == implied)
                hierarchy_prediction_total += 1

    elapsed = max(time.perf_counter() - started, 1e-12)
    if dist.is_available() and dist.is_initialized():
        processed_tensor = torch.tensor(float(processed), device=device)
        elapsed_tensor = torch.tensor(float(elapsed), device=device)
        dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        processed = int(processed_tensor.item())
        elapsed = float(elapsed_tensor.item())
        count_keys = list(contrastive_counts)
        count_tensor = torch.tensor(
            [contrastive_counts[key] for key in count_keys],
            device=device, dtype=torch.long,
        )
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        contrastive_counts = {
            key: int(value) for key, value in zip(count_keys, count_tensor.tolist())
        }
    metrics = {
        "loss": float(np.mean(loss_values)) if loss_values else float("nan"),
        "images_per_second": processed / elapsed,
        **{name + "_loss": float(np.mean(values)) for name, values in component_values.items() if values},
        **contrastive_counts,
    }
    if train:
        return metrics, all_true, all_pred

    all_true = _gather_prediction_dict(all_true)
    all_pred = _gather_prediction_dict(all_pred)
    head_predictions = _gather_prediction_dict(head_predictions)
    if "species" in tasks:
        top5_flags = _gather_prediction_dict({"value": top5_flags})["value"]
    if bool(metric_context.get("minimum_risk", False)):
        min_risk_predictions = _gather_prediction_dict({"value": min_risk_predictions})["value"]
    macro_values = []
    for task in tasks:
        truth = np.asarray(all_true[task], dtype=int)
        prediction = np.asarray(all_pred[task], dtype=int)
        metrics[f"{task}_n"] = int(len(truth))
        if not len(truth):
            for suffix in ("accuracy", "balanced_accuracy", "macro_f1"):
                metrics[f"{task}_{suffix}"] = float("nan")
            continue
        macro = f1_score(truth, prediction, average="macro", zero_division=0)
        macro_values.append(macro)
        metrics[f"{task}_accuracy"] = safe_metric(accuracy_score, truth, prediction)
        metrics[f"{task}_balanced_accuracy"] = safe_metric(balanced_accuracy_score, truth, prediction)
        metrics[f"{task}_macro_f1"] = float(macro)
    metrics["mean_macro_f1"] = float(np.mean(macro_values)) if macro_values else float("nan")
    if tasks and all_true[tasks[0]]:
        exact = np.ones(len(all_true[tasks[0]]), dtype=bool)
        for task in tasks:
            exact &= np.asarray(all_true[task]) == np.asarray(all_pred[task])
        metrics["complete_exact_match_accuracy"] = float(exact.mean())
        metrics["complete_exact_match_n"] = int(len(exact))
    else:
        metrics["complete_exact_match_accuracy"] = float("nan")
        metrics["complete_exact_match_n"] = 0
    if "species" in tasks:
        truth = np.asarray(all_true["species"], dtype=int)
        prediction = np.asarray(all_pred["species"], dtype=int)
        metrics["species_represented_species_count"] = int(
            len(np.unique(truth))
        )
        metrics["species_top1_accuracy"] = metrics.get("species_accuracy")
        metrics["species_top5_accuracy"] = float(np.mean(top5_flags)) if top5_flags else float("nan")
        for key, predictions in head_predictions.items():
            short = key.removeprefix("species_")
            values = np.asarray(predictions, dtype=int)
            metrics[f"species_{short}_accuracy"] = safe_metric(accuracy_score, truth, values)
            metrics[f"species_{short}_macro_f1"] = float(f1_score(truth, values, average="macro", zero_division=0))
        species_counts = metric_context.get("species_counts", {}) or {}
        species_names = index_to_label.get("species", {})
        buckets = {
            "few_shot_2_to_5": lambda count: 2 <= count <= 5,
            "medium_shot_6_to_20": lambda count: 6 <= count <= 20,
            "many_shot_over_20": lambda count: count > 20,
            "head_over_10": lambda count: count > 10,
            "tail_10_or_fewer": lambda count: 0 < count <= 10,
        }
        for name, predicate in buckets.items():
            selected = np.asarray([predicate(int(species_counts.get(str(species_names[int(v)]), 0))) for v in truth], dtype=bool) if species_counts and species_names else np.zeros(len(truth), bool)
            metrics[f"species_{name}_macro_f1"] = float(f1_score(truth[selected], prediction[selected], labels=np.unique(truth[selected]), average="macro", zero_division=0)) if selected.any() else float("nan")
            metrics[f"species_{name}_n_samples"] = int(selected.sum())
            metrics[f"species_{name}_n_species"] = int(len(np.unique(truth[selected]))) if selected.any() else 0
            metrics[f"species_{name}_accuracy"] = safe_metric(accuracy_score, truth[selected], prediction[selected]) if selected.any() else float("nan")
            metrics[f"species_{name}_balanced_accuracy"] = safe_metric(balanced_accuracy_score, truth[selected], prediction[selected]) if selected.any() else float("nan")
            if taxonomy_mapping is not None and selected.any():
                metrics.update(taxonomic_metrics(truth[selected], prediction[selected], taxonomy_mapping, prefix=f"species_{name}_taxonomic", **metric_context.get("taxonomy_costs", {})))
        if taxonomy_mapping is not None:
            metrics.update(taxonomic_metrics(truth, prediction, taxonomy_mapping, **metric_context.get("taxonomy_costs", {})))
            if min_risk_predictions:
                metrics.update(taxonomic_metrics(truth, np.asarray(min_risk_predictions), taxonomy_mapping, prefix="species_taxonomic_minimum_risk", **metric_context.get("taxonomy_costs", {})))
    if "species" in tasks and "genus" in tasks and index_to_label and all_pred["species"]:
        consistent = [
            str(index_to_label["genus"][g])
            == infer_parent_label_from_child_label(str(index_to_label["species"][s]))
            for g, s in zip(all_pred["genus"], all_pred["species"])
        ]
        metrics["genus_species_consistency"] = float(np.mean(consistent))
    else:
        metrics["genus_species_consistency"] = float("nan")
    return metrics, all_true, all_pred
