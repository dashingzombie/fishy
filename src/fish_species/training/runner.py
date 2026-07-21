"""Single-run canonical training lifecycle. This module never expands sweeps."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from ..evaluation.condition_matrix import evaluate_condition_matrix
from ..evaluation.cue_suppression import evaluate_test_cue_suppression
from ..logging import create_wandb_logger
from ..models.multitask import build_multitask_model
from ..results.writing import save_json
from .checkpoints import build_checkpoint_payload
from .checkpoints import load_checkpoint
from .checkpoints import load_model_state_compat
from .checkpoints import save_checkpoint
from .epochs import run_hierarchy_epoch
from .distributed import barrier
from .distributed import finalise_distributed
from .distributed import initialise_distributed
from .distributed import set_loader_epoch
from .distributed import unwrap_model
from .distributed import wrap_model
from .loaders import get_input_condition
from .loaders import make_profile_loaders
from .losses import build_child_to_parent_matrix
from .losses import build_criteria
from .losses import build_dual_species_criteria
from .stages import apply_stage2_trainable_scope
from .stages import initialise_species_classifier
from ..evaluation.taxonomic import species_to_genus_indices
from .metrics import score_for_selection
from .modes import TrainingProfile
from .modes import resolved_run_name
from .modes import stress_evaluation_enabled
from .prediction import export_unlabeled_predictions
from .reproducibility import set_seed

def initialise_wandb_run(
    cfg: dict,
    run_name: str,
    out_dir: Path,
    profile: TrainingProfile,
):
    return create_wandb_logger(cfg, run_name, out_dir, profile).run


def make_experiment_run_name(cfg: dict, profile: TrainingProfile) -> str:
    pipeline_run = cfg.get("pipeline_run", {}) or {}
    if pipeline_run.get("run_id"):
        return str(pipeline_run["run_id"])
    return resolved_run_name(cfg, profile)


def _write_pipeline_result_files(
    cfg: dict,
    out_dir: Path,
    validation_metrics: dict,
    test_metrics: dict,
    *,
    best_checkpoint: Path,
    best_epoch: int,
    selection_metric: str,
) -> None:
    """Write the strict local result contract consumed by sweep pipelines."""
    pipeline_run = cfg.get("pipeline_run", {}) or {}
    if not pipeline_run.get("configuration_hash"):
        return
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(validation_metrics, metrics_dir / "validation_summary.json")
    save_json(test_metrics, metrics_dir / "test_summary.json")
    save_json(
        {
            "status": "completed",
            "exit_code": 0,
            "configuration_hash": str(pipeline_run["configuration_hash"]),
            "best_checkpoint": str(best_checkpoint.resolve()),
            "best_epoch": int(best_epoch),
            "selection_metric": selection_metric,
            "selection_value": validation_metrics.get(selection_metric),
        },
        out_dir / "run_status.json",
    )


def _metric_context(bundle, cfg: dict, dual_criteria=None) -> dict:
    context = {
        "index_to_label_by_task": bundle.index_to_label_by_task,
        "species_counts": bundle.species_counts or {},
    }
    taxonomy = (cfg.get("evaluation", {}) or {}).get("taxonomic_distance", {}) or {}
    minimum_risk = bool(((cfg.get("inference", {}) or {}).get(
        "taxonomic_minimum_risk", {}
    ) or {}).get("enabled", False))
    if bool(taxonomy.get("enabled", False)) or minimum_risk:
        context["species_to_genus"] = species_to_genus_indices(
            bundle.index_to_label_by_task,
            ((cfg.get("multi_task", {}) or {}).get("hierarchy_loss", {}) or {}).get("child_to_parent"),
        )
        context["taxonomy_costs"] = {
            "same_species_cost": float(taxonomy.get("same_species_cost", 0.0)),
            "same_genus_cost": float(taxonomy.get("same_genus_cost", 1.0)),
            "different_genus_cost": float(taxonomy.get("different_genus_cost", 2.0)),
        }
        context["minimum_risk"] = minimum_risk
    if dual_criteria is not None:
        context["dual_criteria"] = dual_criteria
    return context


def _build_optimizer(cfg: dict, model: torch.nn.Module):
    name = str(
        (cfg.get("training", {}).get("optimizer", {}) or {}).get("name", "adamw")
    ).lower()
    if name != "adamw":
        raise ValueError("training.optimizer.name must be 'adamw'")
    fused = bool((cfg.get("training", {}).get("optimizer", {}) or {}).get("fused", False))
    return torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        fused=fused and torch.cuda.is_available(),
    )


def _long_tail_checkpoint_metadata(
    model: torch.nn.Module, cfg: dict, staged_cfg: dict
) -> dict:
    """Build explicit long-tail checkpoint metadata alongside model state."""
    module = unwrap_model(model)
    species_cfg = ((cfg.get("model", {}) or {}).get("species_classifier", {}) or {})
    prototype_cfg = ((cfg.get("model", {}) or {}).get("prototype_classifier", {}) or {})
    scales: dict[str, float] = {}
    for name in ("species_head", "natural_head", "balanced_head"):
        head = getattr(module, name, None)
        if head is not None and hasattr(head, "scale"):
            scales[name] = float(head.scale.detach().cpu().item())
    prototype = getattr(module, "prototype_classifier", None)
    return {
        "species_classifier_type": str(species_cfg.get("type", "linear")),
        "cosine_scale": scales,
        "prototypes": prototype.prototypes.detach().cpu() if prototype is not None else None,
        "prototype_counts": prototype.counts.detach().cpu() if prototype is not None else None,
        "prototype_update_mode": str(prototype_cfg.get("update", "static")) if prototype is not None else None,
        "dual_species_heads": bool(getattr(module, "dual_enabled", False)),
        "projection_head_dimensions": sorted(getattr(module, "projection_heads", {}).keys()),
        "stage2_classifier_initialisation": str(staged_cfg.get("classifier_initialisation", "keep")),
        "stage2_trainable_scope": str(staged_cfg.get("trainable_scope", "full_model")),
        "taxonomic_inference": ((cfg.get("inference", {}) or {}).get("taxonomic_minimum_risk", {}) or {}),
    }


def run_test_evaluation(
    *,
    checkpoint_name: str,
    checkpoint_path: Path,
    write_legacy_outputs: bool,
    run_name: str,
    out_dir: Path,
    model: torch.nn.Module,
    bundle,
    criteria,
    device: torch.device,
    use_amp: bool,
    weights: dict[str, float],
    normalize: bool,
    hierarchy_cfg: dict,
    matrix: torch.Tensor | None,
    wandb_logger,
    input_condition: dict,
    cfg: dict,
    metric_context: dict | None = None,
) -> tuple[dict, dict[str, list[int]], dict[str, list[int]]]:
    """Evaluate one checkpoint on the test split and save its outputs."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    missing, ignored = load_model_state_compat(unwrap_model(model), checkpoint["model_state"])
    if missing:
        print("Newly initialized checkpoint modules: " + ", ".join(missing))
    if ignored:
        print("Checkpoint parameters not used by this architecture: " + ", ".join(ignored))

    test_metrics, true, pred = run_hierarchy_epoch(
        model,
        bundle.test_loader,
        criteria,
        None,
        device,
        False,
        None,
        use_amp,
        weights,
        normalize,
        hierarchy_cfg,
        matrix,
        metric_context or _metric_context(bundle, cfg),
        cfg,
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return test_metrics, true, pred

    wandb_condition = (
       f"original_{checkpoint_name}"
    )

    for task in bundle.target_cols:
        labels = list(range(len(bundle.index_to_label_by_task[task])))
        names = [bundle.index_to_label_by_task[task][index] for index in labels]
        y_true = np.asarray(true[task], dtype=int)
        y_pred = np.asarray(pred[task], dtype=int)

        report_path = (
            out_dir / f"classification_report_{checkpoint_name}_{task}.csv"
        )
        if not len(y_true):
            empty_report = pd.DataFrame(
                [{"note": "No labelled test examples for this task."}]
            )
            empty_report.to_csv(report_path, index=False)
            if write_legacy_outputs:
                empty_report.to_csv(
                    out_dir / f"classification_report_{task}.csv",
                    index=False,
                )
            continue

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0,
        )
        report_frame = pd.DataFrame(report).transpose()

        report_frame.to_csv(report_path)

        if write_legacy_outputs:
            report_frame.to_csv(out_dir / f"classification_report_{task}.csv")

        wandb_logger.log_classification_report(
            condition=wandb_condition,
            task=task,
            report=report,
            metrics=test_metrics,
            train_condition=input_condition,
        )

    metrics_path = out_dir / f"test_metrics_{checkpoint_name}.json"
    save_json(test_metrics, metrics_path)
    if write_legacy_outputs:
        save_json(test_metrics, out_dir / "test_metrics.json")

    wandb_logger.log_test_condition(
        wandb_condition,
        test_metrics,
        train_condition=input_condition,
    )
    print(
        f"[{run_name}] {checkpoint_name.capitalize()} checkpoint test "
        "species accuracy: "
        f"{test_metrics.get('species_accuracy', float('nan')):.4f}"
    )
    return test_metrics, true, pred


def run_one(cfg: dict, profile: TrainingProfile) -> dict:
    """Run exactly one resolved configuration; never generate another config."""
    distributed = initialise_distributed(cfg)
    set_seed(int(cfg["seed"]) + distributed.rank)
    device_name = (
        torch.cuda.get_device_name(distributed.local_rank)
        if torch.cuda.is_available()
        else "CPU"
    )
    print(f"Using device: {device_name}")
    print("Starting training")
    device = distributed.device

    input_condition = {
        "condition": "original",
        "feature": "baseline",
        "transform": "original",
        "strength": 0.0,
    }
    if profile.loader_mode == "condition":
        input_condition = get_input_condition(cfg)
        stress_enabled = stress_evaluation_enabled(cfg)
        if stress_enabled and input_condition["transform"] != "original":
            raise ValueError(
                "Fixed-RGB stress evaluation requires an original-trained "
                "input condition"
            )

    if profile.loader_mode == "condition":
        print(
            "Matched train/validation/test condition: "
            f"{input_condition['condition']} ({input_condition['transform']})"
        )

    run_name = make_experiment_run_name(cfg, profile)
    out_dir = Path(cfg["output"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if distributed.is_main:
        save_json(cfg, out_dir / "config.json")

    logger_cfg = cfg
    if not distributed.is_main:
        logger_cfg = copy.deepcopy(cfg)
        logger_cfg.setdefault("wandb", {})["enabled"] = False
    wandb_logger = create_wandb_logger(logger_cfg, run_name, out_dir, profile)
    bundle = make_profile_loaders(cfg, profile)
    if distributed.is_main:
        save_json(bundle.split_summary, out_dir / "split_summary.json")
        save_json(
            bundle.label_to_index_by_task,
            out_dir / "label_to_index_by_task.json",
        )
        print(f"Split summary and label maps saved to {out_dir}")

    num_classes_by_task = {
        task: len(label_to_index)
        for task, label_to_index in bundle.label_to_index_by_task.items()
    }
    model = build_multitask_model(
        cfg,
        num_classes_by_task,
    ).to(device)
    fine_tuning = cfg.get("fine_tuning", {}) or {}
    resume_path = None
    if bool(fine_tuning.get("enabled", False)):
        resume_path = Path(str(fine_tuning["checkpoint_path"]))
        resumed = load_checkpoint(resume_path, map_location=device)
        resumed_model = (resumed.get("cfg", {}).get("model", {}) or {}).get("name")
        current_model = (cfg.get("model", {}) or {}).get("name")
        if resumed_model and resumed_model != current_model:
            raise ValueError(
                "Fine-tuning checkpoint architecture does not match the current "
                f"model: checkpoint={resumed_model!r}, config={current_model!r}. "
                "Pass MODEL=<checkpoint model> to the Make target."
            )
        missing, ignored = load_model_state_compat(model, resumed["model_state"])
        if missing:
            print("Newly initialized fine-tuning modules: " + ", ".join(missing))
        if ignored:
            print("Fine-tuning parameters not used: " + ", ".join(ignored))
        print(f"Resumed model weights from {resume_path}")
    base_model = model
    species_counts_tensor = torch.zeros_like(base_model.species_class_counts)
    for label, index in bundle.label_to_index_by_task.get("species", {}).items():
        species_counts_tensor[index] = int((bundle.species_counts or {}).get(label, 0))
    base_model.species_class_counts.copy_(species_counts_tensor.to(device))
    prototype_cfg = (cfg.get("model", {}) or {}).get("prototype_classifier", {}) or {}
    if bool(prototype_cfg.get("enabled", False)):
        if bundle.prototype_loader is None:
            raise ValueError("prototype use requires the labelled training subset")
        amp_dtype = torch.bfloat16 if cfg["training"].get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
        base_model.rebuild_prototypes(
            bundle.prototype_loader, device,
            use_amp=bool(cfg["training"].get("use_amp", True)),
            amp_dtype=amp_dtype,
        )
        print("Built species prototypes from the labelled training subset")
    compile_cfg = cfg.get("training", {}).get("compile", {}) or {}
    if bool(compile_cfg.get("enabled", False)):
        if not hasattr(torch, "compile"):
            raise RuntimeError("training.compile.enabled requires torch.compile")
        model = torch.compile(model, mode=str(compile_cfg.get("mode", "default")))
        print("torch.compile applied before DistributedDataParallel")
    model = wrap_model(model, distributed)
    print("Model built and moved to device.")

    staged_cfg = (cfg.get("long_tail", {}) or {}).get("staged_training", {}) or {}
    staged_training = bool(staged_cfg.get("enabled", False))
    initial_criteria_df = (
        bundle.stage_one_train_df
        if staged_training and bundle.stage_one_train_df is not None
        else bundle.train_df
    )
    criteria = build_criteria(
        initial_criteria_df,
        bundle.target_cols,
        cfg["data"]["group_col"],
        bundle.label_to_index_by_task,
        device,
        use_class_weights=cfg.get("training", {}).get("class_weight", False),
        class_weighting=cfg.get("training", {}).get(
            "class_weighting", {}
        ),
        logit_adjustment=cfg.get("training", {}).get(
            "logit_adjustment", {}
        ),
    )
    weights = cfg.get("multi_task", {}).get(
        "loss_weights",
        {task: 1.0 for task in bundle.target_cols},
    )
    normalize = cfg.get("multi_task", {}).get(
        "normalize_loss_by_active_tasks", True
    )
    hierarchy_cfg = (
        cfg.get("multi_task", {}).get("hierarchy_loss", {})
        if profile.hierarchy
        else {}
    )
    matrix = None
    if hierarchy_cfg.get("enabled", False):
        parent_task = hierarchy_cfg.get("parent_task", "genus")
        child_task = hierarchy_cfg.get("child_task", "species")
        matrix = build_child_to_parent_matrix(
            bundle.label_to_index_by_task,
            parent_task,
            child_task,
            device,
            hierarchy_cfg.get("child_to_parent"),
        )
        print(
            f"Using hierarchy loss: {child_task} -> {parent_task} with weight "
            f"{hierarchy_cfg.get('weight', weights.get('hierarchy', 0.1))}"
        )

    dual_criteria = None
    if bool((((cfg.get("model", {}) or {}).get("dual_species_classifier", {}) or {}).get("enabled", False))):
        species_col = bundle.target_cols["species"]
        dual_criteria = build_dual_species_criteria(
            initial_criteria_df,
            species_col,
            bundle.label_to_index_by_task["species"],
            cfg["data"]["group_col"],
            device,
            (cfg.get("training", {}).get("dual_species_classifier", {}) or {}),
        )

    optimizer = _build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )
    use_amp = cfg["training"].get("use_amp", True)
    scaler = torch.amp.GradScaler(
        enabled=(use_amp and device.type == "cuda" and cfg["training"].get("amp_dtype", "bfloat16") == "float16")
    )

    early = cfg.get("early_stopping", {})
    early_enabled = early.get("enabled", True)
    patience = early.get("patience", 3)
    min_delta = early.get("min_delta", 0.001)
    best = -float("inf")
    best_epoch = 0
    best_selection_metrics: dict = {}
    stale = 0
    history = []
    selection = cfg.get("multi_task", {}).get(
        "selection_metric", "mean_macro_f1"
    )
    interval = cfg["training"].get("val_interval", 3)
    selection_loader = (
        bundle.head_val_loader
        if staged_training and bundle.head_val_loader is not None
        else bundle.val_loader
    )
    stage_one_checkpoint = (
        out_dir / "stage1_best_model.pt" if staged_training else out_dir / "best_model.pt"
    )
    metric_context = _metric_context(bundle, cfg, dual_criteria)
    stage_one_last_epoch = 0

    if str((cfg.get("pipeline_run", {}) or {}).get("execution_mode", "train")) == "evaluation_only":
        if resume_path is None:
            raise ValueError("evaluation-only pipeline runs require a parent checkpoint")
        validation_metrics = run_hierarchy_epoch(
            model, bundle.val_loader, criteria, None, device, False, None,
            use_amp, weights, normalize, hierarchy_cfg, matrix, metric_context, cfg,
        )[0]
        test_metrics = run_hierarchy_epoch(
            model, bundle.test_loader, criteria, None, device, False, None,
            use_amp, weights, normalize, hierarchy_cfg, matrix, metric_context, cfg,
        )[0]
        finalise_distributed(distributed)
        if not distributed.is_main:
            return {"run_name": run_name, "worker_rank": distributed.rank}
        selection = cfg.get("multi_task", {}).get("selection_metric", "mean_macro_f1")
        _write_pipeline_result_files(
            cfg, out_dir, validation_metrics, test_metrics,
            best_checkpoint=resume_path, best_epoch=0,
            selection_metric=selection,
        )
        result = {
            "run_name": run_name, "out_dir": str(out_dir),
            "evaluation_only": True,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        if profile.run_summary:
            save_json(result, out_dir / "run_summary.json")
        wandb_logger.finalise_run(
            status="completed",
            summary={"evaluation_only": True, **validation_metrics},
        )
        return result

    print(
        f"Training for {cfg['training']['epochs']} epochs with early stopping: "
        f"{early_enabled}, patience: {patience}, min_delta: {min_delta}"
    )
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        stage_one_last_epoch = epoch
        set_loader_epoch(bundle.train_loader, epoch)
        train_metrics, _, _ = run_hierarchy_epoch(
            model,
            bundle.train_loader,
            criteria,
            optimizer,
            device,
            True,
            scaler,
            use_amp,
            weights,
            normalize,
            hierarchy_cfg,
            matrix,
            metric_context,
            cfg,
        )
        validate = (
            epoch == 1
            or epoch % interval == 0
            or epoch == cfg["training"]["epochs"]
        )
        if validate:
            val_metrics = run_hierarchy_epoch(
                model,
                selection_loader,
                criteria,
                None,
                device,
                False,
                None,
                use_amp,
                weights,
                normalize,
                hierarchy_cfg,
                matrix,
                metric_context,
                cfg,
            )[0]
        else:
            val_metrics = {}

        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        wandb_logger.log_epoch_metrics(
            epoch=epoch,
            learning_rate=learning_rate,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        if not validate:
            print(
                f"[{run_name}] Epoch {epoch:03d}/{cfg['training']['epochs']} | "
                f"train loss {train_metrics['loss']:.4f} | validation skipped"
            )
            continue

        if selection not in val_metrics:
            raise ValueError(
                f"multi_task.selection_metric={selection!r} is not available. "
                f"Available validation metrics: {list(val_metrics)}"
            )
        score = score_for_selection(val_metrics, selection)
        print(
            f"[{run_name}] Epoch {epoch:03d}/{cfg['training']['epochs']} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val {selection} {val_metrics[selection]:.4f} | "
            "complete exact-match "
            f"{val_metrics['complete_exact_match_accuracy']:.4f} "
            f"n={val_metrics['complete_exact_match_n']}"
        )
        for task in bundle.target_cols:
            print(
                f"    {task}: val macro-F1 "
                f"{val_metrics[f'{task}_macro_f1']:.4f} | val bal-acc "
                f"{val_metrics[f'{task}_balanced_accuracy']:.4f} | "
                f"n={val_metrics[f'{task}_n']}"
            )

        improved = score > best + min_delta
        if improved or epoch == 1:
            best = score
            best_epoch = epoch
            best_selection_metrics = dict(val_metrics)
            stale = 0
            if distributed.is_main:
                payload = build_checkpoint_payload(
                    profile=profile,
                    model_state=unwrap_model(model).state_dict(),
                    cfg=cfg,
                    label_to_index_by_task=bundle.label_to_index_by_task,
                    index_to_label_by_task=bundle.index_to_label_by_task,
                    best_val_score=best,
                    selection_metric=selection,
                    best_epoch=best_epoch,
                    training_condition=input_condition,
                    long_tail_metadata=_long_tail_checkpoint_metadata(
                        model, cfg, staged_cfg
                    ),
                )
                save_checkpoint(payload, stage_one_checkpoint)
            barrier(distributed)
            wandb_logger.update_best(
                best_epoch=best_epoch,
                best_val_score=best,
                selection_metric=selection,
            )
            print(
                f"[{run_name}] New best model saved | best val {selection} "
                f"{best:.4f} at epoch {best_epoch}"
            )
        else:
            stale += 1
            print(
                f"[{run_name}] No improvement for {stale}/{patience} "
                "validation checks"
            )

        if early_enabled and stale >= patience:
            print(
                f"[{run_name}] Early stopping at epoch {epoch}. Best val "
                f"{selection} {best:.4f} at epoch {best_epoch}."
            )
            break

    if distributed.is_main:
        pd.DataFrame(history).to_csv(out_dir / "history_stage1.csv", index=False)
        save_json(
            best_selection_metrics,
            out_dir / "validation_metrics_stage1_best.json",
        )
    barrier(distributed)

    stage_two_history: list[dict] = []
    final_validation_metrics: dict = dict(best_selection_metrics)
    if staged_training:
        if bundle.tail_replay_loader is None:
            raise ValueError(
                "Staged long-tail training requires a non-empty tail replay loader"
            )
        stage_one_payload = load_checkpoint(stage_one_checkpoint, map_location=device)
        missing, ignored = load_model_state_compat(
            unwrap_model(model), stage_one_payload["model_state"]
        )
        if missing:
            print("Newly initialized Stage 2 modules: " + ", ".join(missing))
        if ignored:
            print("Stage 1 parameters not used in Stage 2: " + ", ".join(ignored))
        initialization = str(staged_cfg.get("classifier_initialisation", "keep"))
        initialise_species_classifier(
            unwrap_model(model), initialization,
            initial_scale=float((((cfg.get("model", {}) or {}).get("species_classifier", {}) or {}).get("initial_scale", 20.0))),
        )
        trainable_scope = str(staged_cfg.get("trainable_scope", "full_model"))
        trainable_count, frozen_count = apply_stage2_trainable_scope(
            unwrap_model(model), trainable_scope
        )
        print(
            f"Stage 2 initialization={initialization}, scope={trainable_scope}; "
            f"trainable parameters={trainable_count:,}, frozen={frozen_count:,}"
        )
        wandb_logger.update_summary({
            "stage2_classifier_initialisation": initialization,
            "stage2_trainable_scope": trainable_scope,
            "stage2_trainable_parameters": trainable_count,
            "stage2_frozen_parameters": frozen_count,
        })
        stage_two_epochs = int(staged_cfg.get("stage2_epochs", 20))
        criteria = build_criteria(
            bundle.tail_replay_df,
            bundle.target_cols,
            cfg["data"]["group_col"],
            bundle.label_to_index_by_task,
            device,
            use_class_weights=cfg.get("training", {}).get("class_weight", False),
            class_weighting=cfg.get("training", {}).get("class_weighting", {}),
            logit_adjustment=cfg.get("training", {}).get("logit_adjustment", {}),
        )
        if dual_criteria is not None:
            dual_criteria = build_dual_species_criteria(
                bundle.tail_replay_df,
                bundle.target_cols["species"],
                bundle.label_to_index_by_task["species"],
                cfg["data"]["group_col"], device,
                (cfg.get("training", {}).get("dual_species_classifier", {}) or {}),
            )
            metric_context["dual_criteria"] = dual_criteria
        optimizer = _build_optimizer(cfg, model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_two_epochs
        )
        print(
            f"Starting fixed tail-adaptation stage for {stage_two_epochs} epochs "
            f"with {float(staged_cfg.get('head_replay_fraction', 0.25)):.0%} "
            "head replay"
        )
        for stage_epoch in range(1, stage_two_epochs + 1):
            set_loader_epoch(bundle.tail_replay_loader, stage_epoch)
            train_metrics = run_hierarchy_epoch(
                model,
                bundle.tail_replay_loader,
                criteria,
                optimizer,
                device,
                True,
                scaler,
                use_amp,
                weights,
                normalize,
                hierarchy_cfg,
                matrix,
                metric_context,
                cfg,
            )[0]
            stage2_interval = int(staged_cfg.get("val_interval", 5))
            validate_stage2 = (
                stage_epoch == 1 or stage_epoch % stage2_interval == 0
                or stage_epoch == stage_two_epochs
            )
            if validate_stage2:
                final_validation_metrics = run_hierarchy_epoch(
                    model, bundle.val_loader, criteria, None, device, False,
                    None, use_amp, weights, normalize, hierarchy_cfg, matrix,
                    metric_context, cfg,
                )[0]
            logged_validation_metrics = final_validation_metrics if validate_stage2 else {}
            learning_rate = float(optimizer.param_groups[0]["lr"])
            scheduler.step()
            global_epoch = stage_one_last_epoch + stage_epoch
            stage_two_history.append({
                "stage_epoch": stage_epoch,
                "global_epoch": global_epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"val_{key}": value
                    for key, value in logged_validation_metrics.items()
                },
            })
            wandb_logger.log_epoch_metrics(
                epoch=global_epoch,
                learning_rate=learning_rate,
                train_metrics=train_metrics,
                val_metrics=logged_validation_metrics,
            )
            print(
                f"[{run_name}] Tail stage {stage_epoch:03d}/{stage_two_epochs} | "
                f"train loss {train_metrics['loss']:.4f} | "
                + (f"all-species val {selection} {final_validation_metrics[selection]:.4f}" if validate_stage2 else "validation skipped")
            )

        best = score_for_selection(final_validation_metrics, selection)
        # Stage two is a fixed-duration adaptation phase rather than another
        # checkpoint-selection phase. Record the epoch of its final model.
        best_epoch = stage_one_last_epoch + stage_two_epochs
        if distributed.is_main:
            pd.DataFrame(stage_two_history).to_csv(
                out_dir / "history_stage2_tail_adaptation.csv", index=False
            )
            save_json(final_validation_metrics, out_dir / "validation_metrics_final.json")
            combined = [
                {"stage": "head", **row} for row in history
            ] + [
                {"stage": "tail_adaptation", **row} for row in stage_two_history
            ]
            pd.DataFrame(combined).to_csv(out_dir / "history.csv", index=False)
    elif distributed.is_main:
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    final_payload = build_checkpoint_payload(
        profile=profile,
        model_state=unwrap_model(model).state_dict(),
        cfg=cfg,
        label_to_index_by_task=bundle.label_to_index_by_task,
        index_to_label_by_task=bundle.index_to_label_by_task,
        best_val_score=best,
        selection_metric=selection,
        best_epoch=best_epoch,
        training_condition=input_condition,
        long_tail_metadata=_long_tail_checkpoint_metadata(model, cfg, staged_cfg),
    )
    if distributed.is_main:
        save_checkpoint(final_payload, out_dir / "last_model.pt")
        if staged_training:
            save_checkpoint(final_payload, out_dir / "best_model.pt")
    barrier(distributed)
    wandb_logger.update_best(
        best_epoch=best_epoch,
        best_val_score=best,
        selection_metric=selection,
    )
    print(
        f"[{run_name}] Last model saved"
    )
    # Evaluate the final checkpoint first, then the best checkpoint. This leaves
    # ``model`` loaded with the best weights for stress and condition evaluation.
    last_test_metrics, _, _ = run_test_evaluation(
        checkpoint_name="last",
        checkpoint_path=out_dir / "last_model.pt",
        write_legacy_outputs=False,
        run_name=run_name,
        out_dir=out_dir,
        model=model,
        bundle=bundle,
        criteria=criteria,
        device=device,
        use_amp=use_amp,
        weights=weights,
        normalize=normalize,
        hierarchy_cfg=hierarchy_cfg,
        matrix=matrix,
        wandb_logger=wandb_logger,
        input_condition=input_condition,
        cfg=cfg,
        metric_context=metric_context,
    )
    test_metrics, true, pred = run_test_evaluation(
        checkpoint_name="best",
        checkpoint_path=out_dir / "best_model.pt",
        write_legacy_outputs=True,
        run_name=run_name,
        out_dir=out_dir,
        model=model,
        bundle=bundle,
        criteria=criteria,
        device=device,
        use_amp=use_amp,
        weights=weights,
        normalize=normalize,
        hierarchy_cfg=hierarchy_cfg,
        matrix=matrix,
        wandb_logger=wandb_logger,
        input_condition=input_condition,
        cfg=cfg,
        metric_context=metric_context,
    )
    model = unwrap_model(model)
    finalise_distributed(distributed)
    if not distributed.is_main:
        return {"run_name": run_name, "worker_rank": distributed.rank}
    prediction_paths = export_unlabeled_predictions(
        model=model,
        bundle=bundle,
        out_dir=out_dir,
        checkpoint_name="best",
        device=device,
        use_amp=use_amp,
        cfg=cfg,
    )

    stress = {"enabled": False, "n_conditions": 0}
    if profile.stress_evaluation:
        stress = evaluate_test_cue_suppression(
            cfg=cfg,
            run_name=run_name,
            out_dir=out_dir,
            model=model,
            checkpoint_name="best",
            checkpoint_path=out_dir / "best_model.pt",
            baseline_metrics=test_metrics,
            test_loader_context=bundle.test_loader_context,
            criteria=criteria,
            target_cols=bundle.target_cols,
            device=device,
            use_amp=use_amp,
            task_loss_weights=weights,
            normalize_loss_by_active_tasks=normalize,
            hierarchy_cfg=hierarchy_cfg,
            child_to_parent_matrix=matrix,
            metric_context=metric_context,
            wandb_logger=wandb_logger,
        )
        stress = evaluate_test_cue_suppression(
            cfg=cfg,
            run_name=run_name,
            out_dir=out_dir,
            model=model,
            checkpoint_name="last",
            checkpoint_path=out_dir / "last_model.pt",
            baseline_metrics=test_metrics,
            test_loader_context=bundle.test_loader_context,
            criteria=criteria,
            target_cols=bundle.target_cols,
            device=device,
            use_amp=use_amp,
            task_loss_weights=weights,
            normalize_loss_by_active_tasks=normalize,
            hierarchy_cfg=hierarchy_cfg,
            child_to_parent_matrix=matrix,
            metric_context=metric_context,
            wandb_logger=wandb_logger,
        )

    condition_matrix = {"enabled": False, "n_conditions": 0, "n_task_rows": 0}
    evaluation = cfg.get("evaluation", {}) or {}
    canonical_matrix = (
        evaluation.get("condition_matrix", {}) or {}
        if isinstance(evaluation, dict)
        else {}
    )
    legacy_matrix = cfg.get("condition_matrix_evaluation", {}) or {}
    matrix_cfg = canonical_matrix
    if (
        not canonical_matrix.get("conditions")
        and isinstance(legacy_matrix, dict)
        and bool(legacy_matrix.get("enabled", False))
    ):
        matrix_cfg = legacy_matrix
    if bool(matrix_cfg.get("enabled", False)):
        condition_matrix = evaluate_condition_matrix(
            cfg=cfg,
            run_name=run_name,
            out_dir=out_dir,
            model=model,
            training_condition=input_condition,
            baseline_metrics=test_metrics,
            baseline_true=true,
            baseline_pred=pred,
            test_loader_context=bundle.test_loader_context,
            criteria=criteria,
            target_cols=bundle.target_cols,
            index_to_label_by_task=bundle.index_to_label_by_task,
            device=device,
            use_amp=use_amp,
            task_loss_weights=weights,
            normalize_loss_by_active_tasks=normalize,
            hierarchy_cfg=hierarchy_cfg,
            child_to_parent_matrix=matrix,
            metric_context=metric_context,
            wandb_logger=wandb_logger,
        )

    result = {
        "run_name": run_name,
        "out_dir": str(out_dir),
        "best_val_score": best,
        "selection_metric": selection,
        "stage2_classifier_initialisation": str(staged_cfg.get("classifier_initialisation", "keep")),
        "stage2_trainable_scope": str(staged_cfg.get("trainable_scope", "full_model")),
        "prediction_paths": json.dumps(prediction_paths, sort_keys=True),
        **{f"test_{key}": value for key, value in test_metrics.items()},
        **{f"last_test_{key}": value for key, value in last_test_metrics.items()},
    }
    if profile.loader_mode == "condition":
        result = {
            "run_name": run_name,
            "model": cfg.get("model", {}).get("name"),
            "out_dir": str(out_dir),
            "train_condition": input_condition["condition"],
            "train_feature": input_condition["feature"],
            "train_transform": input_condition["transform"],
            "train_strength": input_condition.get("strength"),
            "train_condition_parameters": json.dumps(
                {
                    key: value
                    for key, value in input_condition.items()
                    if key
                    not in {"condition", "feature", "transform", "strength"}
                },
                sort_keys=True,
            ),
            "best_epoch": best_epoch,
            "best_val_score": best,
            "selection_metric": selection,
            "stage2_classifier_initialisation": str(staged_cfg.get("classifier_initialisation", "keep")),
            "stage2_trainable_scope": str(staged_cfg.get("trainable_scope", "full_model")),
            "prediction_paths": json.dumps(
                prediction_paths, sort_keys=True
            ),
            "cue_suppression_enabled": stress["enabled"],
            "cue_suppression_n_conditions": stress["n_conditions"],
            "cue_suppression_n_unique_evaluations": stress.get(
                "n_unique_evaluations", 0
            ),
            "condition_matrix_enabled": condition_matrix["enabled"],
            "condition_matrix_n_conditions": condition_matrix["n_conditions"],
            "condition_matrix_n_task_rows": condition_matrix["n_task_rows"],
            "condition_matrix_manifest_path": condition_matrix.get(
                "manifest_path"
            ),
            "condition_matrix_condition_metrics_path": condition_matrix.get(
                "condition_metrics_path"
            ),
            "condition_matrix_task_metrics_path": condition_matrix.get(
                "task_metrics_path"
            ),
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **{
                f"last_test_{key}": value
                for key, value in last_test_metrics.items()
            },
        }

    if profile.run_summary:
        save_json(result, out_dir / "run_summary.json")

    _write_pipeline_result_files(
        cfg,
        out_dir,
        final_validation_metrics,
        test_metrics,
        best_checkpoint=out_dir / "best_model.pt",
        best_epoch=best_epoch,
        selection_metric=selection,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_score": best,
        "selection_metric": selection,
        f"best_test_{selection}": test_metrics.get(selection),
        f"last_test_{selection}": last_test_metrics.get(selection),
        "stage2_classifier_initialisation": str(staged_cfg.get("classifier_initialisation", "keep")),
        "stage2_trainable_scope": str(staged_cfg.get("trainable_scope", "full_model")),
    }
    if profile.loader_mode == "condition":
        summary.update({
            "train_condition": input_condition["condition"],
            "train_feature": input_condition["feature"],
            "train_transform": input_condition["transform"],
            "train_strength": input_condition.get("strength"),
        })
    artifact_paths = [
        out_dir / "config.json",
        out_dir / "test_metrics.json",
        out_dir / "test_metrics_best.json",
        out_dir / "test_metrics_last.json",
        out_dir / "split_summary.json",
        out_dir / "label_to_index_by_task.json",
        out_dir / "history.csv",
        out_dir / "history_stage1.csv",
        out_dir / "history_stage2_tail_adaptation.csv",
        out_dir / "validation_metrics_stage1_best.json",
        out_dir / "validation_metrics_final.json",
        out_dir / "run_summary.json",
        out_dir / "best_model.pt",
        out_dir / "stage1_best_model.pt",
        *sorted(out_dir.glob("classification_report_*.csv")),
        *sorted(out_dir.glob("predictions_*.csv")),
        out_dir / "prediction.json",
    ]
    wandb_logger.log_artifacts(
        artifact_paths,
        model_metadata={
            **summary,
            "training_condition": input_condition,
            "class_mappings": bundle.label_to_index_by_task,
        },
    )
    wandb_logger.finalise_run(status="completed", summary=summary)

    print("\nBest-checkpoint test metrics:")
    print(test_metrics)
    print("\nLast-checkpoint test metrics:")
    print(last_test_metrics)
    return result
