from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class _MissingDefault:
    def __repr__(self) -> str:
        return "<required>"


MISSING_DEFAULT = _MissingDefault()


@dataclass(frozen=True)
class ConfigField:
    """Description of one public dictionary-configuration field."""

    path: str
    expected_types: tuple[type, ...]
    default: Any = MISSING_DEFAULT
    required_in: frozenset[str] = frozenset()
    choices: tuple[Any, ...] = ()
    range_description: str | None = None
    consumers: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    status: str = "current"


def _field(
    path: str,
    *expected_types: type,
    default: Any = MISSING_DEFAULT,
    required_in: tuple[str, ...] = (),
    choices: tuple[Any, ...] = (),
    range_description: str | None = None,
    consumers: tuple[str, ...] = (),
    categories: tuple[str, ...] = (),
    status: str = "current",
) -> ConfigField:
    return ConfigField(
        path=path,
        expected_types=expected_types,
        default=default,
        required_in=frozenset(required_in),
        choices=choices,
        range_description=range_description,
        consumers=consumers,
        categories=categories,
        status=status,
    )


# This registry documents the dictionary contract; defaults are descriptive and
# are deliberately not injected into loaded configurations.
CONFIG_FIELDS: tuple[ConfigField, ...] = (
    _field("seed", int, required_in=("training",), consumers=("training", "conditions")),
    _field("wandb.enabled", bool, default=False, consumers=("wandb",), categories=("logging",)),
    _field("wandb.project", str, type(None), default=None, consumers=("wandb",), categories=("logging",)),
    _field("wandb.entity", str, type(None), default=None, consumers=("wandb",), categories=("logging",)),
    _field("wandb.group", str, type(None), default=None, consumers=("wandb",), categories=("logging",)),
    _field("wandb.name", str, type(None), default=None, consumers=("wandb",), categories=("logging",)),
    _field(
        "wandb.mode",
        str,
        type(None),
        default=None,
        choices=("online", "offline", "disabled", "dryrun", "run", "shared", None),
        consumers=("wandb",),
        categories=("logging",),
    ),
    _field("wandb.job_type", str, default="train", consumers=("wandb",), categories=("logging",)),
    _field("wandb.tags", list, default=(), consumers=("wandb",), categories=("logging",)),
    _field("wandb.save_code", bool, default=True, consumers=("wandb",), categories=("logging",)),
    _field("wandb.log_model", bool, default=False, choices=(False,), consumers=("wandb",), categories=("logging",)),
    _field("pipeline_run.run_id", str, consumers=("sweep pipeline", "run_name")),
    _field("pipeline_run.configuration_hash", str, consumers=("sweep pipeline", "result collection")),
    _field("pipeline_run.phase", str, consumers=("sweep pipeline",)),
    _field("pipeline_run.parent_phase", str, consumers=("sweep pipeline",)),
    _field("pipeline_run.parent_run_id", str, type(None), consumers=("sweep pipeline",)),
    _field("pipeline_run.parent_configuration_hash", str, type(None), consumers=("sweep pipeline",)),
    _field("pipeline_run.parent_checkpoint", str, type(None), consumers=("sweep pipeline",)),
    _field("pipeline_run.inherited_metrics", dict, consumers=("sweep pipeline",)),
    _field("pipeline_run.inherited_metrics.*", object, consumers=("sweep pipeline",)),
    _field("pipeline_run.phase_overrides", dict, consumers=("sweep pipeline",)),
    _field("pipeline_run.phase_overrides.*", object, consumers=("sweep pipeline",)),
    _field("pipeline_run.execution_mode", str, choices=("train", "evaluation_only"), consumers=("sweep pipeline", "training")),
    _field("data.root_dir", str, required_in=("training",), consumers=("metadata", "dataset", "cache"), categories=("storage",)),
    _field("data.dataset_format", str, default="csv", choices=("csv", "fish_pickle"), consumers=("metadata",)),
    _field("data.metadata_csv", str, consumers=("metadata",), categories=("storage",)),
    _field("data.metadata_dir", str, consumers=("fish metadata",), categories=("storage",)),
    _field("data.labels_json", str, default="label_train.json", consumers=("fish metadata",), categories=("storage",)),
    _field("data.all_classes_pickle", str, default="all_classes.pkl", consumers=("fish exploration",), categories=("storage",)),
    _field("data.split_dir", str, default="splits", consumers=("fish metadata",), categories=("storage",)),
    _field("data.split_files", dict, default={}, consumers=("fish metadata",)),
    _field("data.split_files.*", str, consumers=("fish metadata",)),
    _field("data.image_dirs", dict, default={}, consumers=("fish dataset",)),
    _field("data.image_dirs.*", str, consumers=("fish dataset",)),
    _field("data.validate_images", bool, default=True, consumers=("metadata",)),
    _field("data.image_col", str, required_in=("training",), consumers=("metadata", "dataset", "run_name")),
    _field("data.mask_col", str, type(None), default=None, consumers=("dataset", "cache")),
    _field("data.target_col", str, required_in=("training",), consumers=("metadata", "run_name"), status="legacy"),
    _field("data.group_col", str, required_in=("training",), consumers=("metadata", "splits")),
    _field("data.strip_final_number_from_group", bool, default=False, consumers=("metadata",)),
    _field("data.crop_to_foreground", bool, default=True, consumers=("dataset", "cache")),
    _field("data.crop_pad", int, float, default=0.15, range_description=">= 0", consumers=("dataset", "cache")),
    _field("data.image_size", int, default=224, range_description="> 0", consumers=("compatibility normalization",), status="legacy"),
    _field("data.target_cols", dict, required_in=("training",), consumers=("labels", "model", "metrics")),
    _field("data.target_cols.*", str, consumers=("labels", "model", "metrics")),
    _field("data.split_target_col", str, default="__taxon_for_split__", consumers=("splits",)),
    _field("preprocessing.image_size", int, default=224, range_description="> 0", consumers=("transforms", "cache")),
    _field("preprocessing.normalisation.enabled", bool, default=True, consumers=("transforms",)),
    _field("preprocessing.normalisation.mean", list, default=(0.485, 0.456, 0.406), consumers=("transforms",)),
    _field("preprocessing.normalisation.std", list, default=(0.229, 0.224, 0.225), consumers=("transforms",)),
    _field("augmentation.enabled", bool, default=True, consumers=("training transforms",)),
    _field("augmentation.horizontal_flip.enabled", bool, default=True, consumers=("training transforms",)),
    _field("augmentation.horizontal_flip.probability", int, float, default=0.5, range_description="[0, 1]", consumers=("training transforms",)),
    _field("augmentation.vertical_flip.enabled", bool, default=True, consumers=("training transforms",)),
    _field("augmentation.vertical_flip.probability", int, float, default=0.5, range_description="[0, 1]", consumers=("training transforms",)),
    _field("augmentation.rotation.enabled", bool, default=True, consumers=("training transforms",)),
    _field("augmentation.rotation.degrees", int, float, default=270, range_description=">= 0", consumers=("training transforms",)),
    _field("multi_task.loss_weights", dict, default={}, consumers=("losses",)),
    _field("multi_task.loss_weights.*", int, float, range_description=">= 0", consumers=("losses",)),
    _field("multi_task.normalize_loss_by_active_tasks", bool, default=True, consumers=("losses",)),
    _field("multi_task.selection_metric", str, default="mean_macro_f1", consumers=("checkpoints",)),
    _field("multi_task.hierarchy_loss.enabled", bool, default=False, consumers=("losses",)),
    _field("multi_task.hierarchy_loss.parent_task", str, default="genus", consumers=("losses",)),
    _field("multi_task.hierarchy_loss.child_task", str, default="species", consumers=("losses",)),
    _field("multi_task.hierarchy_loss.weight", int, float, default=0.1, range_description=">= 0", consumers=("losses",)),
    _field("multi_task.hierarchy_loss.child_to_parent", dict, type(None), default=None, consumers=("losses",)),
    _field("multitask.*", object, status="legacy", consumers=("historical run specifications",), categories=("sweeps",)),
    _field("early_stopping.enabled", bool, default=True, consumers=("training",)),
    _field("early_stopping.monitor", str, default="macro_f1", status="legacy", consumers=("saved configuration",)),
    _field("early_stopping.mode", str, default="max", choices=("min", "max"), status="legacy", consumers=("saved configuration",)),
    _field("early_stopping.patience", int, default=3, range_description=">= 0", consumers=("training",)),
    _field("early_stopping.min_delta", int, float, default=0.001, range_description=">= 0", consumers=("training",)),
    _field("split.test_size", int, float, default=0.2, required_in=("training",), range_description="(0, 1)", consumers=("splits",)),
    _field("split.val_size", int, float, default=0.15, required_in=("training",), range_description="(0, 1)", consumers=("splits",)),
    _field("split.predefined_split_dir", str, default=".", consumers=("splits",), categories=("storage",)),
    _field("split.use_predefined_splits", bool, default=False, consumers=("splits",)),
    _field("split.save_splits", bool, default=False, consumers=("splits",)),
    _field("split.strategy", str, default="group_stratified", choices=("group_stratified", "class_stratified", "long_tail"), consumers=("splits",)),
    _field("model.name", str, required_in=("training", "run_specs"), consumers=("model_factory", "run_specs")),
    _field("model.provider", str, default="auto", choices=("auto", "torchvision", "timm"), consumers=("model_factory",)),
    _field("model.checkpoint_path", str, type(None), default=None, consumers=("model_factory",), categories=("storage",)),
    _field("model.pretrained", bool, default=True, consumers=("model_factory",)),
    _field("model.freeze_backbone", bool, default=False, consumers=("model_factory",)),
    _field("model.species_classifier.type", str, default="linear", choices=("linear", "cosine"), consumers=("model",)),
    _field("model.species_classifier.initial_scale", int, float, default=20.0, consumers=("model",)),
    _field("model.species_classifier.learnable_scale", bool, default=True, consumers=("model",)),
    _field("model.species_classifier.maximum_scale", int, float, default=100.0, consumers=("model",)),
    _field("model.prototype_classifier.enabled", bool, default=False, consumers=("model", "training")),
    _field("model.prototype_classifier.update", str, default="static", choices=("static", "ema"), consumers=("training",)),
    _field("model.prototype_classifier.momentum", int, float, default=0.99, consumers=("training",)),
    _field("model.prototype_classifier.scale", int, float, default=20.0, consumers=("model",)),
    _field("model.prototype_classifier.fusion.enabled", bool, default=True, consumers=("model",)),
    _field("model.prototype_classifier.fusion.mode", str, default="fixed", choices=("fixed", "frequency_dependent"), consumers=("model",)),
    _field("model.prototype_classifier.fusion.learned_weight", int, float, default=0.5, consumers=("model",)),
    _field("model.prototype_classifier.fusion.prototype_strength", int, float, default=10.0, consumers=("model",)),
    _field("model.dual_species_classifier.enabled", bool, default=False, consumers=("model", "losses")),
    _field("model.dual_species_classifier.classifier_type", str, default="cosine", choices=("linear", "cosine"), consumers=("model",)),
    _field("model.dual_species_classifier.inference.mode", str, default="fused", choices=("natural", "balanced", "fused"), consumers=("model",)),
    _field("model.dual_species_classifier.inference.natural_weight", int, float, default=0.5, consumers=("model",)),
    _field("model.dual_species_classifier.inference.frequency_dependent", bool, default=False, consumers=("model",)),
    _field("training.epochs", int, required_in=("training",), range_description="> 0", consumers=("training",)),
    _field("training.batch_size", int, required_in=("training",), range_description="> 0", consumers=("loaders",)),
    _field("training.lr", int, float, required_in=("training",), range_description="> 0", consumers=("optimiser",)),
    _field("training.weight_decay", int, float, required_in=("training",), range_description=">= 0", consumers=("optimiser",)),
    _field("training.use_amp", bool, default=True, consumers=("training",)),
    _field("training.amp_dtype", str, default="bfloat16", choices=("bfloat16", "float16"), consumers=("training",)),
    _field("training.eval_batch_size", int, default=1024, consumers=("loaders",)),
    _field("training.persistent_workers", bool, default=False, consumers=("loaders",)),
    _field("training.optimizer.fused", bool, default=False, consumers=("optimiser",)),
    _field("training.compile.enabled", bool, default=False, consumers=("training",)),
    _field("training.compile.mode", str, default="default", consumers=("training",)),
    _field("training.class_weight", bool, default=False, consumers=("losses",)),
    _field("training.class_weighting", dict, default={}, consumers=("losses",)),
    _field("training.class_weighting.basis", str, default="samples", choices=("samples", "groups"), consumers=("losses",)),
    _field("training.class_weighting.method", str, default="inverse_frequency", choices=("inverse_frequency", "sqrt_inverse_frequency", "effective_number"), consumers=("losses",)),
    _field("training.class_weighting.beta", int, float, default=0.9999, range_description="[0, 1)", consumers=("losses",)),
    _field("training.optimizer.name", str, default="adamw", choices=("adamw",), consumers=("optimiser",)),
    _field("training.sampling.strategy", str, default="random", choices=("random", "weighted"), consumers=("loaders",)),
    _field("training.logit_adjustment.enabled", bool, default=False, consumers=("losses",)),
    _field("training.logit_adjustment.task", str, default="species", consumers=("losses",)),
    _field("training.logit_adjustment.tau", int, float, default=1.0, range_description=">= 0", consumers=("losses",)),
    _field("training.dual_species_classifier.natural_loss_weight", int, float, default=1.0, consumers=("losses",)),
    _field("training.dual_species_classifier.balanced_loss_weight", int, float, default=1.0, consumers=("losses",)),
    _field("training.dual_species_classifier.balanced_method", str, default="logit_adjustment", choices=("logit_adjustment", "class_weight", "none"), consumers=("losses",)),
    _field("training.dual_species_classifier.tau", int, float, default=1.0, consumers=("losses",)),
    _field("training.distributed.enabled", bool, default=False, consumers=("training", "loaders", "SLURM")),
    _field("training.distributed.backend", str, default="nccl", choices=("nccl", "gloo"), consumers=("training",)),
    _field("training.distributed.timeout_minutes", int, default=120, range_description="> 0", consumers=("training", "cache")),
    _field(
        "training.mode",
        str,
        default="multitask",
        choices=("multitask",),
        consumers=("canonical trainer",),
    ),
    _field("training.num_workers", int, default=4, range_description=">= 0", consumers=("loaders",)),
    _field("training.val_interval", int, default=3, range_description="> 0", consumers=("training",)),
    _field("long_tail.head_min_samples", int, default=11, range_description=">= 2", consumers=("splits", "metrics")),
    _field("long_tail.staged_training.enabled", bool, default=False, consumers=("training", "loaders")),
    _field("long_tail.staged_training.stage2_epochs", int, default=20, range_description="> 0", consumers=("training",)),
    _field("long_tail.staged_training.head_replay_fraction", int, float, default=0.25, range_description="[0, 1)", consumers=("loaders",)),
    _field("long_tail.staged_training.classifier_initialisation", str, default="keep", choices=("keep", "random", "prototype"), consumers=("training",)),
    _field("long_tail.staged_training.trainable_scope", str, default="full_model", choices=("heads", "heads_and_last_block", "full_model"), consumers=("training",)),
    _field("long_tail.staged_training.val_interval", int, default=5, consumers=("training",)),
    _field("multi_task.hierarchical_contrastive.enabled", bool, default=False, consumers=("model", "losses")),
    _field("multi_task.hierarchical_contrastive.weight", int, float, default=0.05, consumers=("losses",)),
    _field("multi_task.hierarchical_contrastive.temperature", int, float, default=0.1, consumers=("losses",)),
    _field("multi_task.hierarchical_contrastive.same_species_weight", int, float, default=1.0, consumers=("losses",)),
    _field("multi_task.hierarchical_contrastive.same_genus_weight", int, float, default=0.25, consumers=("losses",)),
    _field("multi_task.hierarchical_contrastive.projection_dim", int, default=256, consumers=("model",)),
    _field("multi_task.hierarchical_contrastive.use_two_views", bool, default=True, consumers=("loaders",)),
    _field("multi_task.balanced_contrastive.enabled", bool, default=False, consumers=("model", "losses")),
    _field("multi_task.balanced_contrastive.weight", int, float, default=0.1, consumers=("losses",)),
    _field("multi_task.balanced_contrastive.temperature", int, float, default=0.1, consumers=("losses",)),
    _field("multi_task.balanced_contrastive.projection_dim", int, default=256, consumers=("model",)),
    _field("multi_task.balanced_contrastive.include_class_prototypes", bool, default=True, consumers=("losses",)),
    _field("multi_task.balanced_contrastive.class_average", bool, default=True, consumers=("losses",)),
    _field("fine_tuning.enabled", bool, default=False, consumers=("training",)),
    _field("fine_tuning.checkpoint_path", str, type(None), default=None, consumers=("training",), categories=("storage",)),
    _field("fine_tuning.reset_optimizer", bool, default=True, consumers=("training",)),
    _field("output.out_dir", str, required_in=("training",), consumers=("training",), categories=("storage",)),
    _field("cache.enabled", bool, default=False, consumers=("cache",)),
    _field("cache.dir", str, default="cache/images", consumers=("cache",), categories=("storage",)),
    _field("cache.root_dir_cache", str, default=None, consumers=("cache",), categories=("storage",)),
    _field("cache.format", str, default="png", choices=("png", "jpg", "jpeg"), consumers=("cache",)),
    _field("cache.rebuild", bool, default=False, consumers=("cache",)),
    _field("cache.num_workers", int, default=4, range_description=">= 1", consumers=("cache",)),
    _field("cache.cache_dir", str, status="runtime", consumers=("SLURM cache setup",), categories=("storage", "SLURM")),
    _field("cache.root_dir", str, status="runtime", consumers=("SLURM cache setup",), categories=("storage", "SLURM")),
    _field("test_cue_suppression.enabled", bool, default=False, consumers=("compatibility normalization",), status="legacy"),
    _field("test_cue_suppression.condition_names", list, consumers=("fixed RGB evaluation",)),
    _field("test_cue_suppression.saturation.enabled", bool, default=True, consumers=("conditions",)),
    _field("test_cue_suppression.saturation.start", int, float, default=1.0, range_description="[0, 1]", consumers=("conditions",)),
    _field("test_cue_suppression.saturation.stop", int, float, default=0.0, range_description="[0, 1]", consumers=("conditions",)),
    _field("test_cue_suppression.saturation.step", int, float, default=0.01, range_description="> 0", consumers=("conditions",)),
    _field("test_cue_suppression.saturation.values", list, consumers=("conditions",)),
    _field("test_cue_suppression.grayscale.enabled", bool, default=True, consumers=("conditions",)),
    _field("test_cue_suppression.channel_shuffle.enabled", bool, default=True, consumers=("conditions",)),
    _field("test_cue_suppression.channel_shuffle.orders", list, default=((2, 0, 1),), consumers=("conditions",)),
    _field("test_cue_suppression.bilateral_filter.enabled", bool, default=True, consumers=("conditions",)),
    _field("test_cue_suppression.bilateral_filter.settings", list, consumers=("conditions",)),
    _field("test_cue_suppression.gaussian_blur.enabled", bool, default=True, consumers=("conditions",)),
    _field("test_cue_suppression.gaussian_blur.sigmas", list, consumers=("conditions",)),
    _field("test_cue_suppression.patch_shuffle.enabled", bool, default=True, consumers=("conditions",)),
    _field("test_cue_suppression.patch_shuffle.grid_sizes", list, consumers=("conditions",)),
    _field("test_cue_suppression.patch_shuffle.seed", int, consumers=("conditions",)),
    _field("condition_matrix_evaluation.enabled", bool, default=False, consumers=("compatibility normalization",), status="legacy"),
    _field("condition_matrix_evaluation.condition_names", list, consumers=("post-training condition matrix",)),
    _field("condition_matrix_evaluation.write_reports", bool, default=True, consumers=("post-training condition matrix",)),
    _field("matched_condition_training.enabled", bool, default=False, consumers=("compatibility normalization",), status="legacy"),
    _field("matched_condition_training.include_original", bool, default=True, consumers=("run_specs",)),
    _field("matched_condition_training.deduplicate_equivalent_conditions", bool, default=True, consumers=("run_specs",)),
    _field("matched_condition_training.evaluate_original_model_on_all_test_conditions", bool, default=True, consumers=("run_specs",)),
    _field("matched_condition_training.condition_names", list, consumers=("run_specs",)),
    _field(
        "experiment.type",
        str,
        choices=(
            "standard",
            "matched_condition",
            "rgb_stress_test",
            "matched_and_rgb_stress",
        ),
        consumers=("canonical trainer profile resolution",),
    ),
    _field("experiment.training_condition", dict, str, type(None), consumers=("canonical trainer profile resolution",)),
    _field("experiment.training_condition.*", object, consumers=("canonical trainer profile resolution",)),
    _field("sweep.enabled", bool, default=False, consumers=("sweeps",)),
    _field("sweep.parameters", dict, default={}, consumers=("sweeps",)),
    _field("sweep.parameters.*", list, consumers=("sweeps",)),
    _field("sweep.conditions", list, consumers=("sweeps",)),
    _field("sweep.run_id_template", str, consumers=("run_specs",)),
    _field("input_condition.enabled", bool, default=False, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.condition", str, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.name", str, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.feature", str, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.transform", str, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.strength", int, float, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.retention", int, float, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.order", list, tuple, str, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.diameter", int, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.sigma_colour", int, float, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.sigma_space", int, float, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.sigma", int, float, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.grid_size", int, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.seed", int, status="runtime", consumers=("matched-condition trainer",)),
    _field("input_condition.parameters", dict, default={}, consumers=("matched-condition trainer",)),
    _field("input_condition.parameters.*", object, consumers=("matched-condition trainer",)),
    _field("evaluation.test_conditions.enabled", bool, default=False, consumers=("post-training evaluation",)),
    _field("evaluation.test_conditions.evaluate_original_training", bool, default=False, consumers=("run_specs",)),
    _field("evaluation.test_conditions.conditions", list, default=(), consumers=("post-training evaluation",)),
    _field("evaluation.condition_matrix.enabled", bool, default=False, consumers=("post-training evaluation",)),
    _field("evaluation.condition_matrix.conditions", list, default=(), consumers=("post-training evaluation",)),
    _field("evaluation.condition_matrix.write_reports", bool, default=True, consumers=("post-training evaluation",)),
    _field("evaluation.taxonomic_distance.enabled", bool, default=False, consumers=("metrics",)),
    _field("evaluation.taxonomic_distance.same_species_cost", int, float, default=0.0, consumers=("metrics",)),
    _field("evaluation.taxonomic_distance.same_genus_cost", int, float, default=1.0, consumers=("metrics",)),
    _field("evaluation.taxonomic_distance.different_genus_cost", int, float, default=2.0, consumers=("metrics",)),
    _field("inference.enabled", bool, default=False, consumers=("prediction export",)),
    _field("inference.splits", list, default=(), consumers=("prediction export",)),
    _field("inference.enforce_hierarchy", bool, default=True, consumers=("prediction export",)),
    _field("inference.hierarchy_genus_weight", int, float, default=1.0, range_description=">= 0", consumers=("prediction export",)),
    _field("inference.taxonomic_minimum_risk.enabled", bool, default=False, consumers=("metrics",)),
)


def field_for_path(path: str) -> ConfigField | None:
    """Return the most specific registered field matching ``path``."""
    exact = next((field for field in CONFIG_FIELDS if field.path == path), None)
    if exact is not None:
        return exact
    matches = [
        field
        for field in CONFIG_FIELDS
        if field.path.endswith(".*")
        and path.startswith(field.path[:-1])
        and len(path) > len(field.path) - 1
    ]
    if not matches:
        return None
    return max(matches, key=lambda field: len(field.path))


def is_known_config_path(path: str) -> bool:
    return bool(path) and field_for_path(path) is not None


__all__ = [
    "CONFIG_FIELDS",
    "ConfigField",
    "MISSING_DEFAULT",
    "field_for_path",
    "is_known_config_path",
]
