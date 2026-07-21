# Fish Species configuration reference

## Long-tail classification

All additions are opt-in. The default learned species head remains linear;
prototype, dual-head, contrastive, and minimum-risk features default to off.
The existing genus head and species/genus hierarchy are unchanged.

`model.species_classifier` selects `linear` or `cosine`. A cosine head
normalizes embeddings and weight rows in float32 and multiplies their cosine
similarity by `min(exp(log_scale), maximum_scale)`.

`model.prototype_classifier` controls static or synchronized EMA training-set
prototypes. Fixed fusion uses `alpha * learned + (1-alpha) * prototype`.
Frequency fusion uses `alpha_c=n_c/(n_c+prototype_strength)`. Prototype rows
and observation counts are model-state buffers; zero-count rows stay finite.

`long_tail.staged_training.classifier_initialisation` is `keep`, `random`, or
`prototype`; `trainable_scope` is `heads`, `heads_and_last_block`, or
`full_model`. The middle scope supports torchvision ConvNeXt/ViT and timm
DINOv3 ConvNeXt/ViT structural interfaces. Stage 2 creates a fresh optimizer
over trainable parameters and has its own `val_interval`.

`multi_task.hierarchical_contrastive` gives same-species pairs more weight than
different species in the same genus. `multi_task.balanced_contrastive` balances
anchor classes and may add training-only prototype positives. An optional
second augmentation supports singleton source images. Compatible projection
dimensions share one `linear -> GELU -> linear -> L2 normalize` head; classifiers
continue to consume the original embedding.

`model.dual_species_classifier` creates natural and balanced species heads.
The natural head uses ordinary cross entropy. The balanced head uses exactly
one of `logit_adjustment`, `class_weight`, or `none` from
`training.dual_species_classifier`. Fixed fusion uses `natural_weight`;
frequency fusion uses the monotonic natural-head weight
`n_c/(n_c+median_positive_count)`.

With active-loss normalization, the exact loss is the sum of weighted species,
genus, hierarchy consistency, hierarchical contrastive, and balanced
contrastive terms divided by the weights of enabled and available terms only.
With normalization disabled, the numerator is unchanged and no division is
performed. Dual-head natural/balanced terms replace the single species term.

`evaluation.taxonomic_distance` reports mean/median cost, within-genus error
fraction, species/genus correctness, and cohort costs. Optional minimum-risk
evaluation reports a second prediction set while preserving raw top-1/top-5.
For candidate species `k`, risk is calculated as
`C2+(C1-C2)P(genus(k)|x)+(C0-C1)p(k|x)`, so no dense species² matrix is needed.

Performance keys are `training.eval_batch_size`, total configured
`num_workers`, `persistent_workers`, `amp_dtype`, `optimizer.fused`, and
`compile.enabled/mode`. Compilation occurs before DDP; validation and labelled
test data use non-padding rank partitions. `images_per_second` is logged for
otherwise-identical throughput comparisons.

Checkpoint model state naturally includes cosine scales, prototypes/counts,
dual heads, and projections. `long_tail_metadata` records classifier type,
prototype mode, Stage 2 initialization/scope, and taxonomic inference config.
Legacy matching tensors load; a legacy species head initializes both dual heads.
Incompatible tensor shapes are errors and newly initialized keys are reported.

The project uses ordinary YAML mappings. `config.yaml` extends
`configs/defaults/base.yaml`; experiment files change scientific choices and
cluster files change only scheduler, resources, scratch, and machine paths.

## Resolution and commands

`extends` paths are relative to the child file. Mappings merge recursively and
lists replace parent lists. Precedence is:

```text
CLI --override
  > cluster YAML
  > experiment YAML
  > config.yaml
  > configs/defaults/base.yaml
```

Useful commands:

```bash
make validate CONFIG=configs/experiments/long_tail.yaml \
  CLUSTER=configs/clusters/genome.yaml

make inspect CONFIG=configs/experiments/long_tail.yaml \
  CLUSTER=configs/clusters/genome.yaml

make dry-run CONFIG=configs/experiments/long_tail.yaml \
  CLUSTER=configs/clusters/genome.yaml GPUS=2

make submit CONFIG=configs/experiments/long_tail.yaml \
  CLUSTER=configs/clusters/genome.yaml GPUS=2

make dry-run CONFIG=configs/experiments/long_tail.yaml \
  CLUSTER=configs/clusters/ghpc.yaml GPUS=2 \
  GHPC_NODES='[gpu001,gpu002]'
```

Dry-run never submits. GHPC node names above are examples and must be replaced
with the exact intended nodes.

## Data and label contract

The supplied layout is:

```text
${HOME}/classification/data/
├── all_classes.pkl
├── label_train.json
├── splits/
│   ├── train.pkl
│   ├── test.pkl
│   └── unseen.pkl
└── images/
```

Key data settings:

| Key | Meaning |
| --- | --- |
| `data.dataset_format` | `fish_pickle` for the supplied dataset. |
| `data.metadata_dir` | Directory containing labels and split pickles. |
| `data.root_dir` | Image/data root. |
| `data.image_col` | Image filename/path column (`image_path`). |
| `data.group_col` | Sampling/group identity column (`image_id`). |
| `data.target_cols.genus` | Complete supervised genus column. |
| `data.target_cols.species` | Complete supervised species column. |
| `inference.splits` | Official image-only splits, normally `test` and `unseen`. |

Every supervised train/validation/test row must have complete genus and species
labels and fails validation otherwise. Official unlabeled test/unseen images use
`InferenceImageDataset` and are never passed to supervised metrics.

## Long-tail split and staged training

The canonical configuration uses:

```yaml
split:
  strategy: long_tail
  test_size: 0.10
  val_size: 0.15

long_tail:
  head_min_samples: 11
  staged_training:
    enabled: true
    stage2_epochs: 20
    head_replay_fraction: 0.25
```

Frequency cohorts are derived from the full labeled dataset before splitting:

- head: more than 10 samples;
- tail: 10 or fewer samples;
- few-shot: 2–5 samples;
- medium-shot: 6–20 samples;
- many-shot: more than 20 samples.

Stage 1 trains only on head species and selects the checkpoint on head-only
validation. Stage 2 reloads that checkpoint, trains for exactly
`stage2_epochs` on tail samples plus the configured fraction of head replay,
and does not early-stop. Final validation and test cover all species and expose
head/tail plus few/medium/many views. Cohort reports include both sample and
species counts.

## Model and resolution

Torchvision models remain available. DINOv3 aliases use timm:

- `dinov3_vits16`, `dinov3_vitb16`, `dinov3_vitl16`;
- `dinov3_convnext_tiny`, `dinov3_convnext_small`;
- `dinov3_convnext_base`, `dinov3_convnext_large`.

The repository deliberately excludes the 7B model and multi-node FSDP.

```yaml
model:
  name: dinov3_vits16
  provider: auto       # auto, torchvision, or timm
  pretrained: true
  checkpoint_path: null
  freeze_backbone: false
```

Initial comparisons run at 224 px. The 320 and 384 px configurations are
separate resumable runs with fresh AdamW state:

```bash
make fine-tune-320 CHECKPOINT=outputs/<224-run>/best_model.pt \
  MODEL=dinov3_vits16

make fine-tune-384 \
  CHECKPOINT=outputs/fine_tune_320/<run>/best_model.pt \
  MODEL=dinov3_vits16
```

The checkpoint architecture must match `MODEL`. DINOv3 ViT input sizes must be
divisible by patch size 16.

## Optimization and imbalance options

```yaml
training:
  optimizer: {name: adamw}
  lr: 0.0003
  weight_decay: 0.0001
  sampling:
    strategy: random       # random or weighted
  logit_adjustment:
    enabled: false
    task: species
    tau: 1.0
```

Weighted sampling uses inverse species frequency with replacement. Logit
adjustment uses the labeled training prior and non-negative `tau`. Sampling,
class weighting, and logit adjustment are independent; do not enable several
imbalance corrections unintentionally when comparing methods.

## Metrics and checkpoint selection

`multi_task.selection_metric` is `species_accuracy`, the overall species top-1
accuracy. Every validation/test result includes:

- species macro-F1, balanced accuracy, top-1 accuracy, and top-5 accuracy;
- genus macro-F1;
- predicted genus/species consistency;
- head and tail species macro-F1, accuracy, and balanced accuracy;
- few-, medium-, and many-shot species macro-F1;
- cohort sample counts and represented-species counts.

Hierarchy loss remains optional and is configured independently under
`multi_task.hierarchy_loss`.

## Single-node multi-GPU training

```yaml
training:
  distributed:
    enabled: true
    backend: nccl
    timeout_minutes: 120
```

For local execution use `make train GPUS=2`. For Slurm use `GPUS=2` with
`make dry-run` or `make submit`. The renderer requests `--gres=gpu:2` and runs:

```text
srun torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m fish_species.training ...
```

One array element remains one model fit and one Slurm task. Random sampling uses
`DistributedSampler`; weighted sampling uses a deterministic rank-partitioned
weighted sampler. Validation is evaluated consistently on every rank so all
workers make the same checkpoint/early-stopping decision. Only rank zero writes
checkpoints, reports, predictions, and W&B records. `training.num_workers` is a
per-job budget and is divided across torchrun ranks.

## Cluster profiles

Genome and GHPC share these source locations:

```text
project: ${HOME}/classification/fish-species
data:    ${HOME}/classification/data
```

`configs/clusters/genome.yaml` reads a ready cache from
`slurm.paths.cache_root`, copies it into the array task's job-local `TMPDIR`,
and explicitly deletes only that staged copy in the task's exit trap. Shared
source cache and copied-back results are never deleted.

`configs/clusters/ghpc.yaml` does not transfer the project, dataset, or a
prebuilt cache. Each selected GPU node builds
`/scratch/${USER}/fish_species_image_cache` directly from shared data under a
file lock and reuses its readiness marker across later sweeps. Per-submission
scratch still has a unique guarded root for outputs and is cleaned after the
array; the persistent node cache sits outside that root. The array is limited
to the explicitly configured `slurm.scratch.nodes`.

## Outputs and verification

Each run writes resolved configuration, split summary, label maps, stage
histories, checkpoints, validation/test metrics, classification reports,
and official unlabeled predictions beneath
`output.out_dir/<run-name>`.

W&B records metrics, configuration, summaries, and lightweight scientific
files only. Checkpoints (`.pt`, `.pth`, `.ckpt`, and `.safetensors`) are always
excluded from W&B artifacts, and confusion matrices are not generated or
uploaded.

Before a costly submission:

1. Run `make test PYTHON=/usr/bin/python3.10` (or the cluster environment).
2. Run `make validate` and `make inspect`.
3. Dry-render the exact cluster/experiment/GPU combination.
4. Inspect the rendered `sbatch` resources and training command.
5. Run a small real-data smoke experiment before the full sweep.

## Sequential sweep pipelines

Use `fish_species.sweeps.pipeline` when ablations depend on results from prior
phases. The pipeline supports phase-local `cartesian` products, atomic
`conditions`, checkpoint-backed `evaluation_only` runs, and fresh-optimizer
`fine_tune` continuation. Children inherit the selected parent's complete
resolved configuration before applying phase overrides.

```bash
PYTHONPATH=src python -m fish_species.sweeps.pipeline plan \
  --pipeline configs/sweeps/fish_long_tail_pipeline.yaml

PYTHONPATH=src python -m fish_species.sweeps.pipeline submit \
  --pipeline configs/sweeps/fish_long_tail_pipeline.yaml \
  --cluster-config configs/clusters/ghpc.yaml

PYTHONPATH=src python -m fish_species.sweeps.pipeline status \
  --pipeline configs/sweeps/fish_long_tail_pipeline.yaml
```

Planning is read-only unless `--artifacts-dir` is supplied. `advance` is also
read-only unless `--submit` is supplied. State is locked and atomically
replaced; phase manifests and resolved YAML are immutable. Runs are matched by
a scientific configuration hash that excludes scheduler, output, host/process,
timestamp, and W&B identity fields.

Local collection requires `metrics/validation_summary.json`,
`metrics/test_summary.json`, and `run_status.json`. A completed run is rejected
when its status/hash/metric/checkpoint is missing or invalid. Ranking applies
coverage and per-parent baseline-relative constraints before the primary
metric, tie breakers, lower compute, and hash. Missing, preempted, and
infrastructure failures may be retried within the configured limit; validation,
checkpoint, dataset, invalid-metric, and CUDA-OOM failures are not automatic
retries.

With `auto_advance`, each training array gets a CPU-only SLURM collector using
`afterany`, so partial failures are recorded before retry or next-phase
selection. `resume` never repeats successful runs, and `cancel` retains state.
The complete YAML and 224 → 320 → 384 continuation are documented in
[`docs/sweep_pipelines.md`](docs/sweep_pipelines.md).
