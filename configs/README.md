# Configuration files

Configuration is plain YAML with recursive, mapping-only inheritance through
`extends`. Lists are replaced, not concatenated. Dotted `key=value` overrides
remain supported and the resolved value is always an ordinary Python dictionary.

Long-tail examples are deliberately separate runs: the full compatible stack
is in `experiments/long_tail_advanced.yaml`, while `ablations/` contains the
linear/cosine, prototype, Stage 2, contrastive, dual-head, and minimum-risk
conditions. They are not forced into a Cartesian sweep.

## Inheritance and precedence

```text
configs/defaults/base.yaml
            |
            v
       config.yaml                 one-run 224 px long-tail default
            |
            +-- configs/experiments/long_tail.yaml
            +-- configs/experiments/fine_tune_320.yaml
            +-- configs/experiments/fine_tune_384.yaml
            +-- configs/experiments/standard.yaml
            +-- configs/experiments/hierarchy.yaml
            +-- configs/experiments/dual_cue.yaml
            |        +-- ghpc_dual_cue.yaml
            +-- configs/experiments/colour_transforms.yaml
            |        +-- ghpc_colour_transforms.yaml
            +-- configs/experiments/patch_shuffle_matrix.yaml
            +-- configs/experiments/persistent_hierarchy.yaml
                         |
                         +-- persistent_hierarchy_wandb.yaml

configs/clusters/local.yaml        independent machine axis
configs/clusters/genome.yaml
            |
            +-- genome_persistent.yaml
configs/clusters/ghpc.yaml
```

For local training, the child YAML overrides its parent. For a SLURM plan, the
general precedence is:

```text
explicit CLI --override
    > explicitly imported --legacy-env compatibility values
    > cluster profile
    > experiment child
    > config.yaml
    > configs/defaults/base.yaml
```

Ambient historical launcher variables are ignored unless `--legacy-env` is
requested. Cluster files contain machine resources and paths; they must not
change scientific run-spec hashes.

## Canonical scientific layers

New child configs should keep these independent layers:

- `preprocessing` for deterministic resize and normalization on every split;
- `augmentation` for train-only random flips and rotation;
- `long_tail` for immutable cohort thresholds and the two-stage schedule;
- `training.sampling` and `training.logit_adjustment` for imbalance methods;
- `fine_tuning` for separate checkpoint-resumed resolution upgrades;
- `sweep.parameters` × `sweep.conditions` for external training expansion;
- `evaluation.test_conditions` and `evaluation.condition_matrix` for
  checkpoint evaluation that never creates another fit.

Each `sweep.conditions` item is a complete object with `name`, `feature`,
`transform`, optional `strength`, and transform-specific `parameters`. Numeric
condition families can use `name_template`, `parameter`, and an inclusive
`range`. The planner resolves one object into one run specification and then
disables expansion before invoking the trainer.

Historical condition sections are accepted as migration input, but they are
normalized into these layers and are not uploaded as duplicate W&B config
columns. See [the full configuration reference](../config.md) for syntax and
examples.

## Which file to use

| Goal | Experiment configuration | Typical count |
| --- | --- | ---: |
| One local 224 px staged long-tail run | `config.yaml` | 1 |
| Practical DINOv3 long-tail sweep | `configs/experiments/long_tail.yaml` | 6 |
| Resume at 320 px | `configs/experiments/fine_tune_320.yaml` | 1 |
| Resume at 384 px | `configs/experiments/fine_tune_384.yaml` | 1 |
| Standard hierarchy-weight sweep | `configs/experiments/standard.yaml` | 8 |
| Standard sweep with hierarchy consistency | `configs/experiments/hierarchy.yaml` | 2 |
| Full matched cue and RGB-stress study | `configs/experiments/dual_cue.yaml` | 224 |
| GHPC dual-cue historical W&B settings | `configs/experiments/ghpc_dual_cue.yaml` | 224 |
| Four-model 2×2/4×4 patch matrix | `configs/experiments/patch_shuffle_matrix.yaml` | 12 trainings, 36 evaluation cells |
| Configured saturation transforms | `configs/experiments/colour_transforms.yaml` | 202 |
| GHPC transform W&B settings | `configs/experiments/ghpc_colour_transforms.yaml` | 202 |
| Genome persistent-cache hierarchy sweep | `configs/experiments/persistent_hierarchy.yaml` | 2 |
| Same persistent sweep with W&B | `configs/experiments/persistent_hierarchy_wandb.yaml` | 2 |
| Sequential long-tail selection and resolution continuation | `configs/sweeps/fish_long_tail_pipeline.yaml` | 15 first-phase; later top-k dependent |

Use `local.yaml` for rendering and CPU-only planning, `genome.yaml` for shared
Genome jobs, `genome_persistent.yaml` for Genome persistent-cache sweeps, and
`ghpc.yaml` for GHPC node-local caching. Genome copies the shared ready cache
into each job and removes the staged copy on exit. GHPC builds a stable cache
directly on each selected node without transferring project/data/cache inputs,
and keeps that cache across sweeps while cleaning only unique run scratch. Both
profiles use the shared `${HOME}/classification/{fish-species,data}` source
layout. GHPC requires an explicit GPU-node list before rendering or submission.

## Preferred commands

Validate and inspect the safe root configuration without touching data:

```bash
PYTHONPATH=src python -m fish_species.config.validate \
  --config config.yaml --workflow training

PYTHONPATH=src python -m fish_species.config.inspect \
  --config config.yaml --workflow training --format yaml
```

Inspect an experiment and cluster plan:

```bash
make validate \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/local.yaml

make inspect \
  CONFIG=configs/experiments/patch_shuffle_matrix.yaml \
  CLUSTER=configs/clusters/local.yaml
```

Render without submission, run one local process, or explicitly submit:

```bash
make dry-run \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/genome.yaml \
  ARTIFACTS_DIR=slurm/generated/dual-cue-check

make train TRAIN_CONFIG=config.yaml

make fine-tune-320 CHECKPOINT=outputs/<run>/best_model.pt \
  MODEL=dinov3_vits16

make submit \
  CONFIG=configs/experiments/long_tail.yaml \
  CLUSTER=configs/clusters/genome.yaml GPUS=2
```

`make dry-run` never calls `sbatch`. Direct equivalents are available through
`PYTHONPATH=src python -m fish_species.slurm launch --dry-run ...` and
`--submit`. Scheduler submission is never implied by rendering.

Use `--check-paths` with the config validator only on a machine where the data
and predefined split paths should exist. Dry-run and cluster planning leave this
off so a login node does not need the full dataset tree.

See [the full configuration reference](../config.md) for every
important switch, condition semantics, resource field, and worked example.
See [phased sweep pipelines](../docs/sweep_pipelines.md) for planning,
submission, ranking, retry, resume, and 224 → 320 → 384 continuation.
