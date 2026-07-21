# Fish Species configuration reference

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

`configs/clusters/genome.yaml` uses shared project/data paths and a per-job
cache under `/tmp/${USER}/fish_species`. It defaults to one GPU and supports a
larger `gpus_per_task` through `GPUS`.

`configs/clusters/ghpc.yaml` copies the project plus labels, split pickles, and
`images/***` into a unique node-local `/scratch` directory. Setup and cleanup
are per node. `slurm.scratch.nodes` must be explicit and the scratch root must
remain unique per submission; validation rejects unsafe cleanup plans.

## Outputs and verification

Each run writes resolved configuration, split summary, label maps, stage
histories, checkpoints, validation/test metrics, classification reports,
confusion matrices, and official unlabeled predictions beneath
`output.out_dir/<run-name>`.

Before a costly submission:

1. Run `make test PYTHON=/usr/bin/python3.10` (or the cluster environment).
2. Run `make validate` and `make inspect`.
3. Dry-render the exact cluster/experiment/GPU combination.
4. Inspect the rendered `sbatch` resources and training command.
5. Run a small real-data smoke experiment before the full sweep.
