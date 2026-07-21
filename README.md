# Fish genus/species classifier

This project trains one shared image backbone with separate genus and species
heads. It reads the supplied `all_classes.pkl`, `label_train.json`, and
`splits/{train,test,unseen}.pkl` files, derives genus from each binomial species
name, and exports predictions for the unlabeled test and unseen filename lists.
Every completed run writes the required
`outputs/<run-name>/prediction.json`, mapping each official test filename to
its predicted species label.

## Data layout

The default [config.yaml](config.yaml) expects:

```text
../data/
├── all_classes.pkl
├── label_train.json
├── splits/
│   ├── train.pkl
│   ├── test.pkl
│   └── unseen.pkl
└── images/
```

The supplied train, test, and unseen images share `images/`, so all three
`data.image_dirs` values use that directory.

## Train and inspect

```bash
python train.py --config config.yaml --dry-run
python train.py --config config.yaml
```

Training uses complete genus/species labels and a deterministic long-tail split.
Species with more than 10 labeled images form the head stage; the fixed second
stage adapts to species with 10 or fewer images while replaying head samples.
The official unlabeled test/unseen lists use a separate image-only inference
dataset.

## Long-tail options

[long_tail.yaml](configs/experiments/long_tail.yaml) provides practical DINOv3
ViT and ConvNeXt sweeps at 224 px. Sampling and loss correction are independent:

```yaml
training:
  optimizer: {name: adamw}
  sampling: {strategy: weighted}  # or random
  logit_adjustment:
    enabled: true
    task: species
    tau: 1.0
```

Checkpoint selection uses overall species top-1 accuracy. Reports also include
species macro-F1, balanced accuracy, top-5, genus macro-F1, genus/species
consistency, head/tail metrics, and few/medium/many-shot macro-F1 with counts.

Use `make fine-tune-320 CHECKPOINT=... MODEL=...` or `make fine-tune-384 ...`
for separate resolution-upgrade runs. Use `GPUS=2` for single-node torchrun/DDP.

## Configured image transforms

Transforms are ordinary condition objects, not a separate ablation subsystem.
[colour_transforms.yaml](configs/experiments/colour_transforms.yaml) gives a
numeric saturation sweep; [dual_cue.yaml](configs/experiments/dual_cue.yaml)
shows all supported transforms:

- `saturation`
- `grayscale`
- `channel_shuffle`
- `bilateral_filter`
- `gaussian_blur`
- `patch_shuffle`

Each condition is passed through the same pipeline in
`src/fish_species/data/transforms.py`.

The official `unseen` split has no supplied labels. This supervised model emits
closed-set predictions among labeled training species; learning truly unseen
species would require a separate zero-shot method.

## Long-tail extensions

The opt-in cosine/prototype/dual classifiers, contrastive losses, Stage 2
controls, and taxonomic-risk evaluation are documented in
[config.md](config.md#long-tail-classification). The complete example is
`configs/experiments/long_tail_advanced.yaml`; independent conditions are in
`configs/ablations/`.

## Automated phased sweeps

Sequential, resumable long-tail model selection is documented in
[docs/sweep_pipelines.md](docs/sweep_pipelines.md). The complete 224 → 320 →
384 example is `configs/sweeps/fish_long_tail_pipeline.yaml`; it uses small
phase-local products, immutable parent inheritance, deterministic hashes,
strict result collection, and optional SLURM `afterany` auto-advancement.
W&B logs metrics and lightweight scientific records only: model checkpoints
are excluded, and confusion matrices are not generated.
The separate [educational internals guide](docs/sweep_pipeline_internals.md)
explains every sweep function, reused repository API, and key library call.
# fishy
