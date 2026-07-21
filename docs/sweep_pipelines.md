# Phased sweep pipelines

The phased-sweep runner evaluates a sequence of small experiments instead of
forming one global Cartesian product. The complete fish example is
[`configs/sweeps/fish_long_tail_pipeline.yaml`](../configs/sweeps/fish_long_tail_pipeline.yaml).
For a function-by-function walkthrough, see
[`sweep_pipeline_internals.md`](sweep_pipeline_internals.md).

## Commands and dry runs

```bash
PYTHONPATH=src python -m fish_species.sweeps.pipeline plan \
  --pipeline configs/sweeps/fish_long_tail_pipeline.yaml

PYTHONPATH=src python -m fish_species.sweeps.pipeline submit \
  --pipeline configs/sweeps/fish_long_tail_pipeline.yaml \
  --cluster-config configs/clusters/ghpc.yaml

PYTHONPATH=src python -m fish_species.sweeps.pipeline status \
  --pipeline configs/sweeps/fish_long_tail_pipeline.yaml

PYTHONPATH=src python -m fish_species.sweeps.pipeline advance \
  --pipeline configs/sweeps/fish_long_tail_pipeline.yaml \
  --cluster-config configs/clusters/ghpc.yaml --submit

PYTHONPATH=src python -m fish_species.sweeps.pipeline resume \
  --pipeline configs/sweeps/fish_long_tail_pipeline.yaml \
  --cluster-config configs/clusters/ghpc.yaml
```

`plan` is read-only unless `--artifacts-dir` is supplied. `submit --dry-run`,
`advance` without `--submit`, `resume --dry-run`, and `cancel --dry-run` do not
call `sbatch`/`scancel`, create training directories, or update pipeline state.
The Makefile exposes the same lifecycle as `sweep-plan`, `sweep-submit`,
`sweep-status`, `sweep-advance`, `sweep-resume`, and `sweep-cancel`.

## YAML structure and phase inheritance

`pipeline` contains the base config, output/state paths, execution policy,
result source, ranking policy, and an ordered `phases` list. Supported phase
types are:

- `cartesian`: a product only among that phase's `parameters`;
- `conditions`: one atomic configuration per condition;
- `evaluation_only`: reload a selected checkpoint and evaluate without fitting;
- `fine_tune`: reload a selected checkpoint, reset optimizer state, and train.

Each selected parent contributes its complete resolved YAML. The child applies
only `overrides`, `common_overrides`, and its condition/parameter values. Every
run records parent phase/run/hash/checkpoint, inherited metrics, and exact
phase-specific overrides. A deterministic SHA-256 hash excludes output paths,
scheduler IDs, timestamps, host/process data, W&B run IDs, and generated logs.
This prevents equivalent scientific configurations from being submitted twice.

DINOv3 ViT continuation resolutions must be divisible by 16. The example
continues the selected 224 px model through separate 320 and 384 px
`fine_tune` phases with fresh optimizer state.

## State, manifests, and resume

`state.json` is written through a temporary file, `fsync`, and atomic replace,
under an advisory lock. It contains the pipeline definition hash, current
phase, overall status, per-phase job/collector IDs and counts, selected run IDs,
best metric, attempts, retry counts, and dynamic completion/failure data.

Each phase has immutable `manifest.json` and `manifest.jsonl` files plus one
resolved YAML per run. Manifest rows contain the run/array index, run ID,
configuration hash/path, lineage, expected output/checkpoint paths, submission
and completion status, and retry count. Dynamic status changes live in state;
the manifest's scientific identity never changes.

`resume` preserves successful runs. It collects the current phase, retries only
eligible missing, preempted, or infrastructure-failed runs below the configured
limit, and then advances. Configuration/checkpoint errors, missing datasets,
invalid metrics, and CUDA OOM are not retried automatically. `cancel` calls
`scancel` only for active IDs recorded in this state and retains all state and
results.

## Result collection and ranking

Local runs emit:

```text
<run>/metrics/validation_summary.json
<run>/metrics/test_summary.json
<run>/run_status.json
```

Collection rejects unfinished status, absent/non-finite primary metrics,
configuration-hash mismatch, missing checkpoints, and failed constraints. W&B
collection is optional and matches `pipeline_run.configuration_hash`, never a
mutable run name.

Ranking is stable: valid constraint-satisfying runs first, primary metric,
configured tie breakers, lower estimated compute within `tie_tolerance`, then
configuration hash. `value_from: phase_max` implements represented-class
coverage. `drop_from_phase_baseline_less_equal` compares each candidate with
the exactly one `baseline: true` condition under the same parent. The phase
writes `leaderboard.json` and `leaderboard.csv`, including raw/selected ranks,
constraint status, rejection reason, and metric table.

To manually replace automatic parent selection, pass successful IDs:

```bash
python -m fish_species.sweeps.pipeline advance --pipeline <pipeline.yaml> \
  --parents run-id-a,run-id-b --cluster-config <cluster.yaml> --submit
```

## SLURM dependencies and local execution

The pipeline reuses the existing immutable SLURM renderer. A training array is
submitted once, then `auto_advance` creates a small CPU collector with
`afterany:<array-job-id>`. `afterany` lets it classify partial failures, retry
eligible elements, rank valid results, materialize the next phase, and submit
that phase. The collector performs no GPU training. Duplicate submission is
refused unless `--force` is used for explicit recovery.

Set `pipeline.execution.backend: local` to execute resolved configurations
sequentially through the canonical trainer. Local and SLURM runs use the same
manifest, result, ranking, and resume contracts.
