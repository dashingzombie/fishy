# Phased sweep pipeline: educational internals

This document explains how the phased sweep system works from the command line
to a training process and back to parent selection. It covers the functions
implemented for this repository, the existing fish-species functions they
reuse, and the important Python/library calls underneath them.

## 1. The system as a state machine

One pipeline definition is immutable scientific intent. One state file is the
mutable operational record. Every phase follows this sequence:

```text
load + validate pipeline YAML
        │
        ▼
inherit complete selected-parent configs
        │
        ▼
apply only this phase's dotted overrides
        │
        ▼
hash scientific config ── reject duplicates
        │
        ▼
write immutable YAML + phase manifest
        │
        ├── local: run canonical trainer sequentially
        │
        └── SLURM: render canonical array → sbatch
                                      │
                                      └── afterany CPU collector
        ▼
read status + metrics + checkpoint
        │
        ▼
reject invalid runs → constraints → deterministic rank → top-k
        │
        ▼
record selected lineage → create next phase
```

The scientific configuration is deliberately separate from scheduler state.
Changing a learning rate changes the scientific hash. Changing an output path,
SLURM job ID, W&B run ID, hostname, or timestamp does not.

## 2. Module boundaries

| Module | Responsibility | Does it mutate state? | Can it run external commands? |
| --- | --- | --- | --- |
| `sweeps/schema.py` | Load and validate pipeline YAML | No | No |
| `sweeps/core.py` | Expansion, hashes, state, collection, ranking | Only explicit write/materialize functions | No |
| `sweeps/execution.py` | Adapt records to local training or SLURM | Writes rendered artifacts/local statuses | Yes |
| `sweeps/pipeline.py` | CLI commands and lifecycle orchestration | Only non-dry commands | Through execution functions |
| `training/runner.py` | Train/evaluate one resolved run | Writes run outputs | Runs PyTorch work |
| `training/cli.py` | Resolve and invoke canonical single-run training | Writes failure status for pipeline runs | No scheduler calls |

This separation is the reason `plan` and dry-run commands can be tested without
a dataset, GPU, W&B account, or SLURM installation.

## 3. `sweeps/schema.py`: configuration safety

### `PipelineConfigError`

A dedicated `ValueError` subtype. The CLI catches it and prints a short
`pipeline error:` message with exit code 2. Separating pipeline-definition
errors from runtime `PipelineError` makes tests and callers distinguish invalid
YAML from failed operations.

### `_mapping(value, path)`

Checks that a YAML node is a dictionary and returns it with a narrowed type.
Every error includes the dotted logical path, such as `pipeline.execution`, so
the user knows exactly which YAML object is malformed.

### `_selection(value, path)`

Validates ranking configuration: direction, positive `top_k`, non-negative tie
tolerance, tie-breaker metric/direction pairs, supported constraint operators,
`phase_max` fractions in `[0, 1]`, and an optional explicit baseline condition.
This validation occurs before any run directory or scheduler call exists.

### `validate_pipeline(document)`

Validates the complete phase graph and execution/result policy. It checks:

- required pipeline names and paths;
- `local` versus `slurm` backend;
- retry limits and local/W&B result source requirements;
- unique phase and condition names;
- required Cartesian parameter lists and atomic conditions;
- evaluation-only and fine-tune checkpoint semantics;
- known parents, cycles, and parent-before-child ordering.

Cycle detection walks each phase's parent chain with a `set`. Encountering the
same phase twice proves a cycle. A second pass enforces declaration order,
because this implementation is a sequential lifecycle rather than a general
parallel DAG executor.

### `load_pipeline(path)`

Uses `Path.resolve`, `Path.read_text`, and `yaml.safe_load`, calls
`validate_pipeline`, then adds `_pipeline_file`. That absolute source path is
runtime context, not part of user YAML, and lets child processes find the same
definition after a login shell exits.

## 4. `sweeps/core.py`: deterministic science and persistence

### Errors and canonical hashing

`PipelineError` represents safe operational refusal: changed immutable files,
missing state, duplicate submission, invalid results, or unavailable parent
checkpoints.

`_scientific_value(value, key)` recursively converts a configuration into a
canonical JSON-compatible tree. Dictionary keys are sorted; paths become
strings; lists preserve order. It removes only execution identity: output,
SLURM, pipeline-run metadata, W&B IDs/names/groups, checkpoint location, host,
PID, timestamps, and generated log fields.

`scientific_config_hash(config)` serializes that tree with `json.dumps` using
sorted keys, compact separators, and `allow_nan=False`, then computes SHA-256.
The strict NaN rule prevents two non-portable floating-point representations
from becoming run identities.

`pipeline_hash(document)` hashes the pipeline definition after removing only
the loader-added `_pipeline_file`. State loading compares this hash so editing
a phase after submission cannot silently attach new intent to old jobs.

### Paths and lookups

`resolve_pipeline_path(document, value)` resolves absolute paths directly,
repository-style `outputs/...` and existing paths from the current working
directory, and other relative paths from the pipeline file's directory.

`pipeline_paths(document)` returns `(output_root, state_file)`, defaulting the
state file to `<output_root>/state.json`.

`phase_by_name(document, name)` performs a unique phase lookup and raises a
clear error rather than returning `None`.

`effective_selection(document, phase)` deep-copies global ranking policy,
overlays phase-specific selection, and inserts defaults. Deep copying matters:
ranking must never mutate the loaded pipeline definition.

### Phase sizing and expansion

`_phase_variants(phase)` converts each phase type into a uniform list of
`(variant_name, dotted_overrides, baseline_flag)` tuples. For Cartesian phases,
`itertools.product` forms a product only across that phase's parameters.
Conditions remain indivisible dictionaries and are never crossed with one
another. Fine-tune phases without conditions produce one variant.

`estimated_phase_counts(document)` propagates configured top-k counts rather
than configurations. A later estimate is `parents × variants`, so the planner
can report downstream sizes without pretending failed runs are already known.

`_apply_dotted(config, overrides)` delegates each dotted key to the existing
`fish_species.config.set_nested`. Reusing that function ensures pipeline YAML
and ordinary command-line overrides have the same nesting rules.

`_slug(value)` uses a regular expression to convert phase/condition names into
safe run-ID fragments. The configuration hash suffix, not the human fragment,
is the uniqueness authority.

`_normalise_parent(parent, base)` gives the first phase a synthetic `base`
parent and validates required lineage fields for real parents.

`expand_phase(document, phase, parents)` is the central scientific planner:

1. load the base config with the repository's `load_config`, including
   `extends`;
2. deep-copy each complete parent config;
3. apply phase, common, and variant overrides in that order;
4. disable internal sweep expanders;
5. attach parent checkpoint and fresh-optimizer fine-tuning settings where
   appropriate;
6. compute the scientific hash before adding ephemeral output/lineage fields;
7. reject duplicate hashes;
8. construct expected output/checkpoint paths;
9. attach `pipeline_run` provenance used by training and W&B.

The crucial property is step 2: a child is not reconstructed from the original
base. Every scientific choice made by the selected parent survives unless the
current phase explicitly overrides it.

`validate_generated_records(records)` calls the repository's canonical
`validate_config(..., workflow="training")` without requiring login-node data
paths or model downloads. It additionally enforces patch-size divisibility for
DINOv3 ViTs.

`validate_pipeline_semantics(document)` walks a deterministic top-k-sized
parent frontier through every phase and validates the actual generated
training configs. This catches incompatible later-phase overrides during
`plan`, while avoiding a global Cartesian explosion.

### Atomic state and immutable manifests

`_atomic_text(path, text)` writes a sibling temporary file then uses
`os.replace`. Replacement is atomic on one filesystem.

`atomic_write_json(path, value)` additionally flushes Python buffers and calls
`os.fsync` before replacement. A killed Python process therefore leaves either
the previous complete state or the new complete state, not half JSON.

`state_lock(state_path)` opens `<state>.lock` and uses `fcntl.flock(LOCK_EX)`.
Only state-changing commands take this advisory lock. Read-only `status` and
dry runs remain available while training is running.

`initial_state(document)` creates the schema-versioned state skeleton with
per-phase counters, job IDs, selections, attempts, and dynamic run maps.

`load_state(document, required)` parses JSON and compares `pipeline_hash`.
`required=False` supports a never-submitted pipeline; corrupt or mismatched
state is never treated as empty.

`write_state(document, state)` is the single state-save entry point and always
uses `atomic_write_json`.

`materialize_phase(document, phase, records)` writes resolved YAML files plus
`manifest.json` and `manifest.jsonl`. Repeating it is idempotent if bytes and
immutable fields match. A mismatch raises instead of overwriting scientific
history. Submission/completion changes are stored in state, leaving manifests
as stable identities.

`load_manifest(document, phase_name)` parses the manifest and checks its
pipeline hash.

`parent_records(document, state, phase)` resolves selected parent IDs back to
immutable YAML, dynamic metrics, and the selected checkpoint. It applies
`inherit_top_k` in stored rank order.

`plan_phase(document, state, phase)` combines parent loading, expansion,
semantic validation, materialization, and initial dynamic state. Repeating it
returns the existing immutable records rather than generating new identities.

### Result collection and retry classification

`_finite_metrics(value)` keeps numeric finite scalar values and converts them
to floats. Booleans, strings, NaN, and infinities cannot enter ranking.

`_failure_category(status, missing)` maps status text into `preempted`,
`infrastructure`, `cuda_oom`, `missing_dataset`, `configuration`, or generic
training failure. This classification drives retry—not a mutable run name.

`retry_decision(run, execution)` permits only missing output, infrastructure
failure, or preemption below `maximum_retries`. OOM, invalid configuration,
missing data, checkpoint mismatch, and invalid metrics are deliberately not
automatic retries.

`collect_local_results(document, state, phase)` checks, in order:

1. `run_status.json` exists and parses;
2. status and exit code are completed/successful;
3. configuration hash exactly matches the manifest;
4. metric JSON exists and the primary metric is finite;
5. the selected checkpoint exists.

It updates only dynamic state and returns a row for every run, including an
explicit rejection reason for failures and missing output.

`collect_wandb_results(document, state, phase)` lazily imports W&B, lists the
configured project, and indexes summaries by
`pipeline_run.configuration_hash`. It requires a finished run, finite primary
metric, and accessible checkpoint. W&B names are never identity keys.

`collect_results(...)` dispatches to local or W&B collection from YAML.

### Constraints and ranking

`_constraint_threshold(constraint, rows)` computes a literal threshold or a
fraction of the phase maximum. Coverage constraints therefore adapt to the
best represented-class count observed in that phase.

`_passes(value, operator, threshold)` implements the direct numeric comparison
operators.

`_compute_estimate(record)` uses `(stage-1 epochs + stage-2 epochs) × image
size²` as a deterministic, deliberately simple lower-compute tie breaker. It
does not claim to predict wall-clock runtime.

`rank_results(document, phase, rows)` performs the complete stable ordering:

1. build one baseline per parent when baseline-relative constraints exist;
2. apply every constraint and store reasons;
3. put valid runs before invalid runs;
4. compare primary metric in configured direction;
5. compare tie breakers within `tie_tolerance`;
6. prefer lower compute;
7. compare configuration hash lexicographically.

It stores both `raw_rank` and `selected_rank`. A raw rank records successful
output even if a scientific constraint rejects it; selected rank is limited to
valid top-k runs.

`write_leaderboard(...)` atomically writes complete JSON and a compact CSV.
`update_phase_ranking(...)` calls the ranker, stores selected IDs/best metric,
sets completed/partial/failed phase status, and writes both leaderboards.

`next_phase(document, current)` returns the ordered successor or `None`.
`status_summary(document, state)` turns raw state into user-facing counts,
selected parents, best metric, and next required action.

## 5. `sweeps/execution.py`: local processes and SLURM

`_yaml_hash(config)` computes the existing renderer's full resolved-YAML hash.
This differs intentionally from the scientific hash: renderer checksums detect
any artifact-byte change, while scientific hashes ignore execution metadata.

`_submission_config(config_path, cluster_config, phase_output, max_active)`
calls the existing `load_submission_config`. Cluster YAML still owns machine
resources; the pipeline overrides only result root, array concurrency, and the
old generic collection job (the phased collector replaces it).

`build_phase_submission_plan(records, cluster_config, phase_output,
max_active)` adapts pipeline records into existing immutable `RunSpec` and
`SubmissionPlan` dataclasses. It first asks `plan_submission` for validated
cluster/dependency defaults, then replaces its ordinary sweep specs with the
phase's exact run IDs and resolved configs.

`render_phase_bundle(...)` calls the existing `write_artifact_bundle`, which
renders scripts/configs/manifests and checksums them. Each retry gets a new
`attempt_NNN` bundle; prior attempts are never overwritten.

`submit_phase_slurm(...)` verifies/submits that bundle through
`submit_manifest`. Its injectable `SbatchClient` makes unit tests record argv
without touching a real scheduler.

`submit_auto_advance(...)` submits a CPU-only `sbatch --wrap` job with
`--dependency=afterany:<train-array>`. It sets project working directory and
`PYTHONPATH`, sources the cluster's configured Conda hook, activates its
configured environment, then invokes `pipeline advance --submit`. `afterany` is essential:
the collector must run after partial failures, not only perfect arrays.

`execute_local_records(records, environment)` calls the canonical trainer with
`subprocess.run` for each immutable config. It preserves `PYTHONPATH` and writes
a fallback failed status only if the trainer could not write one itself.

`cancel_jobs(job_ids, dry_run)` constructs explicit `scancel <numeric-id>`
commands. It never accepts directory globs or deletes state/results.

`active_slurm_jobs(job_ids)` calls `squeue` for only recorded train IDs. Manual
advance refuses to collect/resubmit while an array is active. The dependent
collector naturally sees no active upstream array.

## 6. `sweeps/pipeline.py`: command orchestration

`_print(value, as_json)` emits stable JSON for automation or readable YAML for
humans.

`plan_command(document, artifacts_dir)` validates every phase, reports exact
first/downstream counts, hashes, generated paths, ranking policy, and commands.
It writes only when an explicit planning-artifact directory is supplied.

`_records_for_submission(...)` filters immutable manifest rows using dynamic
status. Successful rows are always skipped. Failed/missing rows must satisfy
retry policy unless the user explicitly supplied `--force`.

`_submit_current(...)` plans an unplanned phase, rejects accidental duplicate
submission, marks an attempt, then dispatches local or SLURM execution. It
persists the expensive training array ID before attempting the optional
collector, preventing a collector error from losing array identity.

`submit_command(...)` loads/creates state under lock, finds the first incomplete
phase, and calls `_submit_current`. Its dry-run path performs no locking, state
write, artifact rendering, or scheduler call.

`_manual_parents(...)` validates user-provided parent IDs against the previous
phase and permits only successful runs.

`advance_command(...)` is read-only without `--submit`. In mutating mode it:

1. checks that recorded SLURM training jobs are no longer active;
2. collects and ranks the current phase;
3. retries only eligible failures;
4. stops if no valid parent exists;
5. otherwise materializes and submits the successor;
6. marks the whole pipeline complete after the final phase.

`resume_command(...)` starts a never-run pipeline or delegates existing state
to collection/advance. Dry resume reports successful runs that will be
preserved and incomplete runs requiring action.

`status_command(...)` reads state, performs a non-persistent collection view,
and uses `squeue` when available to distinguish active missing outputs from
truly missing completed output.

`cancel_command(...)` collects active train/collector IDs, calls `cancel_jobs`,
marks state cancelled, and retains all manifests/results.

`build_parser()` declares the six subcommands and safety flags. `main(argv)`
loads YAML, dispatches exactly one command, formats the result, and converts
known exceptions into exit code 2. The `if __name__ == "__main__"` block and
`sweeps/__main__.py` provide both supported module entry points.

## 7. Training integration

### `training.runner.make_experiment_run_name`

Returns `pipeline_run.run_id` when present, making the manifest's expected
directory exact. Ordinary training still uses the existing derived name.

### `training.runner._write_pipeline_result_files`

Writes validation/test summaries and completed status only for configurations
carrying a pipeline hash. Ordinary runs retain their legacy outputs. The status
contains the resolved checkpoint, epoch, selection metric/value, exit code, and
configuration hash required by strict collection.

### `training.runner.run_one`

The existing complete single-run lifecycle remains the training authority.
For `execution_mode: evaluation_only`, it loads the parent through existing
fine-tuning compatibility logic, evaluates validation and test loaders, writes
pipeline results, and returns before optimizer training. Normal/fine-tune runs
write pipeline results after their established best/last evaluation.

The runner now writes classification reports but does not compute, save, or log
confusion matrices.

### `training.cli.execute`

Wraps each canonical `run_one`. If a pipeline run raises, rank zero writes a
failure status with a categorized error and re-raises, preserving the original
nonzero job outcome. Non-pipeline behavior is unchanged.

### `training.epochs.run_hierarchy_epoch`

Adds `species_represented_species_count` from unique ground-truth species in an
evaluation split. This is the metric used by `phase_max × fraction` coverage
constraints.

### `logging.wandb_logger.WandbLogger.log_artifacts`

Filters `.pt`, `.pth`, `.ckpt`, and `.safetensors` before creating any W&B
artifact. There is no model-artifact branch. `wandb.log_model: true` is also
rejected by canonical config validation. Confusion-matrix logging is a no-op;
metrics, summaries, configuration, classification reports, and other
lightweight scientific files remain available.

## 8. Existing repository functions reused

| Existing function/class | Why it is reused |
| --- | --- |
| `config.load_config` | Resolves canonical YAML `extends`; prevents a second config loader |
| `config.deep_merge` | Merges experiment and cluster config inside SLURM loading |
| `config.set_nested` | Applies the same dotted-key semantics as existing sweeps |
| `config.validate_config` | Enforces classifier, prototype, contrastive, imbalance, taxonomy, and training compatibility |
| `slurm.load_submission_config` | Resolves cluster paths/resources/environment exactly as ordinary jobs do |
| `slurm.plan_submission` | Supplies validated canonical dependencies and renderer contract |
| `slurm.RunSpec` / `SubmissionPlan` | Keeps each array element equal to one resolved training run |
| `slurm.write_artifact_bundle` | Produces immutable scripts/configs/checksums |
| `slurm.submit_manifest` | Verifies bundle checksums before `sbatch` |
| `slurm.SbatchClient` | Separates real subprocess submission from mocked tests |
| `training.load_model_state_compat` | Loads legacy/single-head checkpoints with explicit compatibility reporting |
| `training.run_hierarchy_epoch` | Uses the same AMP/DDP-aware loss and metric path for training and evaluation-only runs |
| `results.save_json` | Preserves the repository's established JSON-writing format |
| `logging.create_wandb_logger` | Keeps one optional, failure-isolated W&B lifecycle per training run |

## 9. Standard-library and third-party building blocks

| Call | Role and safety property |
| --- | --- |
| `copy.deepcopy` | Prevents parent/pipeline mutation during inheritance |
| `itertools.product` | Builds only a phase-local Cartesian product |
| `hashlib.sha256` | Stable content identity and renderer checksums |
| `json.dumps/loads` | Canonical hashes, state, manifests, and result parsing |
| `yaml.safe_load/safe_dump` | YAML without arbitrary Python object construction |
| `pathlib.Path` | Explicit path joining/resolution/existence checks |
| `os.replace` | Atomic same-filesystem replacement |
| `os.fsync` | Flushes state bytes before atomic replacement |
| `fcntl.flock` | Serializes state-changing commands on a shared filesystem |
| `math.isfinite` | Rejects NaN/infinite ranking metrics |
| `functools.cmp_to_key` | Implements tolerance-aware multi-field ordering |
| `csv.DictWriter` | Creates a human-inspectable leaderboard |
| `argparse` | Defines validated command/subcommand syntax |
| `subprocess.run(list, shell=False)` | Executes tokenized trainer/sbatch/squeue/scancel commands without shell interpolation |
| `shlex.join` | Quotes the collector's displayed/wrapped child command |
| `wandb.Api().runs` | Reads optional remote summaries; matching still uses config hash |
| PyTorch/DDP/AMP APIs | Remain inside the canonical trainer; the planner never initializes CUDA |

## 10. Worked lineage example

Suppose `backbone_lr` selects run `backbone_lr-combination_004-abc123`.
Its resolved config contains ConvNeXt and learning rate `3e-5`. The next
`cosine_prototype_fixed` child begins with that exact resolved config, then
changes only classifier/prototype keys. Its manifest records:

```json
{
  "parent_phase": "backbone_lr",
  "parent_run_id": "backbone_lr-combination_004-abc123",
  "parent_configuration_hash": "abc123...",
  "parent_checkpoint": ".../best_model.pt",
  "phase_overrides": {
    "model.species_classifier.type": "cosine",
    "model.prototype_classifier.enabled": true
  }
}
```

Later, `taxonomic_inference` creates two evaluation-only configs pointing to
the same selected parent checkpoint. Their hashes differ because minimum-risk
inference is scientific configuration. The selected evaluation config then
feeds `fine_tune_320`; that child reloads the same model weights, changes image
size and learning rate, and uses fresh AdamW state. `fine_tune_384` inherits the
complete selected 320 config and repeats the process.

This lineage is why resuming is safe: identity comes from immutable content and
explicit parents, while job IDs and process lifetime are merely mutable state.
