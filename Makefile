.DEFAULT_GOAL := help

PYTHON ?= python
EXPERIMENT ?= standard
CONFIG ?= configs/experiments/$(EXPERIMENT).yaml
CLUSTER ?= configs/clusters/genome.yaml
TRAIN_CONFIG ?= config.yaml
RESULTS_ROOT ?= outputs
MAX_ACTIVE ?=
MODEL ?=
GPUS ?=
GHPC_NODES ?=
CHECKPOINT ?=
ARTIFACTS_DIR ?= logs/generated/plan-$(shell date -u +%Y%m%dT%H%M%S%N)
PIPELINE ?= configs/sweeps/fish_long_tail_pipeline.yaml
CLUSTER_CONFIG ?= configs/clusters/ghpc.yaml

SLURM = PYTHONPATH=src $(PYTHON) -m fish_species.slurm
CLUSTER_ARG = $(if $(strip $(CLUSTER)),--cluster-config "$(CLUSTER)",)
PLAN_OVERRIDE_VALUES = slurm.paths.results_root=$(RESULTS_ROOT) \
	$(if $(strip $(MAX_ACTIVE)),slurm.array.max_active=$(MAX_ACTIVE),) \
	$(if $(strip $(MODEL)),model.name=$(MODEL),) \
	$(if $(strip $(GPUS)),slurm.gpus_per_task=$(GPUS) training.distributed.enabled=$(if $(filter 1,$(GPUS)),false,true),) \
	$(if $(strip $(GHPC_NODES)),slurm.scratch.nodes=$(GHPC_NODES),)
PLAN_OVERRIDE_ARGS = --override "slurm.paths.results_root=$(RESULTS_ROOT)" \
	$(if $(strip $(MAX_ACTIVE)),--override "slurm.array.max_active=$(MAX_ACTIVE)",) \
	$(if $(strip $(MODEL)),--override "model.name=$(MODEL)",) \
	$(if $(strip $(GPUS)),--override "slurm.gpus_per_task=$(GPUS)" --override "training.distributed.enabled=$(if $(filter 1,$(GPUS)),false,true)",) \
	$(if $(strip $(GHPC_NODES)),--override "slurm.scratch.nodes=$(GHPC_NODES)",)
TRAIN_OVERRIDE_VALUES = sweep.enabled=false \
	matched_condition_training.enabled=false \
	$(if $(strip $(MODEL)),model.name=$(MODEL),) \
	$(if $(strip $(GPUS)),training.distributed.enabled=$(if $(filter 1,$(GPUS)),false,true),) \
	$(if $(filter-out file undefined,$(origin RESULTS_ROOT)),output.out_dir=$(RESULTS_ROOT),)
TRAIN_OVERRIDE_ARGS = $(if $(strip $(TRAIN_OVERRIDE_VALUES)),--override $(TRAIN_OVERRIDE_VALUES),)
TRAIN_LAUNCH = PYTHONPATH=src $(if $(and $(strip $(GPUS)),$(filter-out 1,$(GPUS))),torchrun --standalone --nproc_per_node=$(GPUS) -m fish_species.training,$(PYTHON) -m fish_species.training)

.PHONY: help validate inspect dry-run train fine-tune-320 fine-tune-384 submit status collect sweep-plan sweep-submit sweep-status sweep-advance sweep-resume sweep-cancel test clean-generated

help: ## Show the supported repository commands.
	@echo "Fish Species commands"
	@echo
	@echo "  make validate           Validate the experiment and cluster plan."
	@echo "  make inspect            Print resolved configuration and run counts."
	@echo "  make dry-run            Render a plan without scheduler submission."
	@echo "  make train              Run one canonical local training command."
	@echo "  make fine-tune-320      Resume a 224 px checkpoint at 320 px."
	@echo "  make fine-tune-384      Resume a 224/320 px checkpoint at 384 px."
	@echo "  make submit             Explicitly render and submit to SLURM."
	@echo "  make status             Summarise filesystem and scheduler status."
	@echo "  make collect            Re-run canonical result aggregation."
	@echo "  make sweep-plan         Validate and inspect a phased sweep pipeline."
	@echo "  make sweep-submit       Submit its first incomplete phase."
	@echo "  make sweep-status       Inspect resumable pipeline state."
	@echo "  make sweep-advance      Collect/rank and submit the next phase."
	@echo "  make sweep-resume       Resume without repeating successful runs."
	@echo "  make test               Run the complete CPU-only unittest suite."
	@echo "  make clean-generated    Remove only generated Slurm plans."
	@echo
	@echo "Variables: CONFIG CLUSTER TRAIN_CONFIG RESULTS_ROOT MAX_ACTIVE MODEL"
	@echo "           EXPERIMENT (standard, hierarchy, dual_cue, colour_transforms,"
	@echo "                       patch_shuffle_matrix, persistent_hierarchy)"
	@echo "           ARTIFACTS_DIR CHECKPOINT"
	@echo "           GPUS GHPC_NODES (YAML list, e.g. '[gpu001,gpu002]')"
	@echo "           PYTHON"
	@echo
	@echo "Examples:"
	@echo "  make dry-run EXPERIMENT=patch_shuffle_matrix"
	@echo "  make submit EXPERIMENT=dual_cue CLUSTER=configs/clusters/genome.yaml"
	@echo "  make train MODEL=dinov3_vits16 GPUS=2"
	@echo "  make fine-tune-320 CHECKPOINT=outputs/.../best_model.pt"

validate: ## Validate configuration and prove the plan is internally consistent.
	@$(SLURM) validate --config "$(CONFIG)" $(CLUSTER_ARG) $(PLAN_OVERRIDE_ARGS)

inspect: ## Print the resolved submission configuration and concise plan summary.
	@$(SLURM) inspect --config "$(CONFIG)" $(CLUSTER_ARG) $(PLAN_OVERRIDE_ARGS)

dry-run: ## Render self-contained SLURM artifacts without calling sbatch.
	$(SLURM) launch --dry-run --config "$(CONFIG)" $(CLUSTER_ARG) \
		$(PLAN_OVERRIDE_ARGS) --artifacts-dir "$(ARTIFACTS_DIR)"

train: ## Run one local process through the canonical trainer.
	$(TRAIN_LAUNCH) --config "$(TRAIN_CONFIG)" \
		--single-run $(TRAIN_OVERRIDE_ARGS)

fine-tune-320: ## Resume a lower-resolution checkpoint and fine-tune at 320 px.
	@test -n "$(CHECKPOINT)" || (echo "CHECKPOINT is required" >&2; exit 2)
	$(TRAIN_LAUNCH) --config configs/experiments/fine_tune_320.yaml --single-run \
		--override fine_tuning.checkpoint_path="$(CHECKPOINT)" $(TRAIN_OVERRIDE_VALUES)

fine-tune-384: ## Resume a lower-resolution checkpoint and fine-tune at 384 px.
	@test -n "$(CHECKPOINT)" || (echo "CHECKPOINT is required" >&2; exit 2)
	$(TRAIN_LAUNCH) --config configs/experiments/fine_tune_384.yaml --single-run \
		--override fine_tuning.checkpoint_path="$(CHECKPOINT)" $(TRAIN_OVERRIDE_VALUES)

submit: ## Explicitly render and submit the validated plan to SLURM.
	$(SLURM) launch --submit --config "$(CONFIG)" $(CLUSTER_ARG) \
		$(PLAN_OVERRIDE_ARGS) --artifacts-dir "$(ARTIFACTS_DIR)"

status: ## Summarise jobs and filesystem-derived run state.
	PYTHONPATH=src $(PYTHON) -m fish_species.slurm status \
		--results-root "$(RESULTS_ROOT)"

collect: ## Aggregate existing results without retraining.
	PYTHONPATH=src $(PYTHON) -m fish_species.slurm collect \
		--results-root "$(RESULTS_ROOT)"

sweep-plan: ## Validate and report a sequential sweep without writing execution state.
	PYTHONPATH=src $(PYTHON) -m fish_species.sweeps.pipeline plan \
		--pipeline "$(PIPELINE)"

sweep-submit: ## Submit the first incomplete pipeline phase.
	PYTHONPATH=src $(PYTHON) -m fish_species.sweeps.pipeline submit \
		--pipeline "$(PIPELINE)" --cluster-config "$(CLUSTER_CONFIG)"

sweep-status: ## Display pipeline phase, jobs, result counts, and next action.
	PYTHONPATH=src $(PYTHON) -m fish_species.sweeps.pipeline status \
		--pipeline "$(PIPELINE)"

sweep-advance: ## Collect/rank the current phase and submit its successor.
	PYTHONPATH=src $(PYTHON) -m fish_species.sweeps.pipeline advance \
		--pipeline "$(PIPELINE)" --cluster-config "$(CLUSTER_CONFIG)" --submit

sweep-resume: ## Resume from atomic state without repeating successful runs.
	PYTHONPATH=src $(PYTHON) -m fish_species.sweeps.pipeline resume \
		--pipeline "$(PIPELINE)" --cluster-config "$(CLUSTER_CONFIG)"

sweep-cancel: ## Cancel active pipeline-owned jobs while retaining state.
	PYTHONPATH=src $(PYTHON) -m fish_species.sweeps.pipeline cancel \
		--pipeline "$(PIPELINE)"

test: ## Run every standard-library test without external data or GPUs.
	PYTHONPATH=.:src $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

clean-generated: ## Delete only repository-local generated plans and indexes.
	@root="$$(realpath -m slurm/generated)"; \
		expected="$$(realpath -m "$(CURDIR)/slurm/generated")"; \
		test "$$root" = "$$expected"; \
		if test -d "$$root"; then find "$$root" -mindepth 1 -maxdepth 1 ! -name .gitignore -exec rm -rf -- {} +; fi
