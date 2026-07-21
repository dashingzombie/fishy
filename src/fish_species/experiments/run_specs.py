from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config.loading import load_config
from ..config.normalization import normalize_config
from ..config.sweeps import expand_sweep_items
from ..config.validation import validate_config
from ..slurm.planning import generate_external_specs


def write_run_specs(config_path: Path, run_specs_dir: Path, sweep_plan_path: Path) -> int:
    config = load_config(config_path)
    # Validate before creating directories or removing stale specifications.
    # Run-spec generation intentionally does not require local data paths or a
    # torchvision import: it is a dry-run/cluster-submission workflow.
    validate_config(
        config,
        workflow="run_specs",
        check_paths=False,
        check_model_registry=False,
    )
    canonical = normalize_config(config)
    items = expand_sweep_items(canonical)
    raw_specs = generate_external_specs(config)

    run_specs_dir.mkdir(parents=True, exist_ok=True)
    for old in run_specs_dir.glob("run_*.args"):
        old.unlink()

    plan_lines = [
        "run_index\tarray_name\tmodel\ttrain_condition\ttrain_transform\toverrides"
    ]
    for run_index, (array_name, override_lines, model_name, condition_name) in enumerate(raw_specs):
        item = items[run_index]
        transform_name = (
            str(item.condition["transform"])
            if item.condition is not None
            else "original"
        )
        (run_specs_dir / f"{array_name}.args").write_text(
            "\n".join(override_lines) + "\n"
        )
        plan_lines.append("\t".join([
            str(run_index), array_name, model_name, condition_name,
            transform_name, " ".join(override_lines),
        ]))

    sweep_plan_path.write_text("\n".join(plan_lines) + "\n")
    combinations = []
    seen_combinations = set()
    for item in items:
        combination = item.parameter_values
        signature = json.dumps(combination, sort_keys=True)
        if signature not in seen_combinations:
            seen_combinations.add(signature)
            combinations.append(combination)
    conditions = [
        item.condition for item in items if item.condition is not None
    ]
    unique_conditions = list({
        condition["name"]: condition for condition in conditions
    }.values())
    metadata = {
        "n_sweep_combinations": len(combinations),
        "n_unique_training_conditions": len(unique_conditions) or 1,
        "n_total_runs": len(raw_specs),
        "conditions": unique_conditions,
        "sweep_combinations": combinations,
    }
    (sweep_plan_path.parent / "dual_cue_experiment_plan.json").write_text(
        json.dumps(metadata, indent=2)
    )
    return len(raw_specs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("run_specs_dir")
    parser.add_argument("sweep_plan")
    args = parser.parse_args()
    count = write_run_specs(
        Path(args.config), Path(args.run_specs_dir), Path(args.sweep_plan)
    )
    print(count)
