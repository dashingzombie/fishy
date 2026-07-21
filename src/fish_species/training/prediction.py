"""Prediction export for unlabeled fish test and unseen filename lists."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .losses import infer_parent_label_from_child_label
from ..data.metadata import load_fish_prediction_metadata


def _decode(index_to_label: dict, indices: torch.Tensor) -> list[str]:
    return [str(index_to_label[int(index)]) for index in indices.cpu().tolist()]


def predict_unlabeled_split(
    model: torch.nn.Module,
    loader,
    index_to_label_by_task: dict[str, dict[int, str]],
    device: torch.device,
    *,
    use_amp: bool,
    enforce_hierarchy: bool,
    hierarchy_genus_weight: float,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> pd.DataFrame:
    """Return top-1 genus/species predictions and calibrated softmax scores."""
    model.eval()
    rows: list[dict] = []
    genus_map = index_to_label_by_task.get("genus")
    species_map = index_to_label_by_task.get("species")
    species_parent_indices: torch.Tensor | None = None

    if enforce_hierarchy and genus_map is not None and species_map is not None:
        genus_to_index = {label: index for index, label in genus_map.items()}
        parent_indices = []
        for index in range(len(species_map)):
            species = species_map[index]
            parent = infer_parent_label_from_child_label(species)
            if parent not in genus_to_index:
                raise ValueError(
                    f"Species {species!r} maps to missing genus {parent!r}."
                )
            parent_indices.append(genus_to_index[parent])
        species_parent_indices = torch.tensor(
            parent_indices, dtype=torch.long, device=device
        )

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(
                enabled=use_amp and device.type == "cuda",
                device_type=device.type,
                dtype=amp_dtype,
            ):
                logits = model(images)

            predictions: dict[str, list[str]] = {}
            confidences: dict[str, list[float]] = {}
            raw_genus: list[str] | None = None
            raw_consistency: list[bool] | None = None
            for task in index_to_label_by_task:
                task_logits = logits[task]
                probabilities = F.softmax(task_logits.float(), dim=1)
                confidence, indices = probabilities.max(dim=1)
                predictions[task] = _decode(
                    index_to_label_by_task[task], indices
                )
                confidences[task] = confidence.cpu().tolist()

            if (
                species_parent_indices is not None
                and "species" in logits
                and "genus" in logits
            ):
                species_log_probs = F.log_softmax(
                    logits["species"].float(), dim=1
                )
                genus_log_probs = F.log_softmax(
                    logits["genus"].float(), dim=1
                )
                compatible_genus_scores = genus_log_probs.index_select(
                    1, species_parent_indices
                )
                joint_scores = (
                    species_log_probs
                    + float(hierarchy_genus_weight) * compatible_genus_scores
                )
                joint_species_indices = joint_scores.argmax(dim=1)
                joint_genus_indices = species_parent_indices[
                    joint_species_indices
                ]
                raw_genus = predictions["genus"]
                raw_consistency = [
                    raw == infer_parent_label_from_child_label(species)
                    for raw, species in zip(
                        raw_genus, predictions["species"]
                    )
                ]
                predictions["species"] = _decode(
                    index_to_label_by_task["species"],
                    joint_species_indices,
                )
                predictions["genus"] = _decode(
                    index_to_label_by_task["genus"],
                    joint_genus_indices,
                )
                species_probs = F.softmax(
                    logits["species"].float(), dim=1
                )
                genus_probs = F.softmax(logits["genus"].float(), dim=1)
                confidences["species"] = species_probs.gather(
                    1, joint_species_indices[:, None]
                ).squeeze(1).cpu().tolist()
                confidences["genus"] = genus_probs.gather(
                    1, joint_genus_indices[:, None]
                ).squeeze(1).cpu().tolist()

            sample_ids = [str(item) for item in batch["sample_id"]]
            for row_index, sample_id in enumerate(sample_ids):
                row = {"image_id": sample_id}
                for task in index_to_label_by_task:
                    row[f"predicted_{task}"] = predictions[task][row_index]
                    row[f"{task}_confidence"] = confidences[task][row_index]
                if raw_genus is not None and raw_consistency is not None:
                    row["raw_predicted_genus"] = raw_genus[row_index]
                    row["raw_taxonomy_consistent"] = raw_consistency[row_index]
                rows.append(row)
    return pd.DataFrame(rows)


def build_submission_mapping(
    predictions: pd.DataFrame,
    expected_image_ids: list[str],
) -> dict[str, str]:
    """Build an exact filename-to-species mapping with complete key coverage."""
    required_columns = {"image_id", "predicted_species"}
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Cannot build prediction.json; missing prediction columns: "
            f"{sorted(missing_columns)}"
        )

    expected = [str(image_id) for image_id in expected_image_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("Official test split contains duplicate image filenames.")

    predicted_ids = predictions["image_id"].astype(str)
    expected_set = set(expected)
    official_predictions = predictions.loc[
        predicted_ids.isin(expected_set),
        ["image_id", "predicted_species"],
    ].copy()
    official_predictions["image_id"] = official_predictions["image_id"].astype(str)
    duplicates = sorted(
        official_predictions.loc[
            official_predictions["image_id"].duplicated(keep=False),
            "image_id",
        ].unique().tolist()
    )
    if duplicates:
        raise ValueError(
            "Cannot build prediction.json; duplicate predicted filenames: "
            f"{duplicates[:10]}"
        )

    prediction_by_id = {
        image_id: "" if pd.isna(label) else str(label).strip()
        for image_id, label in zip(
            official_predictions["image_id"],
            official_predictions["predicted_species"],
        )
    }
    missing = [image_id for image_id in expected if image_id not in prediction_by_id]
    if missing:
        raise ValueError(
            "Cannot build prediction.json; predictions are missing "
            f"{len(missing)} official test filenames. Examples: {missing[:10]}"
        )

    # Iterate over the official list so extra predictions are ignored and the
    # JSON has deterministic test-split ordering.
    submission = {
        image_id: prediction_by_id[image_id]
        for image_id in expected
    }
    empty_labels = [
        image_id for image_id, label in submission.items() if not label.strip()
    ]
    if empty_labels:
        raise ValueError(
            "Cannot build prediction.json; empty species labels for: "
            f"{empty_labels[:10]}"
        )
    return submission


def export_unlabeled_predictions(
    *,
    model: torch.nn.Module,
    bundle,
    out_dir: Path,
    checkpoint_name: str,
    device: torch.device,
    use_amp: bool,
    cfg: dict,
) -> dict[str, str]:
    """Write diagnostic CSVs and the required test ``prediction.json``."""
    loaders = bundle.prediction_loaders or {}
    if not loaders:
        return {}
    inference_cfg = cfg.get("inference", {}) or {}
    outputs: dict[str, str] = {}
    predicted_frames: dict[str, pd.DataFrame] = {}
    for split_name, loader in loaders.items():
        frame = predict_unlabeled_split(
            model,
            loader,
            bundle.index_to_label_by_task,
            device,
            use_amp=use_amp,
            enforce_hierarchy=bool(
                inference_cfg.get("enforce_hierarchy", True)
            ),
            hierarchy_genus_weight=float(
                inference_cfg.get("hierarchy_genus_weight", 1.0)
            ),
            amp_dtype=(
                torch.bfloat16 if cfg.get("training", {}).get(
                    "amp_dtype", "bfloat16"
                ) == "bfloat16" else torch.float16
            ),
        )
        path = out_dir / f"predictions_{split_name}_{checkpoint_name}.csv"
        frame.to_csv(path, index=False)
        predicted_frames[split_name] = frame
        outputs[f"{split_name}_csv"] = str(path)
        print(f"Saved {len(frame)} {split_name} predictions to {path}")

    if "test" not in predicted_frames:
        raise ValueError(
            "inference.splits must include 'test' to create prediction.json."
        )
    official_test = load_fish_prediction_metadata(cfg, ["test"])["test"]
    submission = build_submission_mapping(
        predicted_frames["test"],
        official_test["image_id"].astype(str).tolist(),
    )
    submission_path = out_dir / "prediction.json"
    submission_path.write_text(
        json.dumps(submission, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    outputs["submission_json"] = str(submission_path)
    print(
        f"Saved complete test submission with {len(submission)} keys to "
        f"{submission_path}"
    )
    return outputs


__all__ = [
    "build_submission_mapping",
    "export_unlabeled_predictions",
    "predict_unlabeled_split",
]
