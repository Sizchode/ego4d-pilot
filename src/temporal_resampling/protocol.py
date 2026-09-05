"""Immutable protocol checks and result metrics for the resampling pilot."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


EXPECTED_MODELS = frozenset({"dinov3", "vjepa2", "qwen3_vl"})
EXPECTED_DATASETS = frozenset({"ssv2", "nextqa"})
EXPECTED_CONDITIONS = ("C0", "C1", "C2", "C3")


class ProtocolError(ValueError):
    """Raised when a run would silently diverge from the experiment plan."""


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ProtocolError("config must be a mapping")
    required = {"protocol_version", "models", "datasets", "sampling", "probes"}
    if not required.issubset(config):
        raise ProtocolError(f"config missing sections: {sorted(required - set(config))}")
    if set(config["models"]) != EXPECTED_MODELS or set(config["datasets"]) != EXPECTED_DATASETS:
        raise ProtocolError("config must contain exactly the three model families and two pilot datasets")
    sampling = config["sampling"]
    if sampling.get("conditions") != list(EXPECTED_CONDITIONS) or sampling.get("canonical_condition") != "C0":
        raise ProtocolError("sampling conditions must be C0 canonical followed by C1/C2/C3")
    if sampling.get("record_real_timestamps") is not True or sampling.get("record_frame_indices") is not True:
        raise ProtocolError("real timestamps must be recorded")
    return resolved, config


def load_canonical_schedules(path: str | Path) -> tuple[dict[str, str], dict[str, list[float]]]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("schedules"), dict):
        raise ProtocolError("canonical schedule file must contain provenance and a schedules mapping")
    provenance = {key: payload.get(key) for key in ("source", "revision", "dataset", "model")}
    if any(not isinstance(value, str) or not value for value in provenance.values()):
        raise ProtocolError("canonical schedule provenance requires source, revision, dataset, and model")
    schedules = payload["schedules"]
    if any(not isinstance(sample_id, str) or not isinstance(values, list) for sample_id, values in schedules.items()):
        raise ProtocolError("canonical schedules must map sample IDs to timestamp lists")
    return provenance, schedules


def top1_accuracy(targets: Sequence[int], predictions: Sequence[int]) -> float:
    if not targets or len(targets) != len(predictions):
        raise ProtocolError("targets and predictions must be non-empty and equally sized")
    return sum(target == prediction for target, prediction in zip(targets, predictions)) / len(targets)


def nextqa_metrics(targets: Sequence[int], predictions: Sequence[int], categories: Sequence[str]) -> dict[str, float]:
    if len(targets) != len(predictions) or len(targets) != len(categories) or not targets:
        raise ProtocolError("NExT-QA targets, predictions, and categories must align")
    result = {"overall_accuracy": top1_accuracy(targets, predictions)}
    for category in ("temporal", "causal", "descriptive"):
        indices = [index for index, value in enumerate(categories) if value == category]
        if not indices:
            raise ProtocolError(f"NExT-QA split has no {category} examples")
        result[f"{category}_accuracy"] = top1_accuracy(
            [targets[index] for index in indices], [predictions[index] for index in indices]
        )
    return result


def cross_sampling_result(
    condition_to_targets: dict[str, Sequence[int]], condition_to_predictions: dict[str, Sequence[int]]
) -> dict[str, float]:
    missing = (set(EXPECTED_CONDITIONS) - set(condition_to_targets)) | (set(EXPECTED_CONDITIONS) - set(condition_to_predictions))
    if missing:
        raise ProtocolError(f"cross-sampling result is missing conditions: {sorted(missing)}")
    metrics = {
        condition: top1_accuracy(condition_to_targets[condition], condition_to_predictions[condition])
        for condition in EXPECTED_CONDITIONS
    }
    return {
        **{f"top1_{condition}": metrics[condition] for condition in EXPECTED_CONDITIONS},
        **{f"delta_{condition}": metrics[condition] - metrics["C0"] for condition in EXPECTED_CONDITIONS[1:]},
    }
