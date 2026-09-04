"""Protocol validation and benchmark metrics for the final experiment plan.

This module intentionally contains no model or dataset downloads.  It is the
small, CPU-safe contract shared by preparation, feature extraction, and probe
training jobs on HPC.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml


EXPECTED_MODELS = frozenset({"qwen3_vl", "internvl3_5", "minicpm_v45", "vjepa2_vitl"})
EXPECTED_DATASETS = frozenset({"ssv2", "nextqa", "egotempo", "vstar"})
NEXTQA_CATEGORIES = frozenset({"temporal", "causal", "descriptive"})
FORBIDDEN_PROTOCOL_TERMS = frozenset(
    {"state_sufficiency_score", "composite_metric", "mean_pool", "max_pool", "prefix_fraction"}
)


class ProtocolError(ValueError):
    """Raised when a run would silently diverge from the written protocol."""


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def load_protocol_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ProtocolError(f"protocol config must be a mapping: {resolved}")
    missing = {"protocol_version", "models", "datasets", "probes", "evaluations"} - set(config)
    if missing:
        raise ProtocolError(f"protocol config is missing required sections: {sorted(missing)}")
    forbidden = sorted(
        term for term in FORBIDDEN_PROTOCOL_TERMS if any(term in key for key in _walk_keys(config))
    )
    if forbidden:
        raise ProtocolError(f"protocol contains forbidden self-defined constructs: {forbidden}")
    models = config["models"]
    datasets = config["datasets"]
    if not isinstance(models, dict) or set(models) != EXPECTED_MODELS:
        raise ProtocolError(f"models must be exactly {sorted(EXPECTED_MODELS)}")
    if not isinstance(datasets, dict) or set(datasets) != EXPECTED_DATASETS:
        raise ProtocolError(f"datasets must be exactly {sorted(EXPECTED_DATASETS)}")
    nextqa = datasets["nextqa"]
    if nextqa.get("official_splits") != {"train": 3870, "val": 570, "test": 1000}:
        raise ProtocolError("NExT-QA official video split counts must remain unchanged")
    if config["evaluations"].get("nextqa_metrics") != [
        "overall_accuracy",
        "temporal_accuracy",
        "causal_accuracy",
        "descriptive_accuracy",
    ]:
        raise ProtocolError("NExT-QA metrics must be the four protocol metrics")
    return resolved, config


def normalized_depth(layer_index: int, total_layers: int) -> float:
    """Return the plotting-only normalized depth ``layer_index / total_layers``."""

    if total_layers <= 0:
        raise ProtocolError("total_layers must be positive")
    if not 0 <= layer_index < total_layers:
        raise ProtocolError(f"layer_index {layer_index} is outside [0, {total_layers})")
    return layer_index / total_layers


def top1_accuracy(targets: Sequence[int], predictions: Sequence[int]) -> float:
    if not targets or len(targets) != len(predictions):
        raise ProtocolError("targets and predictions must be non-empty and equally sized")
    return sum(target == prediction for target, prediction in zip(targets, predictions)) / len(targets)


def nextqa_metrics(
    targets: Sequence[int], predictions: Sequence[int], categories: Sequence[str]
) -> dict[str, float]:
    """Compute only NExT-QA's official overall and C/T/D accuracies."""

    if len(targets) != len(predictions) or len(targets) != len(categories) or not targets:
        raise ProtocolError("NExT-QA targets, predictions, and categories must have equal non-zero length")
    if any(category not in NEXTQA_CATEGORIES for category in categories):
        raise ProtocolError(f"unknown NExT-QA category: {sorted(set(categories) - NEXTQA_CATEGORIES)}")
    metrics = {"overall_accuracy": top1_accuracy(targets, predictions)}
    for category in ("temporal", "causal", "descriptive"):
        selected = [index for index, value in enumerate(categories) if value == category]
        if not selected:
            raise ProtocolError(f"NExT-QA split has no examples in category {category}")
        metrics[f"{category}_accuracy"] = top1_accuracy(
            [targets[index] for index in selected], [predictions[index] for index in selected]
        )
    return metrics
