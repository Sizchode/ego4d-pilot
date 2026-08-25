"""Load and validate the immutable scientific protocol manifest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PROVIDERS = {"qwen3", "vjepa2", "videomaev2"}


class ProtocolError(ValueError):
    """Raised when a protocol would make runs incomparable or ambiguous."""


@dataclass(frozen=True)
class Protocol:
    """Validated protocol and its source path."""

    path: Path
    data: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.data["project"]["seed"])

    def model(self, provider: str) -> dict[str, Any]:
        if provider not in _PROVIDERS:
            raise ProtocolError(f"unknown provider: {provider}")
        return self.data["models"][provider]


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ProtocolError(f"missing {context}.{key}")
    return mapping[key]


def _validate_revision(value: Any, context: str) -> None:
    if not isinstance(value, str) or not _COMMIT_PATTERN.fullmatch(value):
        raise ProtocolError(f"{context} must be an immutable 40-character Git revision")


def _validate_qwen_architecture(model: dict[str, Any]) -> None:
    architecture = _require(model, "expected_architecture", "models.qwen3")
    representations = _require(model, "representations", "models.qwen3")
    vision_depth = int(_require(architecture, "vision_depth", "models.qwen3.expected_architecture"))
    language_depth = int(_require(architecture, "language_depth", "models.qwen3.expected_architecture"))
    deepstack = list(_require(architecture, "deepstack_visual_indexes", "models.qwen3.expected_architecture"))
    vision_blocks = list(_require(representations, "vision_blocks", "models.qwen3.representations"))
    language_layers = list(
        _require(representations, "post_deepstack_language_layers", "models.qwen3.representations")
    )

    if len(set(vision_blocks)) != len(vision_blocks) or any(i < 0 or i >= vision_depth for i in vision_blocks):
        raise ProtocolError("Qwen3 vision extraction blocks must be unique valid zero-based block indexes")
    if language_layers != list(range(len(deepstack))):
        raise ProtocolError(
            "post-DeepStack language extraction must cover each architecture-defined injection stage"
        )
    if any(i < 0 or i >= language_depth for i in language_layers):
        raise ProtocolError("Qwen3 language extraction layer is outside the declared architecture")
    if not set(deepstack).issubset(vision_blocks):
        raise ProtocolError("all DeepStack source blocks must also be extracted as vision representations")
    if max(vision_blocks) != vision_depth - 1:
        raise ProtocolError("the final Qwen3 vision block must be included")


def _validate_probe_dimensions(data: dict[str, Any]) -> None:
    heads = int(data["probe"]["architecture"]["heads"])
    models = data["models"]
    dimensions = {
        "qwen3_vision": int(models["qwen3"]["expected_architecture"]["vision_hidden_size"]),
        "qwen3_merged": int(models["qwen3"]["expected_architecture"]["vision_output_size"]),
        "vjepa2": int(models["vjepa2"]["expected_hidden_size"]),
        "videomaev2": int(models["videomaev2"]["expected_hidden_size"]),
    }
    invalid = [name for name, dimension in dimensions.items() if dimension % heads]
    if invalid:
        raise ProtocolError(f"probe head count does not divide representation dimensions: {invalid}")


def validate_protocol_data(data: dict[str, Any]) -> None:
    """Validate all protocol fields that affect scientific comparability."""

    if data.get("schema_version") != 1:
        raise ProtocolError("schema_version must be 1")
    for section in ("project", "upstreams", "models", "datasets", "video", "probe", "language_evaluation"):
        _require(data, section, "protocol")

    for name, upstream in data["upstreams"].items():
        _require(upstream, "repository", f"upstreams.{name}")
        _validate_revision(_require(upstream, "revision", f"upstreams.{name}"), f"upstreams.{name}.revision")

    if set(data["models"]) != _PROVIDERS:
        raise ProtocolError(f"models must be exactly {sorted(_PROVIDERS)}")
    for name, model in data["models"].items():
        if model.get("provider") != name:
            raise ProtocolError(f"models.{name}.provider must equal {name}")
        _validate_revision(_require(model, "revision", f"models.{name}"), f"models.{name}.revision")
    if data["models"]["vjepa2"].get("representation") != "last_hidden_state":
        raise ProtocolError("V-JEPA 2 must expose its official last_hidden_state tokens")
    if data["models"]["videomaev2"].get("representation") != "last_encoder_tokens":
        raise ProtocolError(
            "VideoMAE V2 must expose the final encoder tokens before its released single-token readout"
        )

    _validate_qwen_architecture(data["models"]["qwen3"])
    _validate_probe_dimensions(data)

    frames = int(data["video"]["frames"])
    if frames != 16:
        raise ProtocolError("the paper-spec AoT protocol requires exactly 16 frames")
    if data["video"].get("reversal") != "virtual_reverse_before_decode":
        raise ProtocolError("temporal reversal must be applied to sampling indexes before decoding")
    if data["video"]["train_sampling"].get("mode") != "rand":
        raise ProtocolError("training sampling must use the published TAT random-within-uniform-bins mode")
    if data["video"]["evaluation_sampling"].get("mode") != "middle":
        raise ProtocolError("evaluation sampling must use the published TAT middle-of-uniform-bins mode")
    spatial = data["video"].get("spatial_preprocessing", {})
    if spatial.get("implementation") != "TAT-VQA/dataset/__init__.py:create_dataset":
        raise ProtocolError("spatial preprocessing must be sourced from the published TAT dataset pipeline")
    if spatial.get("train") != {
        "transform": "random_resized_crop_and_horizontal_flip",
        "size": 224,
        "scale": [0.5, 1.0],
        "ratio": [0.75, 1.3333333333333333],
        "horizontal_flip_probability": 0.5,
        "interpolation": "bicubic",
    }:
        raise ProtocolError("training spatial preprocessing differs from published TAT")
    if spatial.get("evaluation") != {
        "transform": "resize",
        "height": 224,
        "width": 224,
        "interpolation": "bicubic",
    }:
        raise ProtocolError("evaluation spatial preprocessing differs from published TAT")

    ssv2 = data["datasets"]["ssv2"]
    direction = ssv2["direction_assignment"]
    if (
        ssv2.get("split") != "train"
        or direction.get("implementation") != "TAT-VQA/aot.py:assign_balanced_aot"
        or direction.get("method") != "balanced_label_shuffle_over_sorted_video_names"
    ):
        raise ProtocolError("SSv2 must use its released train split and the published TAT balanced assignment")
    aot = data["datasets"]["aot_psyphybench"]
    if int(aot["expected_source_clips"]) * 2 != int(aot["expected_examples"]):
        raise ProtocolError("AoT-PsyPhyBENCH must pair every source clip forward and backward")

    expected_tasks = ["vitatecs_direction", "tvbench", "mvbench"]
    if data["language_evaluation"].get("tasks") != expected_tasks:
        raise ProtocolError(f"first-pilot language tasks must remain ordered as {expected_tasks}")
    for task in expected_tasks:
        if data["datasets"][task].get("lmms_task") != task:
            raise ProtocolError(f"dataset {task} must use the matching published lmms-eval task")

    architecture = data["probe"]["architecture"]
    optimization = data["probe"]["optimization"]
    if (
        data["probe"].get("implementation")
        != "facebookresearch/vjepa2/src/models/attentive_pooler.py:AttentiveClassifier"
    ):
        raise ProtocolError("probe must use the pinned official V-JEPA 2 attentive classifier")
    if architecture != {
        "type": "attentive_classifier",
        "queries": 1,
        "cross_attention_layers": 1,
        "heads": 16,
        "mlp_ratio": 4.0,
        "classes": 2,
        "qkv_bias": True,
        "complete_block": True,
        "initialization_standard_deviation": 0.02,
    }:
        raise ProtocolError("probe architecture differs from the disclosed TAT attentive probe")
    if int(optimization["epochs"]) != 1 or float(optimization["learning_rate"]) != 1e-4:
        raise ProtocolError("probe epochs and learning rate must match the disclosed TAT settings")
    if int(optimization["batch_size_per_rank"]) != 4:
        raise ProtocolError("probe batch size per rank must match the disclosed TAT setting")


def load_protocol(path: str | Path) -> Protocol:
    """Load YAML and return a validated protocol."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ProtocolError(f"protocol does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ProtocolError("protocol root must be a mapping")
    validate_protocol_data(data)
    return Protocol(path=resolved, data=data)
