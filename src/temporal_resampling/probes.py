"""Cross-sampling probes: fit on C0 once, evaluate C0--C3 without refitting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from .protocol import ProtocolError, cross_sampling_result


def _validate_labels(labels: torch.Tensor, classes: int) -> None:
    if labels.ndim != 1 or not len(labels) or any(int(value) < 0 or int(value) >= classes for value in labels.tolist()):
        raise ProtocolError("probe labels must be a non-empty vector in the configured class range")


def _mean_frame_features(features: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        if features.ndim != 3:
            raise ProtocolError("DINO representation must have shape [examples, frames, channels]")
        return features.mean(dim=1)
    if not features or any(feature.ndim != 2 for feature in features):
        raise ProtocolError("DINO representation must have shape [examples, frames, channels]")
    channels = {int(feature.shape[1]) for feature in features}
    if len(channels) != 1:
        raise ProtocolError("DINO frame features must share channel dimensions")
    return torch.stack([feature.mean(dim=0) for feature in features])


class AttentiveProbe(nn.Module):
    """One-query attentive probe matching the V-JEPA frozen-probe family."""

    def __init__(self, channels: int, heads: int, classes: int, mlp_ratio: float = 4.0):
        super().__init__()
        if channels % heads:
            raise ProtocolError("probe channels must be divisible by heads")
        self.heads = heads
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, channels))
        self.input_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key_value = nn.Linear(channels, 2 * channels)
        self.query_norm = nn.LayerNorm(channels)
        self.mlp_1 = nn.Linear(channels, int(channels * mlp_ratio))
        self.mlp_2 = nn.Linear(int(channels * mlp_ratio), channels)
        self.linear = nn.Linear(channels, classes)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ProtocolError("attentive probe input must have shape [examples, tokens, channels]")
        batch, token_count, channels = tokens.shape
        query = self.query(self.query_tokens.expand(batch, -1, -1)).reshape(batch, 1, self.heads, channels // self.heads).transpose(1, 2)
        key_value = self.key_value(self.input_norm(tokens)).reshape(batch, token_count, 2, self.heads, channels // self.heads)
        key, value = key_value.unbind(dim=2)
        attended = functional.scaled_dot_product_attention(query, key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3))
        pooled = attended.transpose(1, 2).reshape(batch, 1, channels)
        pooled = pooled + self.mlp_2(functional.gelu(self.mlp_1(self.query_norm(pooled))))
        return self.linear(pooled.squeeze(1))


def _batches(tokens: list[torch.Tensor], batch_size: int) -> list[tuple[list[int], torch.Tensor]]:
    if any(token.ndim != 2 for token in tokens):
        raise ProtocolError("distributed tokens must have shape [tokens, channels]")
    channels = {int(token.shape[1]) for token in tokens}
    if len(channels) != 1:
        raise ProtocolError("distributed tokens must share channel dimensions")
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, token in enumerate(tokens):
        buckets[int(token.shape[0])].append(index)
    return [
        (selected, torch.stack([tokens[index] for index in selected]))
        for indices in buckets.values()
        for start in range(0, len(indices), batch_size)
        for selected in [indices[start : start + batch_size]]
    ]


def train_cross_sampling_probe(
    train_features: Any,
    train_labels: torch.Tensor,
    validation_features: dict[str, Any],
    validation_labels: dict[str, torch.Tensor],
    *,
    classes: int,
    readout: str,
    heads: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, float]:
    """Train only on C0 and return exactly Top1(C0..C3) plus deltas."""

    if set(validation_features) != {"C0", "C1", "C2", "C3"} or set(validation_labels) != set(validation_features):
        raise ProtocolError("cross-sampling evaluation requires C0, C1, C2, and C3 validation shards")
    _validate_labels(train_labels, classes)
    if epochs <= 0 or batch_size <= 0:
        raise ProtocolError("probe epochs and batch_size must be positive")
    if readout == "mean":
        train_input = _mean_frame_features(train_features)
        validation_input = {condition: _mean_frame_features(value) for condition, value in validation_features.items()}
        channels = int(train_input.shape[1])
        classifier: Any = nn.Linear(channels, classes).to(train_input.device)
    elif readout == "attentive":
        train_input = list(train_features)
        validation_input = {condition: list(value) for condition, value in validation_features.items()}
        if not train_input:
            raise ProtocolError("C0 distributed feature list is empty")
        channels = int(train_input[0].shape[1])
        classifier = AttentiveProbe(channels, heads, classes).to(train_input[0].device)
        make_batch = None
    else:
        raise ProtocolError(f"unknown readout: {readout}")
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    classifier.train()
    for _ in range(epochs):
        if readout == "mean":
            batches = [(list(range(start, min(start + batch_size, len(train_input)))), train_input[start : start + batch_size]) for start in range(0, len(train_input), batch_size)]
        else:
            batches = _batches(train_input, batch_size)
        for indices, batch_input in batches:
            index_tensor = torch.tensor(indices, device=train_labels.device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(classifier(batch_input), train_labels[index_tensor]).backward()
            optimizer.step()
    predictions: dict[str, list[int]] = {}
    classifier.eval()
    with torch.inference_mode():
        for condition, value in validation_input.items():
            if readout == "mean":
                prediction = torch.cat([classifier(value[start : start + batch_size]).argmax(-1) for start in range(0, len(value), batch_size)])
            else:
                prediction = torch.empty(len(value), dtype=torch.long, device=validation_labels[condition].device)
                for indices, batch_input in _batches(value, batch_size):
                    index_tensor = torch.tensor(indices, device=prediction.device)
                    prediction[index_tensor] = classifier(batch_input).argmax(-1)
            predictions[condition] = prediction.cpu().tolist()
    targets = {condition: validation_labels[condition].cpu().tolist() for condition in validation_labels}
    return cross_sampling_result(targets, predictions)
