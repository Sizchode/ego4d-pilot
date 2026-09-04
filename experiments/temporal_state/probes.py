"""Linear and V-JEPA-2-style attentive probes over saved representations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from .protocol import ProtocolError, nextqa_metrics, top1_accuracy


def _check_features(features: torch.Tensor, labels: torch.Tensor, classes: int) -> None:
    if features.ndim != 2 or len(features) != len(labels) or len(features) == 0:
        raise ProtocolError("linear features must be a non-empty [examples, channels] tensor")
    if classes <= 1 or any(int(value) < 0 or int(value) >= classes for value in labels.tolist()):
        raise ProtocolError("labels must be in the configured class range")


def _metric(targets: torch.Tensor, predictions: torch.Tensor, categories: Sequence[str] | None) -> dict[str, float]:
    target_values = targets.cpu().tolist()
    prediction_values = predictions.cpu().tolist()
    if categories is None:
        return {"top1": top1_accuracy(target_values, prediction_values)}
    return nextqa_metrics(target_values, prediction_values, categories)


def run_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    categories: Sequence[str] | None = None,
) -> dict[str, float]:
    """Train one independent linear classifier and report validation accuracy."""

    _check_features(train_features, train_labels, classes)
    _check_features(val_features, val_labels, classes)
    if train_features.shape[1] != val_features.shape[1] or epochs <= 0 or batch_size <= 0:
        raise ProtocolError("train/validation dimensions must match and epochs/batch_size must be positive")
    classifier = nn.Linear(int(train_features.shape[1]), classes).to(train_features.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    classifier.train()
    for _ in range(epochs):
        for start in range(0, len(train_features), batch_size):
            batch = slice(start, start + batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(classifier(train_features[batch]), train_labels[batch]).backward()
            optimizer.step()
    classifier.eval()
    with torch.inference_mode():
        predictions = torch.cat(
            [classifier(val_features[start : start + batch_size]).argmax(dim=-1) for start in range(0, len(val_features), batch_size)]
        )
    return _metric(val_labels, predictions, categories)


class MLP(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int):
        super().__init__()
        self.fc1 = nn.Linear(dimension, hidden_dimension)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dimension, dimension)
        self.dropout = nn.Dropout(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(self.activation(self.fc1(inputs)))))


class CrossAttention(nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        if dimension % heads:
            raise ProtocolError("attentive probe channels must be divisible by heads")
        self.heads = heads
        self.query = nn.Linear(dimension, dimension, bias=True)
        self.key_value = nn.Linear(dimension, 2 * dimension, bias=True)

    def forward(self, query: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        batch, query_count, dimension = query.shape
        projected_query = self.query(query).reshape(batch, query_count, self.heads, dimension // self.heads).transpose(1, 2)
        token_count = inputs.shape[1]
        key_value = self.key_value(inputs).reshape(batch, token_count, 2, self.heads, dimension // self.heads)
        key, value = key_value.unbind(dim=2)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attended = functional.scaled_dot_product_attention(projected_query, key, value)
        return attended.transpose(1, 2).reshape(batch, query_count, dimension)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.input_norm = nn.LayerNorm(dimension)
        self.cross_attention = CrossAttention(dimension, heads)
        self.query_norm = nn.LayerNorm(dimension)
        self.mlp = MLP(dimension, int(dimension * mlp_ratio))

    def forward(self, query: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        query = query + self.cross_attention(query, self.input_norm(inputs))
        return query + self.mlp(self.query_norm(query))


class AttentiveProbe(nn.Module):
    """The depth-one, one-query attentive classifier used by V-JEPA 2."""

    def __init__(self, input_dimension: int, heads: int, classes: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, input_dimension))
        self.cross_attention_block = CrossAttentionBlock(input_dimension, heads, mlp_ratio)
        self.linear = nn.Linear(input_dimension, classes, bias=True)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.apply(self._initialize)
        self.cross_attention_block.mlp.fc2.weight.data.div_(math.sqrt(2.0))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ProtocolError("attentive features must have shape [examples, tokens, channels]")
        query = self.query_tokens.expand(tokens.shape[0], -1, -1)
        pooled = self.cross_attention_block(query, tokens).squeeze(1)
        return self.linear(pooled)


def _token_batches(tokens: list[torch.Tensor], batch_size: int) -> list[tuple[list[int], torch.Tensor]]:
    """Group only by exact token shape; no padding or pooling is introduced."""

    if any(value.ndim != 2 for value in tokens):
        raise ProtocolError("each distributed sample must have shape [tokens, channels]")
    channels = {int(value.shape[1]) for value in tokens}
    if len(channels) != 1:
        raise ProtocolError("distributed representations must share the channel dimension")
    buckets: dict[int, list[int]] = {}
    for index, value in enumerate(tokens):
        buckets.setdefault(int(value.shape[0]), []).append(index)
    batches: list[tuple[list[int], torch.Tensor]] = []
    for indices in buckets.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batches.append((selected, torch.stack([tokens[index] for index in selected])))
    return batches


def run_attentive_probe(
    train_tokens: Sequence[torch.Tensor],
    train_labels: torch.Tensor,
    val_tokens: Sequence[torch.Tensor],
    val_labels: torch.Tensor,
    heads: int,
    classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    categories: Sequence[str] | None = None,
) -> dict[str, float]:
    """Train one official-style attentive probe without altering token sets."""

    train = list(train_tokens)
    validation = list(val_tokens)
    if not train or not validation or len(train) != len(train_labels) or len(validation) != len(val_labels):
        raise ProtocolError("attentive features and labels must be non-empty and equally sized")
    if epochs <= 0 or batch_size <= 0:
        raise ProtocolError("epochs and batch_size must be positive")
    train_batches = _token_batches(train, batch_size)
    validation_batches = _token_batches(validation, batch_size)
    probe = AttentiveProbe(int(train[0].shape[1]), heads=heads, classes=classes).to(train[0].device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for indices, token_batch in train_batches:
            index_tensor = torch.tensor(indices, device=train_labels.device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(probe(token_batch), train_labels[index_tensor]).backward()
            optimizer.step()
    predictions = torch.empty(len(validation), dtype=torch.long, device=val_labels.device)
    probe.eval()
    with torch.inference_mode():
        for indices, token_batch in validation_batches:
            index_tensor = torch.tensor(indices, device=val_labels.device)
            predictions[index_tensor] = probe(token_batch).argmax(dim=-1)
    return _metric(val_labels, predictions, categories)
