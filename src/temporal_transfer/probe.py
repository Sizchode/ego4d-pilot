"""Published one-query attentive probe from the official V-JEPA 2 code.

The module structure and initialization are an attributed adaptation of
``facebookresearch/vjepa2/src/models/attentive_pooler.py`` and
``src/models/utils/modules.py`` (MIT license), pinned in the protocol.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.fc1(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return self.dropout(outputs)


class CrossAttention(nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        if dimension % heads:
            raise ValueError(f"input dimension {dimension} must be divisible by heads {heads}")
        self.heads = heads
        self.query = nn.Linear(dimension, dimension, bias=True)
        self.key_value = nn.Linear(dimension, 2 * dimension, bias=True)

    def forward(self, query: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        batch, queries, dimension = query.shape
        projected_query = self.query(query).reshape(batch, queries, self.heads, dimension // self.heads)
        projected_query = projected_query.permute(0, 2, 1, 3)
        _, tokens, _ = inputs.shape
        key_value = self.key_value(inputs).reshape(batch, tokens, 2, self.heads, dimension // self.heads)
        key_value = key_value.permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]
        with torch.backends.cuda.sdp_kernel():
            attended = functional.scaled_dot_product_attention(projected_query, key, value)
        return attended.transpose(1, 2).reshape(batch, queries, dimension)


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


class AttentiveClassifier(nn.Module):
    """Exact depth-one, one-query V-JEPA 2 attentive classifier."""

    def __init__(self, input_dim: int, heads: int, mlp_ratio: float, classes: int = 2):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.cross_attention_block = CrossAttentionBlock(input_dim, heads, mlp_ratio)
        self.linear = nn.Linear(input_dim, classes, bias=True)
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
            raise ValueError("probe input must have shape [batch, tokens, channels]")
        query = self.query_tokens.repeat(tokens.shape[0], 1, 1)
        pooled = self.cross_attention_block(query, tokens).squeeze(1)
        return self.linear(pooled)


def stack_token_sequences(sequences: list[torch.Tensor]) -> torch.Tensor:
    """Stack the fixed-length token sequences required by the published probe."""

    if not sequences or any(sequence.ndim != 2 for sequence in sequences):
        raise ValueError("expected a non-empty list of [tokens, channels] tensors")
    shapes = {tuple(sequence.shape) for sequence in sequences}
    if len(shapes) != 1:
        raise ValueError(f"published attentive probe requires equal token shapes per batch: {sorted(shapes)}")
    return torch.stack(sequences)


class MultiRepresentationProbe(nn.Module):
    """Train an independent official attentive probe for each representation."""

    def __init__(self, dimensions: dict[str, int], heads: int, mlp_ratio: float, classes: int):
        super().__init__()
        self.probes = nn.ModuleDict(
            {
                name: AttentiveClassifier(dimension, heads=heads, mlp_ratio=mlp_ratio, classes=classes)
                for name, dimension in dimensions.items()
            }
        )

    def forward(self, representations: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
        if set(representations) != set(self.probes):
            raise ValueError("representation names do not match initialized probes")
        return {
            name: self.probes[name](stack_token_sequences(sequences))
            for name, sequences in representations.items()
        }
