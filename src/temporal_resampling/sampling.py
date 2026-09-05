"""Deterministic C0--C3 schedules over real video time, not frame indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .protocol import ProtocolError


Condition = Literal["C0", "C1", "C2", "C3"]


@dataclass(frozen=True)
class SamplingSchedule:
    condition: Condition
    timestamps: tuple[float, ...]
    frame_indices: tuple[int, ...] | None = None
    actual_timestamps: tuple[float, ...] | None = None

    def validate(self, duration_seconds: float) -> None:
        if not np.isfinite(duration_seconds) or duration_seconds <= 0 or not self.timestamps:
            raise ProtocolError("video duration and schedule must be positive")
        if any(not np.isfinite(timestamp) for timestamp in self.timestamps):
            raise ProtocolError("sampling timestamps must be finite")
        if any(timestamp < 0 or timestamp >= duration_seconds for timestamp in self.timestamps):
            raise ProtocolError("sampling schedule contains an out-of-bounds timestamp")
        if any(right <= left for left, right in zip(self.timestamps, self.timestamps[1:])):
            raise ProtocolError("sampling timestamps must be strictly increasing")
        if self.frame_indices is not None and len(self.frame_indices) != len(self.timestamps):
            raise ProtocolError("frame indices must align one-to-one with real timestamps")
        if self.actual_timestamps is not None and len(self.actual_timestamps) != len(self.timestamps):
            raise ProtocolError("actual frame timestamps must align with requested timestamps")


def _strictly_increasing(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or values.size == 0 or np.any(np.diff(values) <= 0):
        raise ProtocolError("canonical timestamps must be a non-empty strictly increasing vector")
    return values


def _c3_schedule(canonical: np.ndarray) -> np.ndarray:
    if canonical.size < 2:
        raise ProtocolError("C3 requires at least two canonical timestamps")
    span = canonical[-1] - canonical[0]
    weights = np.resize(np.asarray([0.5, 1.5], dtype=np.float64), canonical.size - 1)
    steps = np.diff(canonical) * weights
    steps = span * steps / steps.sum()
    return np.concatenate(([canonical[0]], canonical[0] + np.cumsum(steps)))


def build_schedule(
    condition: Condition,
    canonical_timestamps: list[float] | tuple[float, ...] | np.ndarray,
    duration_seconds: float,
    *,
    offset_seconds: float | None = None,
    lower_rate_ratio: float | None = None,
) -> SamplingSchedule:
    """Build one schedule while preserving the canonical video time span."""

    canonical = _strictly_increasing(np.asarray(canonical_timestamps, dtype=np.float64))
    if canonical[0] < 0 or canonical[-1] >= duration_seconds:
        raise ProtocolError("canonical timestamps must lie within the video")
    if condition == "C0":
        timestamps = canonical
    elif condition == "C1":
        if lower_rate_ratio is None or not 0 < lower_rate_ratio < 1:
            raise ProtocolError("C1 lower_rate_ratio must be in (0, 1)")
        count = max(2, int(np.floor((canonical.size - 1) * lower_rate_ratio)) + 1)
        timestamps = np.linspace(canonical[0], canonical[-1], count)
    elif condition == "C2":
        if offset_seconds is None:
            raise ProtocolError("C2 requires an explicit offset_seconds")
        timestamps = canonical + float(offset_seconds)
        if timestamps[0] < 0 or timestamps[-1] >= duration_seconds:
            raise ProtocolError("C2 offset crosses the video boundary; choose an in-bounds offset")
    elif condition == "C3":
        timestamps = _c3_schedule(canonical)
    else:
        raise ProtocolError(f"unknown sampling condition: {condition}")
    schedule = SamplingSchedule(condition, tuple(float(value) for value in timestamps))
    schedule.validate(duration_seconds)
    return schedule


def map_to_actual_frames(schedule: SamplingSchedule, frame_timestamps: list[float] | np.ndarray) -> SamplingSchedule:
    """Map requested times to nearest decoded-frame PTS and retain both values upstream."""

    available = _strictly_increasing(np.asarray(frame_timestamps, dtype=np.float64))
    requested = np.asarray(schedule.timestamps, dtype=np.float64)
    if requested.size == 0 or np.any(~np.isfinite(requested)) or np.any(requested < available[0]) or np.any(requested > available[-1]):
        raise ProtocolError("requested timestamps must be inside the decoded video PTS range")
    indices = tuple(int(np.abs(available - timestamp).argmin()) for timestamp in schedule.timestamps)
    actual_timestamps = tuple(float(available[index]) for index in indices)
    return SamplingSchedule(schedule.condition, schedule.timestamps, indices, actual_timestamps)
