"""Optional PyAV video inspection and timestamp-based frame selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .protocol import ProtocolError
from .sampling import SamplingSchedule, map_to_actual_frames


def decode_video(video_path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Decode RGB frames and their presentation timestamps from the source video."""

    try:
        import av
    except ImportError as exc:
        raise ProtocolError("PyAV is required for timestamp-aware video audit/extraction") from exc
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base) if stream.time_base is not None else None
        for frame in container.decode(stream):
            timestamp = frame.time
            if timestamp is None and frame.pts is not None and time_base is not None:
                timestamp = float(frame.pts * time_base)
            if timestamp is None or not np.isfinite(timestamp):
                raise ProtocolError(f"video frame has no usable presentation timestamp: {video_path}")
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(float(timestamp))
    if not frames:
        raise ProtocolError(f"video contains no decodable frames: {video_path}")
    actual_timestamps = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(actual_timestamps) <= 0):
        raise ProtocolError(f"video presentation timestamps are not monotonic: {video_path}")
    return frames, actual_timestamps


def video_duration(video_path: str | Path) -> float:
    try:
        import av
    except ImportError as exc:
        raise ProtocolError("PyAV is required for timestamp-aware video audit/extraction") from exc
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.duration is None or stream.time_base is None:
            raise ProtocolError(f"video has no explicit duration: {video_path}")
        duration = float(stream.duration * stream.time_base)
    if duration <= 0:
        raise ProtocolError(f"video duration is not positive: {video_path}")
    return duration


def sample_video(video_path: str | Path, schedule: SamplingSchedule) -> tuple[list[np.ndarray], SamplingSchedule]:
    frames, timestamps = decode_video(video_path)
    actual_schedule = map_to_actual_frames(schedule, timestamps)
    if actual_schedule.frame_indices is None:
        raise ProtocolError("timestamp mapping did not produce frame indices")
    sampled = [frames[index] for index in actual_schedule.frame_indices]
    return sampled, actual_schedule
