"""DINOv3 image-foundation loader: one frozen image feature per sampled frame."""

from __future__ import annotations

from typing import Any

from .base import EncodedVideo, LoaderError, VideoLoader


class DINOv3Loader(VideoLoader):
    def __init__(self, config: dict[str, Any], device: Any):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise LoaderError("DINOv3 requires the HPC transformers environment") from exc
        revision = config.get("revision")
        repo_id = config.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id or not isinstance(revision, str) or not revision:
            raise LoaderError("DINOv3 repo_id and immutable revision must be recorded before execution")
        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(repo_id, revision=revision).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(repo_id, revision=revision)
        self.model.requires_grad_(False)
        self.repo_id = repo_id

    def encode_video(self, video_path: str, timestamps: list[float], model_family: str, *, prompt: str | None = None, return_layers: bool = False) -> EncodedVideo:
        if model_family != "image_foundation":
            raise LoaderError(f"DINOv3 loader received unexpected model family: {model_family!r}")
        if return_layers:
            raise LoaderError("DINOv3 pilot representation is the final frozen per-frame feature; layer extraction is not configured")
        from src.temporal_resampling.video import sample_video
        from src.temporal_resampling.sampling import SamplingSchedule

        frames, actual_schedule = sample_video(video_path, SamplingSchedule("C0", tuple(timestamps)))
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        if outputs.last_hidden_state is None or outputs.last_hidden_state.ndim != 3:
            raise LoaderError("DINOv3 did not expose per-frame token features")
        features = outputs.last_hidden_state[:, 0].detach().cpu()
        return EncodedVideo(
            "dinov3", "image_foundation", {"per_frame": features},
            {"video_path": video_path, "requested_timestamps": list(actual_schedule.timestamps), "actual_frame_timestamps": list(actual_schedule.actual_timestamps or ()), "sampled_frame_indices": list(actual_schedule.frame_indices or ())},
        )
