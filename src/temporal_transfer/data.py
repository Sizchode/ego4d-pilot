"""Runtime video dataset using the frame sampler fixed by the protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifests import VideoExample, read_jsonl
from .video import get_published_sampler, virtual_reversed_indices


class ManifestVideoDataset:
    """Decode only the frames selected by the published TAT sampler."""

    def __init__(self, manifest: str | Path, tat_root: str | Path, frames: int, sampling: str):
        if sampling not in {"rand", "middle"}:
            raise ValueError("sampling must be the published TAT 'rand' or 'middle' mode")
        self.examples = read_jsonl(manifest)
        self.frames = frames
        self.sampling = sampling
        self.sampler = get_published_sampler(tat_root)

    def _spatial_transform(self, video: Any) -> Any:
        import torch
        from torchvision.transforms import InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
        from torchvision.transforms.functional import resize

        frames = torch.from_numpy(video).permute(0, 3, 1, 2)
        if self.sampling == "rand":
            # Exact spatial augmentation disclosed by TAT's create_dataset.
            frames = RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC)(frames)
            frames = RandomHorizontalFlip()(frames)
        else:
            frames = resize(frames, [224, 224], interpolation=InterpolationMode.BICUBIC, antialias=True)
        return frames.permute(0, 2, 3, 1).numpy()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import decord
        except ImportError as error:
            raise RuntimeError("decord is required to decode probe videos") from error

        example: VideoExample = self.examples[index]
        reader = decord.VideoReader(example.video_path, num_threads=1)
        video_length = len(reader)
        sampled = self.sampler(self.frames, video_length, sample=self.sampling)
        decode_indices = virtual_reversed_indices(sampled, video_length, backward=bool(example.label))
        # Decord returns T,H,W,C uint8. Preserve temporal order, including reverse order.
        video = self._spatial_transform(reader.get_batch(decode_indices).asnumpy())
        return {
            "example_id": example.example_id,
            "video": video,
            "label": example.label,
            "motion_categories": example.motion_categories,
        }


def video_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep videos as a list because source spatial dimensions can differ."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is required to collate videos") from error
    return {
        "example_ids": [item["example_id"] for item in batch],
        "videos": [item["video"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "motion_categories": [item["motion_categories"] for item in batch],
    }
