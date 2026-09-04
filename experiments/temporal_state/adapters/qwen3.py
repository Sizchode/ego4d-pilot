"""Qwen3-VL adapter for the two protocol-defined language-side readouts."""

from __future__ import annotations

from typing import Any

from .base import AdapterError, RepresentationAdapter


class Qwen3VLAdapter(RepresentationAdapter):
    def __init__(self, config: dict[str, Any], device: Any):
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise AdapterError("Qwen3-VL extraction requires the pinned transformers environment on HPC") from exc
        self.torch = torch
        self.device = device
        self.config = config
        revision = config.get("revision")
        if not isinstance(revision, str) or not revision:
            raise AdapterError("Qwen3-VL config must pin an immutable Hugging Face revision before HPC execution")
        self.processor = AutoProcessor.from_pretrained(config["repo_id"], revision=revision)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["repo_id"], revision=revision, torch_dtype=getattr(torch, config.get("dtype", "bfloat16"))
        ).to(device)
        self.model.eval().requires_grad_(False)
        self.layer_count = len(self.model.model.language_model.layers)
        if config.get("language_layers", "all") != "all":
            self.layers = [int(layer) for layer in config["language_layers"]]
        else:
            self.layers = list(range(self.layer_count))
        hidden_size = int(self.model.config.text_config.hidden_size)
        self._dimensions = {f"layer_{layer}": hidden_size for layer in self.layers}

    @property
    def representation_dimensions(self) -> dict[str, int]:
        return dict(self._dimensions)

    @property
    def total_layers(self) -> int:
        return self.layer_count

    def _inputs(self, video_paths: list[str], prompts: list[str]) -> Any:
        if len(video_paths) != len(prompts) or not video_paths:
            raise AdapterError("video_paths and prompts must be non-empty and equally sized")
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise AdapterError("qwen-vl-utils is required for official Qwen3-VL video preprocessing") from exc
        messages = [
            [{"role": "user", "content": [{"type": "video", "video": path}, {"type": "text", "text": prompt}]}]
            for path, prompt in zip(video_paths, prompts)
        ]
        texts = [self.processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for item in messages]
        videos = []
        for item in messages:
            _, video_inputs = process_vision_info(item)
            if not video_inputs:
                raise AdapterError("Qwen official processor returned no video input")
            videos.append(video_inputs[0])
        return self.processor(text=texts, videos=videos, padding=True, return_tensors="pt").to(self.device)

    def _forward(self, video_paths: list[str], prompts: list[str]) -> tuple[Any, Any, Any]:
        inputs = self._inputs(video_paths, prompts)
        with self.torch.inference_mode():
            outputs = self.model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) != self.layer_count + 1:
            raise AdapterError("Qwen3-VL did not expose one hidden-state tensor per decoder layer")
        video_token_id = int(self.model.config.video_token_id)
        masks = inputs["input_ids"].eq(video_token_id)
        boundaries = inputs["attention_mask"].sum(dim=-1) - 1
        return hidden_states, masks, boundaries

    def extract_compact_state(self, video_paths: list[str], prompts: list[str]) -> dict[str, Any]:
        hidden_states, _, boundaries = self._forward(video_paths, prompts)
        return {
            f"layer_{layer}": hidden_states[layer + 1][self.torch.arange(len(video_paths)), boundaries]
            for layer in self.layers
        }

    def extract_distributed_state(self, video_paths: list[str], prompts: list[str]) -> dict[str, list[Any]]:
        hidden_states, masks, _ = self._forward(video_paths, prompts)
        return {
            f"layer_{layer}": [hidden_states[layer + 1][index][masks[index]] for index in range(len(video_paths))]
            for layer in self.layers
        }
