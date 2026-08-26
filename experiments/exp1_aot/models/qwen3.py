"""Qwen3-VL representation extraction at the experiment-fixed architecture points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

def representation_names(config: dict[str, Any]) -> list[str]:
    requested = config["representations"]
    names = [f"vision_block_{index}" for index in requested["vision_blocks"]]
    if requested["include_visual_merger"]:
        names.append("visual_merger")
    names.extend(
        f"post_deepstack_language_layer_{index}" for index in requested["post_deepstack_language_layers"]
    )
    if requested["include_final_language_visual_tokens"]:
        names.append("final_language_visual_tokens")
    return names


class Qwen3VLProvider:
    def __init__(self, model_path: str | Path, config: dict[str, Any], device: Any):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dtype = getattr(torch, config["dtype"])
        self.config = config
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation=config["attention"],
        ).to(device)
        self.model.eval().requires_grad_(False)
        expected = config["expected_architecture"]
        visual = self.model.model.visual
        language = self.model.model.language_model
        actual = {
            "vision_depth": len(visual.blocks),
            "vision_hidden_size": int(visual.config.hidden_size),
            "vision_output_size": int(self.model.config.text_config.hidden_size),
            "deepstack_visual_indexes": list(visual.deepstack_visual_indexes),
            "language_depth": len(language.layers),
        }
        if actual != expected:
            raise RuntimeError(f"Qwen3-VL architecture mismatch: actual={actual}, expected={expected}")
        self.representation_dimensions = {}
        for name in representation_names(config):
            dimension = (
                expected["vision_hidden_size"]
                if name.startswith("vision_block_")
                else expected["vision_output_size"]
            )
            self.representation_dimensions[name] = dimension

    def _texts(self, batch_size: int) -> list[str]:
        messages = [
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": self.config["prompt"]}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return [text] * batch_size

    def extract(self, videos: list[Any], training: bool = False) -> dict[str, list[Any]]:
        import torch

        inputs = self.processor(
            text=self._texts(len(videos)), videos=videos, padding=True, return_tensors="pt"
        ).to(self.device)
        grid = inputs["video_grid_thw"]
        raw_lengths = grid.prod(-1).tolist()
        merge = self.model.model.visual.spatial_merge_size
        merged_lengths = (grid.prod(-1) // merge**2).tolist()
        video_token_id = int(self.model.config.video_token_id)
        video_masks = inputs["input_ids"].eq(video_token_id)
        captures: dict[str, torch.Tensor] = {}
        handles = []

        def save(name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                captures[name] = output[0] if isinstance(output, tuple) else output

            return hook

        requested = self.config["representations"]
        for index in requested["vision_blocks"]:
            handles.append(
                self.model.model.visual.blocks[index].register_forward_hook(save(f"vision_block_{index}"))
            )
        if requested["include_visual_merger"]:
            handles.append(self.model.model.visual.merger.register_forward_hook(save("visual_merger")))
        language_model = self.model.model.language_model
        original_deepstack_process = language_model._deepstack_process
        deepstack_stage = 0

        def capture_deepstack(
            hidden_states: torch.Tensor, visual_masks: torch.Tensor, visual_embeds: torch.Tensor
        ):
            nonlocal deepstack_stage
            output = original_deepstack_process(hidden_states, visual_masks, visual_embeds)
            if deepstack_stage in requested["post_deepstack_language_layers"]:
                captures[f"post_deepstack_language_layer_{deepstack_stage}"] = output
            deepstack_stage += 1
            return output

        language_model._deepstack_process = capture_deepstack
        if requested["include_final_language_visual_tokens"]:
            handles.append(
                self.model.model.language_model.norm.register_forward_hook(save("final_language_visual_tokens"))
            )

        try:
            with torch.inference_mode():
                self.model(**inputs, use_cache=False, return_dict=True)
        finally:
            language_model._deepstack_process = original_deepstack_process
            for handle in handles:
                handle.remove()

        expected_names = set(representation_names(self.config))
        if set(captures) != expected_names:
            raise RuntimeError(
                f"Qwen capture mismatch: got {sorted(captures)}, expected {sorted(expected_names)}"
            )
        representations: dict[str, list[torch.Tensor]] = {}
        for name, tensor in captures.items():
            if name.startswith("vision_block_"):
                representations[name] = list(torch.split(tensor, raw_lengths))
            elif name == "visual_merger":
                representations[name] = list(torch.split(tensor, merged_lengths))
            else:
                representations[name] = [tensor[index][video_masks[index]] for index in range(len(videos))]
        return representations
