"""Qwen3-VL loader for frozen visual/decoder representation extraction."""

from __future__ import annotations

from typing import Any

from .base import EncodedVideo, LoaderError, VideoLoader


def _representation_layer(name: str) -> int | str | None:
    if name.startswith("decoder_layer_") or name.startswith("vision_layer_"):
        return int(name.split("_")[2])
    if name.endswith("_final_tokens") or name.endswith("_final"):
        return "final"
    return None


class Qwen3VLLoader(VideoLoader):
    def __init__(self, config: dict[str, Any], device: Any):
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise LoaderError("Qwen3-VL requires the pinned HPC transformers environment") from exc
        repo_id = config.get("repo_id")
        revision = config.get("revision")
        if not isinstance(repo_id, str) or not repo_id or not isinstance(revision, str) or not revision:
            raise LoaderError("Qwen3-VL repo_id and immutable revision must be recorded before execution")
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(repo_id, revision=revision)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            repo_id, revision=revision, torch_dtype=getattr(torch, config.get("dtype", "bfloat16"))
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.repo_id = repo_id
        base_model = getattr(self.model, "model", None)
        visual = getattr(base_model, "visual", None)
        blocks = getattr(visual, "blocks", None)
        merger = getattr(visual, "merger", None)
        if visual is None or blocks is None or len(blocks) == 0 or merger is None:
            raise LoaderError("Qwen3-VL model must expose model.visual.blocks and model.visual.merger for the planned visual readouts")
        self.visual = visual
        self.visual_blocks = tuple(blocks)
        self.visual_merger = merger
        decoder_layers = config.get("decoder_layers")
        language_model = getattr(base_model, "language_model", None)
        decoder_block_count = len(getattr(language_model, "layers", ()))
        if not isinstance(decoder_layers, list) or not decoder_layers or len(set(decoder_layers)) != len(decoder_layers):
            raise LoaderError("Qwen3-VL config must declare one or more selected decoder_layers")
        if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= decoder_block_count for index in decoder_layers):
            raise LoaderError("Qwen3-VL selected decoder_layers are outside the loaded decoder")
        self.decoder_layers = tuple(decoder_layers)

    def _visual_capture(self) -> tuple[dict[str, Any], list[Any]]:
        captures: dict[str, Any] = {}
        handles = []

        def register(name: str, module: Any) -> None:
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                if not isinstance(output, self.torch.Tensor) or output.ndim != 2:
                    raise LoaderError(f"Qwen3-VL {name} output must be a 2-D visual-token tensor")
                captures[name] = output.detach().cpu()

            handles.append(module.register_forward_hook(hook))

        final_index = len(self.visual_blocks) - 1
        register("vision_encoder_final_tokens", self.visual_blocks[final_index])
        register("visual_projector_tokens", self.visual_merger)
        return captures, handles

    def encode_video(self, video_path: str, timestamps: list[float], model_family: str, *, prompt: str | None = None, return_layers: bool = False) -> EncodedVideo:
        if model_family != "video_mllm":
            raise LoaderError(f"Qwen3-VL loader received unexpected model family: {model_family!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise LoaderError("Qwen3-VL representation extraction requires a registered fixed prompt")
        from src.temporal_resampling.sampling import SamplingSchedule
        from src.temporal_resampling.video import sample_video

        frames, actual_schedule = sample_video(video_path, SamplingSchedule("C0", tuple(timestamps)))
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[frames], return_tensors="pt").to(self.device)
        visual_captures, handles = self._visual_capture()
        try:
            with self.torch.inference_mode():
                outputs = self.model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if outputs.hidden_states is None:
            raise LoaderError("Qwen3-VL did not expose decoder hidden states")
        masks = inputs["input_ids"].eq(int(self.model.config.video_token_id))[0]
        if set(visual_captures) != {"vision_encoder_final_tokens", "visual_projector_tokens"}:
            raise LoaderError("Qwen3-VL did not execute all required visual representation hooks")
        representations = dict(visual_captures)
        representations["decoder_video_tokens_final"] = outputs.hidden_states[-1][0][masks].detach().cpu()
        if representations["visual_projector_tokens"].shape[0] != representations["decoder_video_tokens_final"].shape[0]:
            raise LoaderError("Qwen3-VL projector tokens and decoder video-token positions are not aligned")
        if return_layers:
            representations.update({f"decoder_layer_{index}_video_tokens": outputs.hidden_states[index + 1][0][masks].detach().cpu() for index in self.decoder_layers})
        return EncodedVideo(
            "qwen3_vl", "video_mllm", representations,
            {
                "video_path": video_path,
                "requested_timestamps": list(actual_schedule.timestamps),
                "actual_frame_timestamps": list(actual_schedule.actual_timestamps or ()),
                "sampled_frame_indices": list(actual_schedule.frame_indices or ()),
                "representation_points": {
                    name: {"representation_point": name, "layer": _representation_layer(name), "num_visual_tokens": int(value.shape[0])}
                    for name, value in representations.items()
                },
            },
        )

    def answer_multiple_choice(self, video_path: str, timestamps: list[float], prompt: str) -> int:
        """Generate one deterministic A--E/0--4 answer; reject free-form output."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise LoaderError("Qwen3-VL native evaluation requires a registered fixed prompt")

        from src.temporal_resampling.sampling import SamplingSchedule
        from src.temporal_resampling.video import sample_video

        frames, _ = sample_video(video_path, SamplingSchedule("C0", tuple(timestamps)))
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[frames], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=1, do_sample=False, num_beams=1)
        continuation = generated[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
        if len(decoded) == 1 and decoded in "ABCDE":
            return ord(decoded) - ord("A")
        if len(decoded) == 1 and decoded in "01234":
            return int(decoded)
        raise LoaderError(f"Qwen3-VL returned a non-deterministic/non-label answer: {decoded!r}")
