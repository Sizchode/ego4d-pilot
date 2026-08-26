"""Train frozen-backbone Arrow-of-Time probes for Experiment 1."""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset import ManifestVideoDataset, video_collate
from models import build_model
from probe import MultiRepresentationProbe
from utils import arrow_of_time_metrics, load_config


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def _loader(
    dataset: ManifestVideoDataset,
    batch_size: int,
    rank: int,
    world_size: int,
    workers: int,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if world_size > 1:
        # PyTorch's released DistributedSampler is used for training. Evaluation
        # duplicates are removed by example_id before computing metrics.
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=video_collate,
    )
    return loader, sampler


def _gather_records(records: list[dict[str, Any]], world_size: int) -> list[dict[str, Any]]:
    if world_size == 1:
        return records
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, records)
    return [record for shard in gathered if shard is not None for record in shard]


def evaluate(
    provider: Any,
    probe: Any,
    loader: DataLoader,
    world_size: int,
) -> dict[str, dict[str, object]]:
    probe.eval()
    local: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            representations = provider.extract(batch["videos"], training=False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = probe(representations)
            for name, values in logits.items():
                predictions = values.argmax(-1).cpu().tolist()
                for example_id, target, prediction in zip(
                    batch["example_ids"], batch["labels"].tolist(), predictions
                ):
                    local.append(
                        {
                            "representation": name,
                            "example_id": example_id,
                            "target": target,
                            "prediction": prediction,
                        }
                    )
    all_records = _gather_records(local, world_size)
    by_representation: dict[str, dict[str, tuple[int, int]]] = {}
    for record in all_records:
        key = record["representation"]
        pair = (record["target"], record["prediction"])
        previous = by_representation.setdefault(key, {}).setdefault(record["example_id"], pair)
        if previous != pair:
            raise RuntimeError(f"distributed duplicate disagrees for {record['example_id']}")
    return {
        name: arrow_of_time_metrics(
            [pair[0] for pair in sorted(records.values())], [pair[1] for pair in sorted(records.values())]
        )
        for name, records in by_representation.items()
    }


def run_probe_training(
    config_path: str | Path,
    model_name: str,
    train_manifest: str | Path,
    eval_manifest: str | Path,
    output_dir: str | Path,
    workers: int,
) -> None:
    rank, world_size, local_rank = _distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("probe training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    resolved_config, config = load_config(config_path)
    _seed_everything(int(config["seed"]), rank)

    model_config = config["models"][model_name]
    provider = build_model(model_name, model_config, device)
    architecture = config["probe"]["architecture"]
    probes = MultiRepresentationProbe(
        provider.representation_dimensions,
        heads=int(architecture["heads"]),
        mlp_ratio=float(architecture["mlp_ratio"]),
        classes=int(architecture["classes"]),
    ).to(device)
    if world_size > 1:
        probes = DistributedDataParallel(probes, device_ids=[local_rank])

    frames = int(model_config.get("frames", config["video"]["frames"]))
    apply_tat_spatial = model_config.get("preprocessing") != "meta_official_frozen_probe"
    train_dataset = ManifestVideoDataset(
        train_manifest,
        frames,
        sampling="rand",
        apply_tat_spatial_transform=apply_tat_spatial,
    )
    eval_dataset = ManifestVideoDataset(
        eval_manifest,
        frames,
        sampling="middle",
        apply_tat_spatial_transform=apply_tat_spatial,
    )
    optimization = config["probe"]["optimization"]
    batch_size = int(optimization["batch_size_per_rank"])
    train_loader, train_sampler = _loader(train_dataset, batch_size, rank, world_size, workers, shuffle=True)
    eval_loader, _ = _loader(eval_dataset, batch_size, rank, world_size, workers, shuffle=False)

    optimizer = torch.optim.AdamW(
        probes.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    total_steps = int(optimization["epochs"]) * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=float(optimization["final_learning_rate"])
    )
    criterion = torch.nn.CrossEntropyLoss()
    join = probes.join if isinstance(probes, DistributedDataParallel) else nullcontext
    probes.train()
    with join():
        for epoch in range(int(optimization["epochs"])):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                labels = batch["labels"].to(device, non_blocking=True)
                frozen = provider.extract(batch["videos"], training=True)
                # Inference tensors cannot be retained by autograd for probe
                # weight gradients; clone the current batch into normal tensors.
                representations = {
                    name: [tokens.detach().clone() for tokens in sequences]
                    for name, sequences in frozen.items()
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = probes(representations)
                    loss = torch.stack([criterion(value, labels) for value in logits.values()]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

    metrics = evaluate(provider, probes, eval_loader, world_size)
    if rank == 0:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.yaml").write_text(
            resolved_config.read_text(encoding="utf-8"), encoding="utf-8"
        )
        result = {
            "model": model_name,
            "world_size": world_size,
            "frames": frames,
            "preprocessing": model_config.get("preprocessing", "tat_paper_spec"),
            "train_examples": len(train_dataset),
            "evaluation_examples": len(eval_dataset),
            "metrics": metrics,
        }
        (destination / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        state = (
            probes.module.state_dict() if isinstance(probes, DistributedDataParallel) else probes.state_dict()
        )
        torch.save(state, destination / "probe.pt")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--model", required=True, choices=("qwen3", "vjepa2_1", "videomaev2"))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_probe_training(
        args.config,
        args.model,
        args.train_manifest,
        args.eval_manifest,
        args.output_dir,
        args.workers,
    )


if __name__ == "__main__":
    main()
