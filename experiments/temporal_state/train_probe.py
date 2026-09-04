"""Train one independent probe per layer from extracted feature files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .feature_store import load_feature_file
from .protocol import load_protocol_config, normalized_depth
from .probes import run_attentive_probe, run_linear_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--readout", choices=("compact", "distributed"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    _, config = load_protocol_config(args.config)
    train = load_feature_file(args.train_features)
    val = load_feature_file(args.val_features)
    device = torch.device(args.device)
    train["labels"] = train["labels"].to(device)
    val["labels"] = val["labels"].to(device)
    if (train["dataset"], train["model"]) != (val["dataset"], val["model"]):
        raise ValueError("train and validation feature files must use the same dataset and model")
    if train["total_layers"] != val["total_layers"]:
        raise ValueError("train and validation feature files must use the same total_layers")
    source = train[args.readout]
    validation = val[args.readout]
    if args.readout == "compact":
        source = {name: value.to(device) for name, value in source.items()}
        validation = {name: value.to(device) for name, value in validation.items()}
    else:
        source = {name: [value.to(device) for value in values] for name, values in source.items()}
        validation = {name: [value.to(device) for value in values] for name, values in validation.items()}
    layer_names = sorted(source)
    if layer_names != sorted(validation):
        raise ValueError("train and validation layers do not match")
    probe_config = config["probes"]
    optimization = probe_config["optimization"]
    result = {"dataset": train["dataset"], "model": train["model"], "readout": args.readout, "layers": {}}
    categories = val["categories"] if train["dataset"] == "nextqa" else None
    for layer_name in layer_names:
        match = re.fullmatch(r"layer_(\d+)", layer_name)
        if match is None:
            raise ValueError(f"feature representation is not a single raw layer: {layer_name}")
        layer_index = int(match.group(1))
        depth = normalized_depth(layer_index, int(train["total_layers"]))
        if args.readout == "compact":
            metrics = run_linear_probe(
                source[layer_name], train["labels"], validation[layer_name], val["labels"],
                classes=int(probe_config["classes"][train["dataset"]]), epochs=int(optimization["epochs"]),
                batch_size=int(optimization["batch_size"]), learning_rate=float(optimization["learning_rate"]),
                weight_decay=float(optimization["weight_decay"]),
                categories=categories,
            )
        else:
            metrics = run_attentive_probe(
                source[layer_name], train["labels"], validation[layer_name], val["labels"],
                heads=int(optimization["heads"]), classes=int(probe_config["classes"][train["dataset"]]),
                epochs=int(optimization["epochs"]), batch_size=int(optimization["batch_size"]),
                learning_rate=float(optimization["learning_rate"]),
                weight_decay=float(optimization["weight_decay"]), categories=categories,
            )
        layer_result = {
            "layer_index": layer_index,
            "total_layers": int(train["total_layers"]),
            "normalized_depth": depth,
            "readout_type": args.readout,
        }
        if train["dataset"] == "ssv2":
            layer_result["val_top1"] = metrics["top1"]
        else:
            layer_result.update(metrics)
        result["layers"][layer_name] = layer_result
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
