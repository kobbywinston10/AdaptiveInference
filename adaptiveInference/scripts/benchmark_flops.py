"""Count fixed and adaptive path arithmetic with one documented method."""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.evaluation.flops import METHOD, count_exit_flops
from src.models.adaptive import AdaptiveResNet50
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256, select_device, split_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    output = resolve_project_path(args.output) if args.output else config["results_root"] / checkpoint_path.parent.name / "flops.json"
    csv_output = output.with_suffix(".csv")
    if not args.overwrite and (output.exists() or csv_output.exists()):
        raise FileExistsError(f"FLOPs output exists: {output} or {csv_output}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("split_hashes") != split_hashes(config):
        raise ValueError("Checkpoint and split manifests differ")
    model = AdaptiveResNet50()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    device = select_device(args.device)
    model.to(device)
    image = torch.zeros(1, 3, config["image_size"], config["image_size"], device=device)
    counts = count_exit_flops(model, image)
    path_flops = {
        "fixed_exit1_flops": counts["fixed_exit1"],
        "fixed_exit2_flops": counts["fixed_exit2"],
        "static_full_flops": counts["full"],
        "adaptive_exit1_flops": counts["exit1"],
        "adaptive_exit2_flops": counts["exit2"],
        "adaptive_exit3_flops": counts["exit3"],
    }
    report = {
        "method": METHOD,
        "convention": "2 FLOPs per multiply-add; counts Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, residual adds; excludes routing entropy and memory operations",
        "batch_size": 1,
        "image_size": config["image_size"],
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        "path_flops": path_flops,
        **counts,
    }
    atomic_write_text(output, json.dumps(report, indent=2))
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(("path", "flops", "method", "batch_size", "image_size"))
    for path, value in counts.items():
        writer.writerow((path, value, METHOD, 1, config["image_size"]))
    atomic_write_text(csv_output, buffer.getvalue())
    print(json.dumps({"output": str(output), **counts}, indent=2))


if __name__ == "__main__":
    main()
