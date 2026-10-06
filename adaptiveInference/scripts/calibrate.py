"""Fit temperatures and routing thresholds using validation data only."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.chestxray import ChestXrayDataset, load_splits
from src.models.adaptive import AdaptiveResNet50
from src.routing.calibration import binary_ece, fit_exit_temperatures
from src.routing.policy import select_thresholds, validation_summary
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256, select_device, split_hashes


def collect_exit_outputs(model, loader, device):
    model.eval()
    outputs = [[], [], []]
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model.forward_all_exits(images.to(device))
            for i, value in enumerate(logits):
                outputs[i].append(value.detach().float().cpu())
            targets.append(labels.cpu())
    if not targets:
        raise ValueError("Validation loader is empty")
    return tuple(torch.cat(parts) for parts in outputs), torch.cat(targets)


def build_artifact(exits, labels, checkpoint_path, split_digest, aggregation="max", quantile_steps=11, max_bce_increase=0.02):
    temperatures = fit_exit_temperatures(exits, labels)
    diagnostics = []
    for logits, temperature in zip(exits, temperatures):
        diagnostics.append({
            "temperature": temperature,
            "bce_before": float(F.binary_cross_entropy_with_logits(logits, labels)),
            "bce_after": float(F.binary_cross_entropy_with_logits(logits / temperature, labels)),
            "binary_ece_before": binary_ece(torch.sigmoid(logits), labels),
            "binary_ece_after": binary_ece(torch.sigmoid(logits / temperature), labels),
        })
    modes = {}
    for name, values in (("uncalibrated", (1.0, 1.0, 1.0)), ("calibrated", temperatures)):
        policy, selection, sweep = select_thresholds(exits, labels, values, aggregation, quantile_steps, max_bce_increase)
        modes[name] = {"policy": policy.to_dict(), "selection": selection, "validation_metrics": validation_summary(exits, labels, policy), "threshold_sweep": sweep}
    return {
        "schema_version": 1,
        "fit_split": "validation",
        "threshold_selection_split": "validation",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "split_hashes": split_digest,
        "aggregation": aggregation,
        "quantile_steps": quantile_steps,
        "max_bce_increase": max_bce_increase,
        "per_exit_calibration": diagnostics,
        "policies": modes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained Phase 4 best.pt")
    parser.add_argument("--output", type=Path, help="default: results/<checkpoint run>/calibration.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--aggregation", choices=("max", "mean"), default="max")
    parser.add_argument("--quantile-steps", type=int, default=11)
    parser.add_argument("--max-bce-increase", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.quantile_steps < 2 or args.max_bce_increase < 0:
        raise ValueError("quantile-steps must be >= 2 and max-bce-increase nonnegative")
    checkpoint_path = resolve_project_path(args.checkpoint)
    output = resolve_project_path(args.output) if args.output else config["results_root"] / checkpoint_path.parent.name / "calibration.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Calibration artifact exists: {output}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    digest = split_hashes(config)
    if checkpoint.get("split_hashes") != digest:
        raise ValueError("Phase 4 checkpoint and validation split manifests differ")
    splits = load_splits(config["split_dir"], config["dataset_root"])
    device = select_device(args.device or config["device"])
    model = AdaptiveResNet50()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    model.to(device)
    loader = DataLoader(
        ChestXrayDataset(splits["validation"], config["dataset_root"], config["image_size"]),
        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"] and device.type == "cuda",
    )
    exits, labels = collect_exit_outputs(model, loader, device)
    artifact = build_artifact(exits, labels, checkpoint_path, digest, args.aggregation, args.quantile_steps, args.max_bce_increase)
    atomic_write_text(output, json.dumps(artifact, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "temperatures": [row["temperature"] for row in artifact["per_exit_calibration"]], "comparison": {name: value["validation_metrics"] for name, value in artifact["policies"].items()}}, indent=2))


if __name__ == "__main__":
    main()
