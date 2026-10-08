"""Measure batch-one fixed and conditional inference on validation images."""

import argparse
import csv
import io
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.chestxray import ChestXrayDataset
from src.evaluation.context import load_evaluation_context
from src.evaluation.latency import measure_latency
from src.evaluation.pareto import policy_candidates
from src.routing.policy import route_one
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256, select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--samples", type=int, default=32, help="number of validation images held in memory")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measurements", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.measurements < 1:
        raise ValueError("samples and measurements must be positive; warmup must be nonnegative")
    config = load_config(args.config)
    checkpoint = resolve_project_path(args.checkpoint)
    run_dir = config["results_root"] / checkpoint.parent.name
    calibration = resolve_project_path(args.calibration) if args.calibration else run_dir / "calibration.json"
    output = resolve_project_path(args.output) if args.output else run_dir / "latency.json"
    csv_output = output.with_suffix(".csv")
    if not args.overwrite and (output.exists() or csv_output.exists()):
        raise FileExistsError(f"Latency output exists: {output} or {csv_output}")
    device = select_device(args.device or config["device"])
    model, _, splits = load_evaluation_context(config, checkpoint, calibration, "calibrated", device)
    artifact = json.loads(calibration.read_text(encoding="utf-8"))
    dataset = ChestXrayDataset(splits["validation"], config["dataset_root"], config["image_size"])
    images = [dataset[i][0].unsqueeze(0).to(device) for i in range(min(args.samples, len(dataset)))]
    if not images:
        raise ValueError("Validation split is empty")
    points = []
    for index, name in ((1, "fixed_exit1"), (2, "fixed_exit2"), (3, "full")):
        timing = measure_latency(
            lambda k, index=index: torch.sigmoid(model.forward_to_exit(images[k % len(images)], index)),
            device, args.warmup, args.measurements,
        )
        points.append({"id": name, "kind": "static", **timing})
    for name, policy in policy_candidates(artifact):
        timing = measure_latency(
            lambda k, policy=policy: route_one(model, images[k % len(images)], policy),
            device, args.warmup, args.measurements,
        )
        with torch.inference_mode():
            cached = model.forward_all_exits(images[0])
        overhead = measure_latency(
            lambda k, policy=policy: (
                policy.should_exit(1, cached[0]),
                policy.should_exit(2, cached[1]),
                torch.sigmoid(cached[2] / policy.temperatures[2]),
            ), device, args.warmup, args.measurements,
        )
        points.append({"id": name, "kind": "adaptive", **timing, "controller_overhead_ms": overhead})
    report = {
        "schema_version": 1,
        "split": "validation",
        "checkpoint_sha256": file_sha256(checkpoint),
        "calibration_sha256": file_sha256(calibration),
        "image_size": config["image_size"],
        "batch_size": 1,
        "samples": len(images),
        "warmup": args.warmup,
        "measurements": args.measurements,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "method": "preloaded_validation_images; inference_mode; warmed repeated wall-clock; CUDA synchronized before and after each measurement",
        "controller_overhead_method": "cached logits: both threshold decisions and final sigmoid, without model layers; diagnostic upper path estimate",
        "points": points,
    }
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=("id", "kind", "mean_ms", "p50_ms", "p95_ms", "controller_mean_ms", "controller_p50_ms", "controller_p95_ms"))
    writer.writeheader()
    for point in points:
        row = {key: point.get(key) for key in ("id", "kind", "mean_ms", "p50_ms", "p95_ms")}
        row.update({f"controller_{key}": point.get("controller_overhead_ms", {}).get(key) for key in ("mean_ms", "p50_ms", "p95_ms")})
        writer.writerow(row)
    atomic_write_text(csv_output, buffer.getvalue())
    print(json.dumps({"output": str(output), "points": len(points), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
