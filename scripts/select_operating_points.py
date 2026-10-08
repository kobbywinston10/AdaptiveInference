"""Freeze representative calibrated operating points using validation artifacts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.operating_points import select_operating_points
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--flops", type=Path)
    parser.add_argument("--points", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = resolve_project_path(args.checkpoint)
    run_dir = config["results_root"] / checkpoint.parent.name
    calibration_path = resolve_project_path(args.calibration) if args.calibration else run_dir / "calibration.json"
    flops_path = resolve_project_path(args.flops) if args.flops else run_dir / "flops.json"
    output = resolve_project_path(args.output) if args.output else run_dir / "operating_points.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Operating points already exist: {output}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    flops = json.loads(flops_path.read_text(encoding="utf-8"))
    digest = file_sha256(checkpoint)
    if calibration.get("checkpoint_sha256") != digest or flops.get("checkpoint_sha256") != digest:
        raise ValueError("Calibration/FLOP artifacts do not match the checkpoint")
    if flops.get("image_size") != config["image_size"]:
        raise ValueError("FLOP benchmark image size differs from config")
    chosen = select_operating_points(calibration, flops, args.points)
    report = {"schema_version": 1, "selection_split": "validation", "checkpoint_sha256": digest,
              "calibration_sha256": file_sha256(calibration_path), "flops_sha256": file_sha256(flops_path),
              "image_size": config["image_size"], "flops_method": flops["method"],
              "split_hashes": calibration["split_hashes"], **chosen}
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "points": len(report["points"]),
                      "frozen_policy": report["frozen_policy"]}, indent=2))


if __name__ == "__main__":
    main()
