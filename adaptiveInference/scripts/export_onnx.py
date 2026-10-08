"""Export one FP32 shared-weight three-output graph and validate on validation images."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.chestxray import ChestXrayDataset
from src.deployment.onnx_utils import INPUT_NAME, OUTPUT_NAMES, OPSET_VERSION, compare_parity, export_all_exits, load_session
from src.models.adaptive import AdaptiveResNet50
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256, split_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else config["results_root"] / checkpoint_path.parent.name / "deployment"
    onnx_path = output_dir / "adaptive_fp32.onnx"
    report_path = output_dir / "fp32_export_report.json"
    if not args.overwrite and (onnx_path.exists() or report_path.exists()):
        raise FileExistsError(f"Deployment output exists under {output_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hashes = split_hashes(config)
    if checkpoint.get("split_hashes") != hashes:
        raise ValueError("Checkpoint and split manifests differ")
    model = AdaptiveResNet50()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    with (config["split_dir"] / "validation.csv").open(newline="", encoding="utf-8") as handle:
        validation_rows = list(csv.DictReader(handle))
    dataset = ChestXrayDataset(validation_rows, config["dataset_root"], config["image_size"])
    if args.validation_samples < 1 or args.validation_samples > len(dataset):
        raise ValueError("validation-samples must be between one and the validation set size")
    export_all_exits(model, onnx_path, config["image_size"])
    session = load_session(onnx_path)
    parity = compare_parity(model, session, dataset, args.validation_samples, args.atol, args.rtol)
    bytes_on_disk = onnx_path.stat().st_size
    report = {"schema_version": 1, "format": "FP32 ONNX", "graph": "shared_weight_all_exits",
              "runtime_scope": "all three exits execute; no conditional adaptive latency or FLOP claim",
              "opset_version": OPSET_VERSION, "input_name": INPUT_NAME,
              "input_shape": [1, 3, config["image_size"], config["image_size"]],
              "output_names": list(OUTPUT_NAMES), "checkpoint_sha256": file_sha256(checkpoint_path),
              "split_hashes": hashes, "onnx_sha256": file_sha256(onnx_path),
              "onnx_size_bytes": bytes_on_disk, "onnx_size_mib": bytes_on_disk / (1024 ** 2),
              "onnx_checker_passed": True, "runtime_provider": session.get_providers(),
              "validation_parity": parity}
    atomic_write_text(report_path, json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"onnx": str(onnx_path), "report": str(report_path),
                      "validation_parity_passed": parity["passed"]}, indent=2))
    if not parity["passed"]:
        raise RuntimeError("ONNX validation parity failed; inspect the export report")


if __name__ == "__main__":
    main()
