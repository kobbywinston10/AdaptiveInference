"""Evaluate an all-exit ONNX graph; optional Python routing is offline only."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.chestxray import ChestXrayDataset, LABELS, load_splits
from src.deployment.onnx_utils import OUTPUT_NAMES, compare_parity, load_session, model_footprint, run_all_exits
from src.evaluation.onnx_comparison import collect_paired_outputs, compare_variants, csv_text, score_variant
from src.models.adaptive import AdaptiveResNet50
from src.routing.policy import RoutingPolicy, load_policy, route_all, selected_metrics
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256, split_hashes


def collect_onnx_outputs(session, dataset):
    """Return all three raw logit arrays and labels in manifest order."""
    outputs = [[], [], []]
    labels = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        result = run_all_exits(session, image.unsqueeze(0))
        for exit_index, logits in enumerate(result):
            outputs[exit_index].append(logits)
        labels.append(target.unsqueeze(0))
    if not labels:
        raise ValueError("Evaluation split is empty")
    return tuple(torch.cat(values) for values in outputs), torch.cat(labels)


def run_comparison(args, config, checkpoint_path, run_dir):
    """Compare verified FP32 and INT8 models without changing frozen policies."""
    deployment_dir = config["results_root"] / checkpoint_path.parent.name / "deployment"
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else deployment_dir
    fp32_path = resolve_project_path(args.onnx) if args.onnx else deployment_dir / "adaptive_fp32.onnx"
    int8_path = resolve_project_path(args.int8) if args.int8 else deployment_dir / "adaptive_int8.onnx"
    export_path = resolve_project_path(args.export_report) if args.export_report else deployment_dir / "fp32_export_report.json"
    ptq_path = resolve_project_path(args.ptq_report) if args.ptq_report else deployment_dir / "ptq_report.json"
    calibration_path = resolve_project_path(args.calibration) if args.calibration else run_dir / "calibration.json"
    flops_path = resolve_project_path(args.flops) if args.flops else run_dir / "flops.json"
    points_path = resolve_project_path(args.operating_points) if args.operating_points else run_dir / "operating_points.json"
    split = args.split
    prefix = "quantization_test" if split == "test" else "quantization"
    targets = {
        "fp32": output_dir / f"fp32_{split}_metrics.json",
        "int8": output_dir / f"int8_{split}_metrics.json",
        "comparison": output_dir / f"{prefix}_comparison.json",
        "size_auroc": output_dir / f"model_size_vs_auroc_{split}.csv",
        "variant_size": output_dir / f"variant_vs_size_{split}.csv",
        "routing": output_dir / f"routing_distribution_{split}.csv",
    }
    if not args.overwrite and any(path.exists() for path in targets.values()):
        raise FileExistsError("One or more quantization comparison outputs already exist")
    digest = file_sha256(checkpoint_path)
    fp32_hash, int8_hash = file_sha256(fp32_path), file_sha256(int8_path)
    export = json.loads(export_path.read_text(encoding="utf-8"))
    ptq = json.loads(ptq_path.read_text(encoding="utf-8"))
    hashes = split_hashes(config)
    if (export.get("checkpoint_sha256") != digest or export.get("onnx_sha256") != fp32_hash
            or not export.get("validation_parity", {}).get("passed")
            or export.get("split_hashes") != hashes
            or export.get("input_shape") != [1, 3, config["image_size"], config["image_size"]]):
        raise ValueError("FP32 model must match a passed validation-parity export")
    if (ptq.get("checkpoint_sha256") != digest
            or ptq.get("source_fp32_onnx_sha256") != fp32_hash
            or ptq.get("int8_onnx_sha256") != int8_hash
            or ptq.get("fp32_export_report_sha256") != file_sha256(export_path)
            or ptq.get("calibration_split") != "train"):
        raise ValueError("INT8 model must match training-calibrated PTQ of this FP32 export")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_hash = file_sha256(calibration_path)
    flops = json.loads(flops_path.read_text(encoding="utf-8"))
    flops_hash = file_sha256(flops_path)
    if (calibration.get("checkpoint_sha256") != digest or calibration.get("split_hashes") != hashes
            or calibration.get("fit_split") != "validation"
            or calibration.get("threshold_selection_split") != "validation"
            or flops.get("checkpoint_sha256") != digest or flops.get("image_size") != config["image_size"]):
        raise ValueError("Calibration and FLOP artifacts must match this checkpoint and validation split")
    base_policy = RoutingPolicy.from_dict(calibration["policies"]["calibrated"]["policy"])
    if points_path.exists():
        frozen = json.loads(points_path.read_text(encoding="utf-8"))
        if (frozen.get("selection_split") != "validation"
                or frozen.get("checkpoint_sha256") != digest
                or frozen.get("calibration_sha256") != calibration_hash
                or frozen.get("flops_sha256") != flops_hash
                or frozen.get("split_hashes") != hashes
                or frozen.get("image_size") != config["image_size"]):
            raise ValueError("Operating points must be frozen from matching validation artifacts")
        points = frozen["points"]
        if len(points) != frozen.get("requested_points") or len({point["id"] for point in points}) != len(points):
            raise ValueError("Frozen operating point IDs are incomplete or duplicated")
        for point in points:
            policy = RoutingPolicy.from_dict(point)
            if policy.temperatures != base_policy.temperatures or policy.aggregation != base_policy.aggregation:
                raise ValueError("Operating points must use frozen FP32 calibration")
        policy_source = "operating_points.json"
        policy_hash = file_sha256(points_path)
    elif args.operating_points:
        raise FileNotFoundError(points_path)
    else:
        points = [{"id": "calibrated_frozen", "label": "calibrated_frozen", **base_policy.to_dict()}]
        policy_source = "calibration.json calibrated policy"
        policy_hash = calibration_hash
    if not points:
        raise ValueError("No frozen routing policies available")
    if split == "test":
        validation_path = output_dir / "quantization_comparison.json"
        if not validation_path.exists() or not args.validation_reviewed:
            raise ValueError("Test evaluation requires an existing validation comparison and --validation-reviewed")
        validation_summary = json.loads(validation_path.read_text(encoding="utf-8"))
        if (validation_summary.get("split") != "validation"
                or validation_summary.get("checkpoint_sha256") != digest
                or validation_summary.get("fp32_onnx_sha256") != fp32_hash
                or validation_summary.get("int8_onnx_sha256") != int8_hash
                or validation_summary.get("policy_artifact_sha256") != policy_hash
                or validation_summary.get("flops_sha256") != flops_hash):
            raise ValueError("Reviewed validation comparison does not match current frozen artifacts")
    with (config["split_dir"] / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dataset = ChestXrayDataset(rows, config["dataset_root"], config["image_size"])
    paired_exits, labels = collect_paired_outputs(load_session(fp32_path), load_session(int8_path), dataset)
    fp32_metrics, fp32_ids = score_variant(paired_exits["fp32"], labels, points, flops)
    int8_metrics, int8_ids = score_variant(paired_exits["int8"], labels, points, flops)
    footprints = {"fp32": model_footprint(fp32_path), "int8": model_footprint(int8_path)}
    fp32_bytes = footprints["fp32"]["deployment_size_bytes"]
    int8_bytes = footprints["int8"]["deployment_size_bytes"]
    comparison = compare_variants(fp32_metrics, int8_metrics, fp32_ids, int8_ids, fp32_bytes, int8_bytes)
    common = {"schema_version": 1, "split": split, "images": len(labels),
              "checkpoint_sha256": digest, "calibration_sha256": calibration_hash,
              "policy_artifact_sha256": policy_hash, "policy_source": policy_source,
              "selection_split": "validation", "flops_sha256": flops_hash,
              "flops_scope": "routed PyTorch conditional-path estimate; ONNX graph always computes all exits",
              "flops_method": flops["method"]}
    variant_reports = {}
    for name, path, size, model_hash, metrics in (
            ("fp32", fp32_path, fp32_bytes, fp32_hash, fp32_metrics),
            ("int8", int8_path, int8_bytes, int8_hash, int8_metrics)):
        variant_reports[name] = {**common, "variant": name, "onnx_path": str(path),
                                 "onnx_sha256": model_hash, "size_bytes": size,
                                 "size_mib": footprints[name]["deployment_size_mib"],
                                 **footprints[name],
                                 "compression_ratio_vs_fp32": fp32_bytes / size,
                                 **metrics}
    comparison_report = {**common, "fp32_onnx_sha256": fp32_hash, "int8_onnx_sha256": int8_hash,
                         "fp32_size_bytes": fp32_bytes, "int8_size_bytes": int8_bytes,
                         "fp32_footprint": footprints["fp32"], "int8_footprint": footprints["int8"],
                         "validation_comparison_sha256": file_sha256(output_dir / "quantization_comparison.json") if split == "test" else None,
                         **comparison}
    size_auroc_rows = []
    variant_size_rows = []
    routing_rows = []
    for name, report in variant_reports.items():
        variant_size_rows.append({"variant": name, "size_bytes": report["size_bytes"],
                                  "size_mib": report["size_mib"],
                                  "compression_ratio_vs_fp32": report["compression_ratio_vs_fp32"]})
        for exit_name in OUTPUT_NAMES:
            size_auroc_rows.append({"variant": name, "point_id": exit_name,
                                    "size_bytes": report["size_bytes"], "size_mib": report["size_mib"],
                                    "macro_auroc": report["exit_metrics"][exit_name]["macro_auroc"]})
        for point in report["operating_points"]:
            size_auroc_rows.append({"variant": name, "point_id": point["id"],
                                    "size_bytes": report["size_bytes"], "size_mib": report["size_mib"],
                                    "macro_auroc": point["metrics"]["macro_auroc"]})
            for exit_index, (count, fraction) in enumerate(zip(point["metrics"]["exit_counts"], point["exit_fractions"]), 1):
                routing_rows.append({"variant": name, "point_id": point["id"],
                                     "exit": exit_index, "count": count, "fraction": fraction})
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(targets["fp32"], json.dumps(variant_reports["fp32"], indent=2, allow_nan=False))
    atomic_write_text(targets["int8"], json.dumps(variant_reports["int8"], indent=2, allow_nan=False))
    atomic_write_text(targets["comparison"], json.dumps(comparison_report, indent=2, allow_nan=False))
    atomic_write_text(targets["size_auroc"], csv_text(("variant", "point_id", "size_bytes", "size_mib", "macro_auroc"), size_auroc_rows))
    atomic_write_text(targets["variant_size"], csv_text(("variant", "size_bytes", "size_mib", "compression_ratio_vs_fp32"), variant_size_rows))
    atomic_write_text(targets["routing"], csv_text(("variant", "point_id", "exit", "count", "fraction"), routing_rows))
    summary = {"split": split, "images": len(labels), "size_reduction_percent": comparison["size_reduction_percent"],
               "compression_ratio": comparison["compression_ratio"],
               "fp32_exit3_macro_auroc": comparison["fp32_exit3_macro_auroc"],
               "int8_exit3_macro_auroc": comparison["int8_exit3_macro_auroc"],
               "exit3_macro_auroc_delta": comparison["exit3_macro_auroc_delta"],
               "operating_points": [{key: point[key] for key in ("id", "fp32_macro_auroc", "int8_macro_auroc",
                                                             "macro_auroc_delta", "routing_switch_rate", "exit_fraction_delta")}
                                    for point in comparison["operating_points"]],
               "output": str(targets["comparison"])}
    print(json.dumps(summary, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--compare-int8", action="store_true", help="paired FP32/INT8 evaluation")
    parser.add_argument("--int8", type=Path)
    parser.add_argument("--export-report", type=Path)
    parser.add_argument("--ptq-report", type=Path)
    parser.add_argument("--calibration", type=Path, help="optional frozen calibrated routing policy")
    parser.add_argument("--operating-points", type=Path)
    parser.add_argument("--flops", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validation-reviewed", action="store_true", help="required for paired test evaluation after inspecting validation summary")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--parity-samples", type=int, default=0, help="compare against PyTorch on validation only")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    run_dir = config["results_root"] / checkpoint_path.parent.name
    if args.compare_int8:
        run_comparison(args, config, checkpoint_path, run_dir)
        return
    deployment_dir = run_dir / "deployment"
    onnx_path = resolve_project_path(args.onnx) if args.onnx else deployment_dir / "adaptive_fp32.onnx"
    export_report_path = resolve_project_path(args.export_report) if args.export_report else deployment_dir / "fp32_export_report.json"
    output = resolve_project_path(args.output) if args.output else deployment_dir / f"onnx_{args.split}_report.json"
    logits_path = output.with_name(output.stem.removesuffix("_report") + "_logits.npz")
    if not args.overwrite and (output.exists() or logits_path.exists()):
        raise FileExistsError(f"ONNX evaluation output exists: {output} or {logits_path}")
    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    digest = file_sha256(checkpoint_path)
    if (export_report.get("checkpoint_sha256") != digest
            or export_report.get("onnx_sha256") != file_sha256(onnx_path)
            or export_report.get("input_shape") != [1, 3, config["image_size"], config["image_size"]]
            or export_report.get("split_hashes") != split_hashes(config)):
        raise ValueError("ONNX export, checkpoint, image shape, or split manifests differ")
    if not export_report.get("validation_parity", {}).get("passed"):
        raise ValueError("ONNX export did not pass validation parity")
    splits = load_splits(config["split_dir"], config["dataset_root"])
    session = load_session(onnx_path)
    dataset = ChestXrayDataset(splits[args.split], config["dataset_root"], config["image_size"])
    exits, labels = collect_onnx_outputs(session, dataset)
    metrics = {}
    for index, name in enumerate(OUTPUT_NAMES, 1):
        ids = torch.full((len(labels),), index, dtype=torch.long)
        metrics[name] = selected_metrics(exits[index - 1], labels, ids)
    policy_result = None
    if args.calibration:
        calibration_path = resolve_project_path(args.calibration)
        artifact = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (artifact.get("fit_split") != "validation"
                or artifact.get("threshold_selection_split") != "validation"
                or artifact.get("split_hashes") != split_hashes(config)):
            raise ValueError("Routing policy must be frozen from this validation split")
        policy = load_policy(calibration_path, "calibrated", checkpoint_path)
        chosen, ids = route_all(exits, policy)
        policy_result = {"policy": policy.to_dict(), "calibration_sha256": file_sha256(calibration_path),
                         "metrics": selected_metrics(chosen, labels, ids)}
    parity = None
    if args.parity_samples:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("split_hashes") != split_hashes(config):
            raise ValueError("Checkpoint and split manifests differ")
        model = AdaptiveResNet50()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        validation = ChestXrayDataset(splits["validation"], config["dataset_root"], config["image_size"])
        parity = compare_parity(model, session, validation, args.parity_samples, args.atol, args.rtol)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(logits_path, **{name: value.numpy() for name, value in zip(OUTPUT_NAMES, exits)},
                        labels=labels.numpy(), label_names=np.asarray(LABELS))
    report = {"schema_version": 1, "split": args.split, "graph": "shared_weight_all_exits",
              "runtime_scope": "all three exits execute; offline routing metrics do not measure adaptive runtime",
              "checkpoint_sha256": digest, "onnx_sha256": file_sha256(onnx_path),
              "export_report_sha256": file_sha256(export_report_path),
              "label_order": list(LABELS), "images": len(labels),
              "logits_file": str(logits_path), "logits_sha256": file_sha256(logits_path),
              "exit_metrics": metrics, "frozen_calibrated_policy": policy_result,
              "validation_parity": parity}
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"report": str(output), "logits": str(logits_path),
                      "split": args.split, "validation_parity_passed": parity["passed"] if parity else None}, indent=2))
    if parity is not None and not parity["passed"]:
        raise RuntimeError("ONNX validation parity failed; inspect the evaluation report")


if __name__ == "__main__":
    main()
