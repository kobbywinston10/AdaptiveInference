"""Training-calibrated static INT8 QDQ conversion of the shared all-exit graph."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

from src.deployment.onnx_utils import OUTPUT_NAMES, load_session, model_footprint, run_all_exits
from src.deployment.quantization import TARGET_OPS, graph_summary, training_reader
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fp32", type=Path)
    parser.add_argument("--export-report", type=Path)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, help="defaults to project config seed")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = resolve_project_path(args.checkpoint)
    deployment_dir = config["results_root"] / checkpoint.parent.name / "deployment"
    source = resolve_project_path(args.fp32) if args.fp32 else deployment_dir / "adaptive_fp32.onnx"
    source_report_path = resolve_project_path(args.export_report) if args.export_report else deployment_dir / "fp32_export_report.json"
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else deployment_dir
    target = output_dir / "adaptive_int8.onnx"
    report_path = output_dir / "ptq_report.json"
    if source.resolve() == target.resolve():
        raise ValueError("INT8 output must differ from the FP32 source model")
    if not args.overwrite and (target.exists() or report_path.exists()):
        raise FileExistsError(f"PTQ output already exists under {output_dir}")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if (source_report.get("schema_version") != 1
            or source_report.get("graph") != "shared_weight_all_exits"
            or source_report.get("format") != "FP32 ONNX"
            or source_report.get("onnx_checker_passed") is not True
            or not source_report.get("validation_parity", {}).get("passed")
            or source_report.get("checkpoint_sha256") != file_sha256(checkpoint)
            or source_report.get("onnx_sha256") != file_sha256(source)
            or source_report.get("input_shape") != [1, 3, config["image_size"], config["image_size"]]
            or source_report.get("split_hashes", {}).get("train") != file_sha256(config["split_dir"] / "train.csv")):
        raise ValueError("FP32 export must pass validation parity and match checkpoint, model, and training split")
    load_session(source)
    source_graph = graph_summary(source)
    if source_graph["quantized_target_operators"]["Conv"] < 1:
        raise ValueError("Expected a convolution-heavy ONNX model for static PTQ")
    seed = config["seed"] if args.seed is None else args.seed
    reader, indices = training_reader(config, args.calibration_samples, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "adaptive_int8.tmp.onnx"
    try:
        quantize_static(str(source), str(temporary), reader,
                        quant_format=QuantFormat.QDQ,
                        op_types_to_quantize=list(TARGET_OPS),
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax,
                        per_channel=False)
        onnx.checker.check_model(str(temporary))
        session = load_session(temporary)
        image, _ = reader.dataset[indices[0]]
        outputs = run_all_exits(session, image.unsqueeze(0))
        if len(outputs) != len(OUTPUT_NAMES) or any(not np.isfinite(output.numpy()).all() for output in outputs):
            raise ValueError("INT8 graph did not produce finite logits at all three exits")
        quantized_graph = graph_summary(temporary)
        if quantized_graph["int8_initializers"] < 1 or quantized_graph["operator_counts"].get("QuantizeLinear", 0) < 1:
            raise ValueError("Quantizer did not create INT8 QDQ parameters")
        if not all(quantized_graph["qdq_wrapped_target_operators"][name] == source_graph["quantized_target_operators"][name]
                   for name in TARGET_OPS):
            raise ValueError("Some requested Conv/Gemm operators were not QDQ-wrapped")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    fp32_footprint = model_footprint(source)
    int8_footprint = model_footprint(target)
    fp32_bytes = fp32_footprint["deployment_size_bytes"]
    int8_bytes = int8_footprint["deployment_size_bytes"]
    report = {
        "schema_version": 1,
        "source_fp32_onnx_path": str(source), "source_fp32_onnx_sha256": file_sha256(source),
        "int8_onnx_path": str(target), "int8_onnx_sha256": file_sha256(target),
        "fp32_export_report_sha256": file_sha256(source_report_path),
        "checkpoint_sha256": file_sha256(checkpoint),
        "quantization_method": "ONNX Runtime static post-training quantization; MinMax calibration",
        "quantization_format": "QDQ", "activation_type": "QInt8", "weight_type": "QInt8",
        "per_channel": False, "op_types_to_quantize": list(TARGET_OPS),
        "calibration_split": "train", "calibration_sample_count": len(indices),
        "calibration_seed": seed,
        "selection_method": "seeded sampling without replacement; selected indices sorted in training manifest order",
        "selected_training_indices": indices,
        "fp32_size_bytes": fp32_bytes, "fp32_size_mib": fp32_footprint["deployment_size_mib"],
        "int8_size_bytes": int8_bytes, "int8_size_mib": int8_footprint["deployment_size_mib"],
        "fp32_footprint": fp32_footprint, "int8_footprint": int8_footprint,
        "compression_ratio_fp32_over_int8": fp32_bytes / int8_bytes,
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__},
        "source_graph": source_graph, "int8_graph": quantized_graph,
        "runtime_provider": session.get_providers(),
        "smoke_output_shapes": {name: list(output.shape) for name, output in zip(OUTPUT_NAMES, outputs)},
        "runtime_scope": "all three exits execute; INT8 changes precision and footprint, not FLOP count or conditional execution",
    }
    atomic_write_text(report_path, json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"int8_onnx": str(target), "report": str(report_path),
                      "compression_ratio": report["compression_ratio_fp32_over_int8"]}, indent=2))


if __name__ == "__main__":
    main()
