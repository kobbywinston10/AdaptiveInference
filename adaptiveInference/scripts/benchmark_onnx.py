"""Supporting CPU latency comparison of FP32 and INT8 all-exit ONNX graphs."""

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort

from src.data.chestxray import ChestXrayDataset
from src.deployment.onnx_utils import INPUT_NAME, OUTPUT_NAMES, load_session
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256, split_hashes


def benchmark_pair(sessions: dict, images: list[np.ndarray], warmup: int, repetitions: int) -> dict:
    """Time identical preloaded inputs, alternating variant order per call."""
    if set(sessions) != {"fp32", "int8"} or not images or warmup < 1 or repetitions < 1:
        raise ValueError("Need both sessions, images, positive warmup, and positive repetitions")
    providers = {name: session.get_providers() for name, session in sessions.items()}
    if providers != {"fp32": ["CPUExecutionProvider"], "int8": ["CPUExecutionProvider"]}:
        raise ValueError("Both ONNX models must use CPUExecutionProvider only")
    for image in images:
        if image.dtype != np.float32 or image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("Expected preloaded batch-one float32 RGB inputs")

    def run_pair(index: int, timed: bool, times: dict):
        feed = {INPUT_NAME: images[index % len(images)]}
        names = ("fp32", "int8") if index % 2 == 0 else ("int8", "fp32")
        for name in names:
            start = perf_counter_ns() if timed else 0
            sessions[name].run(list(OUTPUT_NAMES), feed)
            if timed:
                times[name].append((perf_counter_ns() - start) / 1e6)

    times = {"fp32": [], "int8": []}
    for index in range(warmup):
        run_pair(index, False, times)
    for index in range(repetitions):
        run_pair(index, True, times)
    return {name: {"mean_ms": float(np.mean(values)),
                   "p50_ms": float(np.percentile(values, 50)),
                   "p95_ms": float(np.percentile(values, 95)),
                   "measured_calls": len(values)} for name, values in times.items()}


def cpu_model() -> str:
    name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.uname().processor
    if name:
        return name
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fp32", type=Path)
    parser.add_argument("--int8", type=Path)
    parser.add_argument("--export-report", type=Path)
    parser.add_argument("--ptq-report", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.warmup, args.repetitions, args.intra_op_threads) < 1:
        raise ValueError("samples, warmup, repetitions, and intra-op-threads must be positive")
    if args.repetitions < args.samples:
        raise ValueError("repetitions must cover every selected sample at least once")
    config = load_config(args.config)
    checkpoint = resolve_project_path(args.checkpoint)
    deployment_dir = config["results_root"] / checkpoint.parent.name / "deployment"
    fp32_path = resolve_project_path(args.fp32) if args.fp32 else deployment_dir / "adaptive_fp32.onnx"
    int8_path = resolve_project_path(args.int8) if args.int8 else deployment_dir / "adaptive_int8.onnx"
    export_path = resolve_project_path(args.export_report) if args.export_report else deployment_dir / "fp32_export_report.json"
    ptq_path = resolve_project_path(args.ptq_report) if args.ptq_report else deployment_dir / "ptq_report.json"
    output = resolve_project_path(args.output) if args.output else deployment_dir / "onnx_latency.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"ONNX latency output exists: {output}")
    checkpoint_hash = file_sha256(checkpoint)
    fp32_hash, int8_hash = file_sha256(fp32_path), file_sha256(int8_path)
    export = json.loads(export_path.read_text(encoding="utf-8"))
    ptq = json.loads(ptq_path.read_text(encoding="utf-8"))
    if (export.get("checkpoint_sha256") != checkpoint_hash
            or export.get("onnx_sha256") != fp32_hash
            or not export.get("validation_parity", {}).get("passed")
            or export.get("input_shape") != [1, 3, config["image_size"], config["image_size"]]
            or export.get("split_hashes") != split_hashes(config)
            or ptq.get("checkpoint_sha256") != checkpoint_hash
            or ptq.get("source_fp32_onnx_sha256") != fp32_hash
            or ptq.get("int8_onnx_sha256") != int8_hash
            or ptq.get("fp32_export_report_sha256") != file_sha256(export_path)
            or ptq.get("calibration_split") != "train"):
        raise ValueError("FP32 and INT8 files must match validated export and training-only PTQ reports")
    with (config["split_dir"] / "validation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.samples > len(rows):
        raise ValueError(f"Requested {args.samples} images, but validation split has {len(rows)}")
    dataset = ChestXrayDataset(rows, config["dataset_root"], config["image_size"])
    images = [np.ascontiguousarray(dataset[index][0].unsqueeze(0).numpy(), dtype=np.float32)
              for index in range(args.samples)]
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.intra_op_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sessions = {"fp32": load_session(fp32_path, options), "int8": load_session(int8_path, options)}
    timings = benchmark_pair(sessions, images, args.warmup, args.repetitions)
    report = {"schema_version": 1, "split": "validation", "scope": "supporting ONNX deployment comparison",
              "runtime_scope": "Both all-exit ONNX graphs compute all three exits; these timings are not conditional early-exit latency and are separate from PyTorch routing results",
              "checkpoint_sha256": checkpoint_hash, "fp32_onnx_sha256": fp32_hash,
              "int8_onnx_sha256": int8_hash, "fp32_export_report_sha256": file_sha256(export_path),
              "ptq_report_sha256": file_sha256(ptq_path),
              "execution_provider": "CPUExecutionProvider", "providers_by_model": {
                  name: session.get_providers() for name, session in sessions.items()},
              "cpu_model": cpu_model(), "gpu_model": None,
              "onnxruntime_version": ort.__version__, "os": platform.platform(),
              "batch_size": 1, "image_size": config["image_size"], "sample_count": len(images),
              "sample_indices_in_validation_manifest": list(range(len(images))),
              "warmup_count_per_model": args.warmup, "repetitions_per_model": args.repetitions,
              "intra_op_threads": args.intra_op_threads,
              "session_execution_mode": "ORT_SEQUENTIAL", "graph_optimization_level": "ORT_ENABLE_ALL",
              "timed_scope": "session.run for all three logits; image loading and preprocessing excluded",
              "order": "same cyclic input order for each model; FP32/INT8 call order alternates by iteration",
              "results": timings}
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "execution_provider": report["execution_provider"],
                      "results": timings}, indent=2))


if __name__ == "__main__":
    main()
