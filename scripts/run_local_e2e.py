"""Opt-in, one-epoch local pipeline check on a configured real dev subset.

Default invocation only checks prerequisites. --run executes short training and
validation stages; outputs are smoke-test artifacts, not experimental results.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.chestxray import LABELS, load_splits, training_class_stats
from src.utils.config import PROJECT_ROOT, load_config, resolve_project_path, validate_run_name
from src.utils.run import atomic_write_text, select_device


def command(script: str, *arguments: object, capture: bool = False) -> str | None:
    argv = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *(str(value) for value in arguments)]
    print("+ " + subprocess.list2cmdline(argv), flush=True)
    result = subprocess.run(argv, cwd=PROJECT_ROOT, check=True, text=True, capture_output=capture)
    return result.stdout if capture else None


def check_splits(config: dict) -> dict:
    splits = load_splits(config["split_dir"], config["dataset_root"])
    stats = training_class_stats(splits["train"])
    absent = [label for label in LABELS if stats[label]["positive"] == 0]
    if absent:
        raise ValueError(f"Training split has no positives for {absent}; increase max_images")
    validation = training_class_stats(splits["validation"])
    if not any(0 < validation[label]["positive"] < len(splits["validation"]) for label in LABELS):
        raise ValueError("Validation macro AUROC is undefined for every label; increase max_images")
    return {name: len(rows) for name, rows in splits.items()}


def main():
    parser = argparse.ArgumentParser(description="Check or run a short, real-image Phase 1-8 local pipeline")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--run-prefix", default="local_e2e")
    parser.add_argument("--scenario", choices=("with-kd", "no-kd", "both"), default="with-kd")
    parser.add_argument("--quantile-steps", type=int, default=3)
    parser.add_argument("--latency-samples", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measurements", type=int, default=3)
    parser.add_argument("--run", action="store_true", help="actually execute the short training and validation pipeline")
    args = parser.parse_args()
    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    validate_run_name(args.run_prefix)
    if args.quantile_steps < 2 or args.latency_samples < 1 or args.warmup < 0 or args.measurements < 1:
        raise ValueError("Invalid sweep or latency measurement settings")
    device = select_device(args.device)
    baseline_name = f"{args.run_prefix}_baseline"
    scenarios = ("no-kd", "with-kd") if args.scenario == "both" else (args.scenario,)
    names = {scenario: f"{args.run_prefix}_{scenario.replace('-', '_')}" for scenario in scenarios}
    required = (config["dataset_root"].is_dir(), config["metadata_csv"].is_file(), config["pretrained_checkpoint"].is_file())
    if not all(required):
        raise FileNotFoundError("Dev images, metadata CSV, and RadImageNet checkpoint must exist")
    manifests = [config["split_dir"] / f"{name}.csv" for name in ("train", "validation", "test")]
    if any(path.exists() for path in manifests) and not all(path.exists() for path in manifests):
        raise FileNotFoundError("Split manifests are incomplete; resolve before running")
    counts = check_splits(config) if all(path.exists() for path in manifests) else None
    output_names = (baseline_name, *names.values())
    occupied = [str(root / name) for root in (config["checkpoint_root"], config["results_root"])
                for name in output_names if (root / name).exists()]
    report = {
        "config": str(config_path), "device": str(device), "run_prefix": args.run_prefix,
        "dataset_root": str(config["dataset_root"]), "pretrained_checkpoint": str(config["pretrained_checkpoint"]),
        "split_counts": counts, "prepare_splits_on_run": counts is None,
        "baseline_checkpoint": str(config["checkpoint_root"] / baseline_name / "best.pt"),
        "adaptive_checkpoints": {scenario: str(config["checkpoint_root"] / name / "best.pt") for scenario, name in names.items()},
        "occupied_output_directories": occupied,
        "training_epochs_per_stage": 1,
        "test_split_used": False,
    }
    print(json.dumps(report, indent=2), flush=True)
    if not args.run:
        return
    if occupied:
        raise FileExistsError("Choose another --run-prefix; existing output directories will not be replaced")
    if counts is None:
        command("prepare_data.py", "--config", config_path)
        report["split_counts"] = check_splits(config)
    command("train_baseline.py", "--config", config_path, "--device", args.device, "--smoke")
    command("train_baseline.py", "--config", config_path, "--device", args.device,
            "--run-name", baseline_name, "--epochs", 1, "--num-workers", 0)
    baseline = config["checkpoint_root"] / baseline_name / "best.pt"
    static_output = command("evaluate_static.py", "--config", config_path, "--checkpoint", baseline,
                            "--split", "validation", "--device", args.device, capture=True)
    atomic_write_text(config["results_root"] / baseline_name / "static_validation.json",
                      json.dumps(json.loads(static_output), indent=2))
    for scenario, name in names.items():
        checkpoint = config["checkpoint_root"] / name / "best.pt"
        flops = config["results_root"] / name / "flops.json"
        command("train_adaptive.py", "--config", config_path, "--device", args.device,
                "--baseline-checkpoint", baseline, "--kd-weight", 1 if scenario == "with-kd" else 0,
                "--run-name", name, "--epochs", 1)
        command("calibrate.py", "--config", config_path, "--checkpoint", checkpoint,
                "--device", args.device, "--quantile-steps", args.quantile_steps)
        command("evaluate_adaptive.py", "--config", config_path, "--checkpoint", checkpoint,
                "--device", args.device, "--split", "validation")
        command("benchmark_flops.py", "--config", config_path, "--checkpoint", checkpoint, "--device", "cpu")
        command("run_degradation_experiment.py", "--config", config_path, "--checkpoint", checkpoint,
                "--device", args.device, "--split", "validation", "--degradation", "both", "--flops-file", flops)
        command("benchmark_latency.py", "--config", config_path, "--checkpoint", checkpoint,
                "--device", args.device, "--samples", args.latency_samples, "--warmup", args.warmup,
                "--measurements", args.measurements)
        command("generate_pareto.py", "--config", config_path, "--checkpoint", checkpoint, "--device", args.device)
    report["status"] = "completed_smoke_test"
    report["warning"] = "Short dev-subset runs and latency samples are pipeline checks, not final experimental results"
    summary_path = config["results_root"] / baseline_name / "local_e2e_summary.json"
    atomic_write_text(summary_path, json.dumps(report, indent=2))
    print(json.dumps({"summary": str(summary_path), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
