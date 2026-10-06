"""Offline preferred/actual exit diagnostics for each resource budget."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from scripts.calibrate import collect_exit_outputs
from src.data.chestxray import ChestXrayDataset
from src.evaluation.context import load_evaluation_context
from src.routing.budget import Budget, evaluate_budget_from_all
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="default: results/<checkpoint run>/calibration.json")
    parser.add_argument("--mode", choices=("calibrated", "uncalibrated"), default="calibrated")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--budget", choices=tuple(b.value for b in Budget), help="default: all three budgets")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    run_name = checkpoint_path.parent.name
    calibration_path = resolve_project_path(args.calibration) if args.calibration else config["results_root"] / run_name / "calibration.json"
    output = resolve_project_path(args.output) if args.output else config["results_root"] / run_name / f"budget_{args.split}_{args.mode}.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Evaluation output exists: {output}")
    device = select_device(args.device or config["device"])
    model, policy, splits = load_evaluation_context(config, checkpoint_path, calibration_path, args.mode, device)
    loader = DataLoader(
        ChestXrayDataset(splits[args.split], config["dataset_root"], config["image_size"]),
        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"] and device.type == "cuda",
    )
    exits, labels = collect_exit_outputs(model, loader, device)
    budgets = [Budget(args.budget)] if args.budget else list(Budget)
    results = {}
    for budget in budgets:
        evaluated = evaluate_budget_from_all(exits, labels, policy, budget)
        results[budget.value] = {
            "metrics": evaluated["metrics"],
            "budget_forced_exit_rate": evaluated["budget_forced_exit_rate"],
            "records": [
                {"image": row["image"], "patient_id": row["patient_id"], "preferred_exit": int(preferred), "actual_exit": int(actual), "budget_forced": bool(forced)}
                for row, preferred, actual, forced in zip(splits[args.split], evaluated["preferred_exit"], evaluated["actual_exit"], evaluated["budget_forced"])
            ],
        }
    report = {"split": args.split, "mode": args.mode, "checkpoint": str(checkpoint_path), "calibration": str(calibration_path), "method": "offline_all_exits_diagnostic", "budgets": results}
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "summary": {name: {"budget_forced_exit_rate": result["budget_forced_exit_rate"], **result["metrics"]} for name, result in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
