"""Evaluation-only blur/noise × severity × budget experiment."""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from scripts.calibrate import collect_exit_outputs
from src.data.chestxray import ChestXrayDataset, LABELS
from src.evaluation.context import load_evaluation_context
from src.evaluation.degradation import DegradedDataset, KINDS, SEVERITIES, average_flops, confident_wrong_labels
from src.routing.budget import Budget, evaluate_budget_from_all
from src.routing.policy import uncertainty
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256, select_device


SUMMARY_FIELDS = (
    "degradation", "severity", "budget", "images", "mean_exit1_uncertainty",
    "mean_exit2_uncertainty", "mean_preferred_depth", "mean_actual_depth",
    "budget_forced_exit_rate", "exit1_count", "exit2_count", "final_count",
    "binary_bce", "binary_ece", "macro_auroc", "macro_f1", "subset_accuracy",
    "average_flops_per_image",
) + tuple(f"auroc_{label}" for label in LABELS) + ("mean_executed_flops",)
ERROR_FIELDS = ("degradation", "severity", "budget", "image", "patient_id", "finding_labels", "wrong_labels", "max_wrong_confidence") + tuple(f"probability_{label}" for label in LABELS)


def csv_text(rows: list[dict], fields: tuple[str, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def load_measured_flops(path: Path | None):
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("method"):
        raise ValueError("FLOPs file must name its measurement method")
    average_flops(torch.tensor([1, 2, 3]), value)
    return value


def run_conditions(model, policy, rows, config, device, kinds, seed, measured_flops, confidence_threshold):
    base = ChestXrayDataset(rows, config["dataset_root"], config["image_size"])
    conditions, summary_rows, errors = [], [], []
    for kind in kinds:
        for severity in SEVERITIES:
            dataset = DegradedDataset(base, kind, severity, config["degradation"], seed)
            loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                                num_workers=config["num_workers"], pin_memory=config["pin_memory"] and device.type == "cuda")
            exits, labels = collect_exit_outputs(model, loader, device)
            u1 = float(uncertainty(exits[0], policy.temperatures[0], policy.aggregation).mean())
            u2 = float(uncertainty(exits[1], policy.temperatures[1], policy.aggregation).mean())
            for budget in Budget:
                evaluated = evaluate_budget_from_all(exits, labels, policy, budget)
                metrics = evaluated["metrics"]
                actual = evaluated["actual_exit"]
                probabilities = torch.sigmoid(evaluated["scaled_logits"])
                high_error, wrong, wrong_confidence = confident_wrong_labels(probabilities, labels, confidence_threshold)
                if severity != "clean":
                    for index in torch.where(high_error)[0].tolist():
                        errors.append({
                            "degradation": kind, "severity": severity, "budget": budget.value,
                            "image": rows[index]["image"], "patient_id": rows[index]["patient_id"],
                            "finding_labels": rows[index]["finding_labels"],
                            "wrong_labels": "|".join(label for label, failed in zip(LABELS, wrong[index].tolist()) if failed),
                            "max_wrong_confidence": float(wrong_confidence[index]),
                            **{f"probability_{label}": float(probabilities[index, j]) for j, label in enumerate(LABELS)},
                        })
                mean_flops = average_flops(actual, measured_flops) if measured_flops is not None else None
                condition = {
                    "degradation": kind, "severity": severity, "budget": budget.value,
                    "images": len(labels), "mean_exit1_uncertainty": u1,
                    "mean_exit2_uncertainty": u2,
                    "mean_preferred_depth": float(evaluated["preferred_exit"].float().mean()),
                    "mean_actual_depth": float(actual.float().mean()),
                    "budget_forced_exit_rate": evaluated["budget_forced_exit_rate"],
                    "mean_executed_flops": mean_flops,
                    "average_flops_per_image": mean_flops,
                    "metrics": metrics,
                }
                conditions.append(condition)
                summary_rows.append({
                    **{key: condition[key] for key in ("degradation", "severity", "budget", "images", "mean_exit1_uncertainty", "mean_exit2_uncertainty", "mean_preferred_depth", "mean_actual_depth", "budget_forced_exit_rate", "mean_executed_flops", "average_flops_per_image")},
                    "exit1_count": metrics["exit_counts"][0], "exit2_count": metrics["exit_counts"][1], "final_count": metrics["exit_counts"][2],
                    **{key: metrics[key] for key in ("binary_bce", "binary_ece", "macro_auroc", "macro_f1", "subset_accuracy")},
                    **{f"auroc_{label}": metrics["per_label_auroc"][label] for label in LABELS},
                })
    return conditions, summary_rows, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="default: results/<checkpoint run>/calibration.json")
    parser.add_argument("--mode", choices=("calibrated", "uncalibrated"), default="calibrated")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--degradation", choices=(*KINDS, "both"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, help="noise seed; defaults to config seed")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--flops-file", type=Path, help="optional measured exit FLOPs JSON from Phase 8")
    parser.add_argument("--output", type=Path, help="default: results/<checkpoint run>/degradation_<split>_<mode>.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.5 < args.confidence_threshold <= 1:
        raise ValueError("confidence-threshold must lie in (0.5, 1]")
    config = load_config(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    run_name = checkpoint_path.parent.name
    calibration_path = resolve_project_path(args.calibration) if args.calibration else config["results_root"] / run_name / "calibration.json"
    output = resolve_project_path(args.output) if args.output else config["results_root"] / run_name / f"degradation_{args.split}_{args.mode}.json"
    summary_csv = output.with_suffix(".csv")
    errors_csv = output.with_name(output.stem + "_confident_errors.csv")
    if not args.overwrite and any(path.exists() for path in (output, summary_csv, errors_csv)):
        raise FileExistsError(f"Degradation output already exists: {output}")
    device = select_device(args.device or config["device"])
    model, policy, splits = load_evaluation_context(config, checkpoint_path, calibration_path, args.mode, device)
    measured_flops = load_measured_flops(resolve_project_path(args.flops_file) if args.flops_file else None)
    kinds = KINDS if args.degradation == "both" else (args.degradation,)
    conditions, rows, errors = run_conditions(model, policy, splits[args.split], config, device, kinds,
                                             config["seed"] if args.seed is None else args.seed,
                                             measured_flops, args.confidence_threshold)
    report = {
        "split": args.split, "mode": args.mode, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path), "calibration": str(calibration_path),
        "method": "offline_all_exits_diagnostic", "degradation_parameters": config["degradation"],
        "noise_seed": config["seed"] if args.seed is None else args.seed,
        "confidence_error_definition": "at least one incorrect binary label with decision confidence >= threshold",
        "confidence_threshold": args.confidence_threshold,
        "flops_source": str(args.flops_file) if args.flops_file else None,
        "measured_flops": measured_flops,
        "conditions": conditions,
    }
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False))
    atomic_write_text(summary_csv, csv_text(rows, SUMMARY_FIELDS))
    atomic_write_text(errors_csv, csv_text(errors, ERROR_FIELDS))
    print(json.dumps({"report": str(output), "summary_csv": str(summary_csv), "confident_errors_csv": str(errors_csv), "conditions": len(conditions), "confident_errors": len(errors)}))


if __name__ == "__main__":
    main()
