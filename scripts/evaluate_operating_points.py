"""Evaluate frozen adaptive operating points and physical fixed paths."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from scripts.calibrate import collect_exit_outputs
from src.data.chestxray import ChestXrayDataset
from src.evaluation.context import load_evaluation_context
from src.evaluation.operating_points import mean_executed_flops
from src.routing.policy import RoutingPolicy, route_all, selected_metrics
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256, select_device, split_hashes


def evaluate_frozen(exits, labels, points, flops):
    """Preserve artifact order and policy values on either evaluation split."""
    evaluated = []
    for point in points:
        policy = RoutingPolicy.from_dict(point)
        chosen, ids = route_all(exits, policy)
        metrics = selected_metrics(chosen, labels, ids)
        counts = metrics["exit_counts"]
        cost = mean_executed_flops(counts, flops)
        evaluated.append({"id": point["id"], "label": point["label"],
                          "original_threshold_sweep_index": point["original_threshold_sweep_index"],
                          "threshold1": policy.threshold1, "threshold2": policy.threshold2,
                          "temperatures": list(policy.temperatures), "aggregation": policy.aggregation,
                          **metrics, "exit_fractions": [n / sum(counts) for n in counts],
                          "mean_executed_flops": cost,
                          "flop_saving_vs_static_full": 1 - cost / flops["full"],
                          "flop_saving_vs_adaptive_exit3": 1 - cost / flops["exit3"]})
    return evaluated


def plot_comparison(fixed, adaptive, path, split):
    figure, axis = plt.subplots(figsize=(7, 5))
    for row in fixed:
        axis.scatter(row["mean_executed_flops"], row["macro_auroc"], marker="s", color="tab:blue")
        axis.annotate(row["label"], (row["mean_executed_flops"], row["macro_auroc"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.plot([row["mean_executed_flops"] for row in adaptive], [row["macro_auroc"] for row in adaptive], "o-", color="tab:orange")
    for row in adaptive:
        axis.annotate(row["id"], (row["mean_executed_flops"], row["macro_auroc"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(xlabel="Mean executed FLOPs per image", ylabel="Macro AUROC", title=f"Frozen operating points ({split})")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--flops", type=Path)
    parser.add_argument("--operating-points", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = resolve_project_path(args.checkpoint)
    run_dir = config["results_root"] / checkpoint.parent.name
    calibration_path = resolve_project_path(args.calibration) if args.calibration else run_dir / "calibration.json"
    flops_path = resolve_project_path(args.flops) if args.flops else run_dir / "flops.json"
    points_path = resolve_project_path(args.operating_points) if args.operating_points else run_dir / "operating_points.json"
    output = resolve_project_path(args.output) if args.output else run_dir / f"operating_points_{args.split}.json"
    plot_path = output.with_suffix(".png")
    if not args.overwrite and (output.exists() or plot_path.exists()):
        raise FileExistsError(f"Operating-point evaluation output exists: {output} or {plot_path}")
    frozen = json.loads(points_path.read_text(encoding="utf-8"))
    flops = json.loads(flops_path.read_text(encoding="utf-8"))
    digest = file_sha256(checkpoint)
    if frozen.get("schema_version") != 1 or frozen.get("selection_split") != "validation":
        raise ValueError("Expected validation-selected operating points")
    if (frozen.get("checkpoint_sha256") != digest or flops.get("checkpoint_sha256") != digest
            or frozen.get("calibration_sha256") != file_sha256(calibration_path)
            or frozen.get("flops_sha256") != file_sha256(flops_path)
            or frozen.get("split_hashes") != split_hashes(config)
            or frozen.get("image_size") != config["image_size"]):
        raise ValueError("Frozen policies, measurements, model, or split manifests differ")
    if len(frozen["points"]) != frozen["requested_points"] or len({p["id"] for p in frozen["points"]}) != len(frozen["points"]):
        raise ValueError("Frozen operating point IDs are incomplete or duplicated")
    device = select_device(args.device or config["device"])
    model, _, splits = load_evaluation_context(config, checkpoint, calibration_path, "calibrated", device)
    loader = DataLoader(ChestXrayDataset(splits[args.split], config["dataset_root"], config["image_size"]),
                        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
                        pin_memory=config["pin_memory"] and device.type == "cuda")
    exits, labels = collect_exit_outputs(model, loader, device)
    fixed = []
    for index, name, key, label in ((1, "fixed_exit1", "fixed_exit1", "Fixed Exit1"),
                                    (2, "fixed_exit2", "fixed_exit2", "Fixed Exit2"),
                                    (3, "full", "full", "Full/Exit3")):
        ids = torch.full((len(labels),), index, dtype=torch.long)
        metrics = selected_metrics(exits[index - 1], labels, ids)
        cost = flops[key]
        fixed.append({"id": name, "label": label, **metrics,
                      "exit_fractions": [n / len(labels) for n in metrics["exit_counts"]],
                      "mean_executed_flops": cost, "static_path_flops": cost,
                      "flop_saving_vs_static_full": 1 - cost / flops["full"],
                      "flop_saving_vs_adaptive_exit3": 1 - cost / flops["exit3"]})
    adaptive = evaluate_frozen(exits, labels, frozen["points"], flops)
    report = {"schema_version": 1, "split": args.split, "selection_split": "validation",
              "checkpoint_sha256": digest, "calibration_sha256": file_sha256(calibration_path),
              "flops_sha256": file_sha256(flops_path), "operating_points_sha256": file_sha256(points_path),
              "method": "offline_all_exits_frozen_policy_evaluation",
              "fixed_points": fixed, "adaptive_points": adaptive}
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False))
    plot_comparison(fixed, adaptive, plot_path, args.split)
    print(json.dumps({"output": str(output), "plot": str(plot_path), "split": args.split,
                      "adaptive_points": len(adaptive)}, indent=2))


if __name__ == "__main__":
    main()
