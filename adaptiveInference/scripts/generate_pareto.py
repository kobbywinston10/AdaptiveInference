"""Compute validation operating points and three measured-cost Pareto plots."""

import argparse
import csv
import io
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
from src.evaluation.degradation import average_flops
from src.evaluation.pareto import pareto_ids, policy_candidates
from src.routing.policy import route_all, selected_metrics
from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text, file_sha256, select_device


def plot_points(points, cost_key, score_key, path, xlabel, ylabel):
    frontier = pareto_ids(points, cost_key, score_key)
    figure, axis = plt.subplots(figsize=(7, 5))
    for kind, color, marker in (("static", "tab:blue", "s"), ("adaptive", "tab:orange", "o")):
        group = [p for p in points if p["kind"] == kind and p[score_key] is not None]
        axis.scatter([p[cost_key] for p in group], [p[score_key] for p in group],
                     c=color, marker=marker, alpha=0.65, label=kind)
    edge = sorted((p for p in points if p["id"] in frontier), key=lambda p: p[cost_key])
    if edge:
        axis.plot([p[cost_key] for p in edge], [p[score_key] for p in edge],
                  color="black", linewidth=1, label="Pareto frontier")
    for point in points:
        if point["kind"] == "static" and point[score_key] is not None:
            axis.annotate(point["id"], (point[cost_key], point[score_key]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(xlabel=xlabel, ylabel=ylabel, title="Validation operating points")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return sorted(frontier)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--flops", type=Path)
    parser.add_argument("--latency", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = resolve_project_path(args.checkpoint)
    run_dir = config["results_root"] / checkpoint.parent.name
    calibration = resolve_project_path(args.calibration) if args.calibration else run_dir / "calibration.json"
    flops_path = resolve_project_path(args.flops) if args.flops else run_dir / "flops.json"
    latency_path = resolve_project_path(args.latency) if args.latency else run_dir / "latency.json"
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in (
        "pareto_points.csv", "pareto_summary.json", "plots/subset_accuracy_vs_flops.png",
        "plots/macro_auroc_vs_flops.png", "plots/subset_accuracy_vs_latency.png",
    )]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("One or more Pareto outputs exist; use --overwrite")
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    device = select_device(args.device or config["device"])
    model, _, splits = load_evaluation_context(config, checkpoint, calibration, "calibrated", device)
    artifact = json.loads(calibration.read_text(encoding="utf-8"))
    flops = json.loads(flops_path.read_text(encoding="utf-8"))
    latency = json.loads(latency_path.read_text(encoding="utf-8"))
    digest = file_sha256(checkpoint)
    if flops.get("checkpoint_sha256") != digest or latency.get("checkpoint_sha256") != digest:
        raise ValueError("FLOPs/latency results do not match checkpoint")
    if latency.get("calibration_sha256") != file_sha256(calibration):
        raise ValueError("Latency results do not match calibration sweep")
    if flops.get("image_size") != config["image_size"] or latency.get("image_size") != config["image_size"]:
        raise ValueError("Benchmark image size differs from evaluation config")
    if latency.get("batch_size") != 1 or latency.get("split") != "validation":
        raise ValueError("Expected batch-one validation latency")
    measured = {point["id"]: point for point in latency["points"]}
    expected = ["fixed_exit1", "fixed_exit2", "full"] + [name for name, _ in policy_candidates(artifact)]
    if len(measured) != len(latency["points"]) or set(measured) != set(expected):
        raise ValueError("Latency results must contain each static and threshold-sweep point exactly once")
    loader = DataLoader(
        ChestXrayDataset(splits["validation"], config["dataset_root"], config["image_size"]),
        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"] and device.type == "cuda",
    )
    exits, labels = collect_exit_outputs(model, loader, device)
    points = []
    for index, name, cost_key in ((1, "fixed_exit1", "fixed_exit1"), (2, "fixed_exit2", "fixed_exit2"), (3, "full", "full")):
        ids = torch.full((len(labels),), index, dtype=torch.long)
        metrics = selected_metrics(exits[index - 1], labels, ids)
        path_cost = flops[cost_key]
        points.append({"id": name, "kind": "static", "threshold1": None, "threshold2": None,
                       "static_path_flops": path_cost, "mean_executed_flops": path_cost,
                       "average_flops": path_cost, "latency_mean_ms": measured[name]["mean_ms"], **metrics})
    for name, policy in policy_candidates(artifact):
        chosen, ids = route_all(exits, policy)
        metrics = selected_metrics(chosen, labels, ids)
        mean_cost = average_flops(ids, flops)
        points.append({"id": name, "kind": "adaptive", "threshold1": policy.threshold1,
                       "threshold2": policy.threshold2, "static_path_flops": None,
                       "mean_executed_flops": mean_cost, "average_flops": mean_cost,
                       "latency_mean_ms": measured[name]["mean_ms"], **metrics})
    frontiers = {
        "subset_accuracy_vs_flops": plot_points(points, "mean_executed_flops", "subset_accuracy", targets[2], "Mean executed FLOPs per image", "Subset accuracy"),
        "macro_auroc_vs_flops": plot_points(points, "mean_executed_flops", "macro_auroc", targets[3], "Mean executed FLOPs per image", "Macro AUROC"),
        "subset_accuracy_vs_latency": plot_points(points, "latency_mean_ms", "subset_accuracy", targets[4], "Mean latency (ms/image)", "Subset accuracy"),
    }
    fields = ("id", "kind", "threshold1", "threshold2", "average_flops", "latency_mean_ms",
              "subset_accuracy", "macro_auroc", "macro_f1", "binary_bce", "binary_ece",
              "average_exit_depth", "exit1_count", "exit2_count", "exit3_count",
              "static_path_flops", "mean_executed_flops")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for point in points:
        row = {key: point.get(key) for key in fields}
        row.update(zip(("exit1_count", "exit2_count", "exit3_count"), point["exit_counts"]))
        writer.writerow(row)
    atomic_write_text(targets[0], buffer.getvalue())
    summary = {"schema_version": 1, "split": "validation", "checkpoint_sha256": digest,
               "calibration_sha256": file_sha256(calibration), "flops_method": flops["method"],
               "flops_cost_basis": {"static": "selected fixed path", "adaptive": "mean of executed adaptive paths"},
               "latency_device": latency["device"], "latency_measurements": latency["measurements"],
               "points": len(points), "frontiers": frontiers}
    atomic_write_text(targets[1], json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({"output_dir": str(output_dir), "points": len(points), "frontiers": frontiers}, indent=2))


if __name__ == "__main__":
    main()
