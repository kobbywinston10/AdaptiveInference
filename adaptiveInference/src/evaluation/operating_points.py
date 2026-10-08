"""Freeze representative policies from a calibrated validation sweep."""

import math

from src.evaluation.pareto import pareto_ids
from src.routing.policy import RoutingPolicy


def mean_executed_flops(counts: list[int], flops: dict) -> float:
    """Weighted mean of the three measured adaptive execution paths."""
    if len(counts) != 3 or any(not isinstance(n, int) or n < 0 for n in counts) or sum(counts) == 0:
        raise ValueError("Expected three nonnegative exit counts with a positive total")
    costs = [float(flops[f"exit{i}"]) for i in (1, 2, 3)]
    if not all(math.isfinite(cost) and cost > 0 for cost in costs) or not costs[0] < costs[1] < costs[2]:
        raise ValueError("Adaptive path FLOPs must be finite, positive, and increasing")
    return sum(n * cost for n, cost in zip(counts, costs)) / sum(counts)


def select_operating_points(calibration: dict, flops: dict, requested: int = 5) -> dict:
    """Select from validation-calibrated Pareto policies; never inspect test data.

    A useful adaptive policy sends at least one percent of validation images
    outside its most common exit. Endpoints and evenly spaced internal cost
    targets cover the retained frontier. The frozen policy replaces the nearest
    internal selection if it is itself on that frontier.
    """
    if requested < 1:
        raise ValueError("requested must be positive")
    if calibration.get("fit_split") != "validation" or calibration.get("threshold_selection_split") != "validation":
        raise ValueError("Calibration and threshold selection must use validation only")
    if calibration.get("checkpoint_sha256") != flops.get("checkpoint_sha256"):
        raise ValueError("Calibration and FLOP artifacts refer to different checkpoints")
    entry = calibration["policies"]["calibrated"]
    frozen = RoutingPolicy.from_dict(entry["policy"])
    rows = []
    frozen_index = None
    for index, row in enumerate(entry["threshold_sweep"]):
        policy = RoutingPolicy(row["threshold1"], row["threshold2"], frozen.temperatures, frozen.aggregation)
        counts = row["exit_counts"]
        cost = mean_executed_flops(counts, flops)
        score = float(row["validation_macro_auroc"])
        bce = float(row["validation_bce"])
        if not math.isfinite(score) or not math.isfinite(bce):
            raise ValueError("Validation metrics must be finite")
        candidate = {"id": f"calibrated_{index:03d}", "original_threshold_sweep_index": index,
                     "threshold1": policy.threshold1, "threshold2": policy.threshold2,
                     "temperatures": list(policy.temperatures), "aggregation": policy.aggregation,
                     "validation_macro_auroc": score, "validation_bce": bce,
                     "average_exit_depth": float(row["average_exit_depth"]),
                     "exit_counts": counts, "exit_fractions": [n / sum(counts) for n in counts],
                     "mean_executed_flops": cost,
                     "compute_saving_vs_adaptive_exit3": 1 - cost / flops["exit3"]}
        rows.append(candidate)
        if policy == frozen:
            frozen_index = index
    if not rows:
        raise ValueError("Calibrated threshold sweep is empty")
    # Equal compute and AUROC coordinates represent the same Pareto point.
    def tie(row):
        return (-row["validation_macro_auroc"], row["validation_bce"],
                row["threshold1"], row["threshold2"], row["original_threshold_sweep_index"])

    unique = {}
    for row in sorted(rows, key=tie):
        unique.setdefault((row["mean_executed_flops"], row["validation_macro_auroc"]), row)
    candidates = list(unique.values())
    frontier_ids = pareto_ids(candidates, "mean_executed_flops", "validation_macro_auroc")
    frontier = sorted((row for row in candidates if row["id"] in frontier_ids),
                      key=lambda row: (row["mean_executed_flops"], tie(row)))
    useful = [row for row in frontier if max(row["exit_fractions"]) <= 0.99]
    if len(useful) < requested:
        raise ValueError(f"Only {len(useful)} distinct useful calibrated Pareto policies; requested {requested}")
    low, high = useful[0]["mean_executed_flops"], useful[-1]["mean_executed_flops"]
    targets = [low + (high - low) * i / (requested - 1) for i in range(requested)] if requested > 1 else [(low + high) / 2]
    selected = []
    for target in targets:
        selected.append(min((row for row in useful if row not in selected),
                            key=lambda row: (abs(row["mean_executed_flops"] - target), tie(row))))
    frozen_row = next((row for row in useful if row["original_threshold_sweep_index"] == frozen_index), None)
    if frozen_row is not None and frozen_row not in selected:
        replaceable = selected[1:-1] if requested > 2 else selected
        victim = min(replaceable, key=lambda row: (abs(row["mean_executed_flops"] - frozen_row["mean_executed_flops"]), tie(row)))
        selected[selected.index(victim)] = frozen_row
    selected.sort(key=lambda row: row["mean_executed_flops"])
    if any(a["mean_executed_flops"] >= b["mean_executed_flops"] for a, b in zip(selected, selected[1:])):
        raise ValueError("Selected policy costs must increase strictly")
    labels = ("low_compute", "moderate_compute", "balanced", "high_compute", "accuracy_focused")
    points = []
    for index, row in enumerate(selected, 1):
        label = labels[index - 1] if len(selected) == 5 else (
            "low_compute" if index == 1 else "high_compute" if index == len(selected) else f"intermediate_compute_{index}")
        points.append({"id": f"adaptive_op{index}", "label": label, **{key: value for key, value in row.items() if key != "id"}})
    nearest = min(points, key=lambda row: (abs(row["mean_executed_flops"] - rows[frozen_index]["mean_executed_flops"]), row["id"])) if frozen_index is not None else None
    return {"requested_points": requested,
            "selection_algorithm": "Validation calibrated macro-AUROC/FLOP Pareto frontier; collapse equal coordinates; require >=1% of images outside the majority exit; nearest unique policies to evenly spaced cost targets; include frozen policy when eligible; ties: higher AUROC, lower BCE, lower thresholds, lower sweep index",
            "candidate_count": len(rows), "pareto_count": len(frontier), "useful_pareto_count": len(useful),
            "frozen_policy": {"original_threshold_sweep_index": frozen_index,
                              "included": frozen_row is not None and frozen_row in selected,
                              "nearest_selected_id": nearest["id"] if nearest else None},
            "points": points}
