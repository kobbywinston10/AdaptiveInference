"""Paired FP32/INT8 all-exit evaluation with frozen Python routing."""

import csv
import io

import torch

from src.deployment.onnx_utils import OUTPUT_NAMES, run_all_exits
from src.evaluation.degradation import average_flops
from src.routing.policy import RoutingPolicy, route_all, selected_metrics


def collect_paired_outputs(fp32_session, int8_session, dataset):
    """Run both models on each preprocessed image in manifest order."""
    outputs = {"fp32": [[], [], []], "int8": [[], [], []]}
    labels = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        batch = image.unsqueeze(0)
        for name, session in (("fp32", fp32_session), ("int8", int8_session)):
            for exit_index, logits in enumerate(run_all_exits(session, batch)):
                outputs[name][exit_index].append(logits)
        labels.append(target.unsqueeze(0))
    if not labels:
        raise ValueError("Evaluation split is empty")
    return {name: tuple(torch.cat(values) for values in parts) for name, parts in outputs.items()}, torch.cat(labels)


def score_variant(exits, labels, frozen_points: list[dict], flops: dict) -> tuple[dict, dict[str, torch.Tensor]]:
    """Score raw exits and each unchanged validation-frozen routing policy."""
    exit_metrics = {}
    for index, name in enumerate(OUTPUT_NAMES, 1):
        ids = torch.full((len(labels),), index, dtype=torch.long)
        exit_metrics[name] = selected_metrics(exits[index - 1], labels, ids)
    operating_points = []
    routed_ids = {}
    for point in frozen_points:
        policy = RoutingPolicy.from_dict(point)
        chosen, ids = route_all(exits, policy)
        metrics = selected_metrics(chosen, labels, ids)
        counts = metrics["exit_counts"]
        routed_ids[point["id"]] = ids
        operating_points.append({"id": point["id"], "label": point.get("label", point["id"]),
                                 "policy": policy.to_dict(), "metrics": metrics,
                                 "exit_fractions": [count / len(labels) for count in counts],
                                 "mean_executed_flops": average_flops(ids, flops)})
    return {"exit_metrics": exit_metrics, "operating_points": operating_points}, routed_ids


def compare_variants(fp32: dict, int8: dict, routed_fp32: dict, routed_int8: dict,
                     fp32_bytes: int, int8_bytes: int) -> dict:
    """Use INT8 minus FP32 for signed metric and routing deltas."""
    if fp32_bytes <= 0 or int8_bytes <= 0:
        raise ValueError("Model file sizes must be positive")
    left = {point["id"]: point for point in fp32["operating_points"]}
    right = {point["id"]: point for point in int8["operating_points"]}
    if list(left) != list(right) or set(routed_fp32) != set(left) or set(routed_int8) != set(left):
        raise ValueError("FP32 and INT8 policy families differ")

    def delta(a, b):
        return None if a is None or b is None else b - a

    points = []
    for point_id in left:
        fp, q = left[point_id], right[point_id]
        a, b = routed_fp32[point_id], routed_int8[point_id]
        if a.shape != b.shape:
            raise ValueError("Paired routing decisions have different lengths")
        transition = [[int(((a == i) & (b == j)).sum()) for j in (1, 2, 3)] for i in (1, 2, 3)]
        points.append({"id": point_id,
                       "fp32_macro_auroc": fp["metrics"]["macro_auroc"],
                       "int8_macro_auroc": q["metrics"]["macro_auroc"],
                       "macro_auroc_delta": delta(fp["metrics"]["macro_auroc"], q["metrics"]["macro_auroc"]),
                       "macro_f1_delta": delta(fp["metrics"]["macro_f1"], q["metrics"]["macro_f1"]),
                       "binary_bce_delta": delta(fp["metrics"]["binary_bce"], q["metrics"]["binary_bce"]),
                       "binary_ece_delta": delta(fp["metrics"]["binary_ece"], q["metrics"]["binary_ece"]),
                       "mean_executed_flops_delta": q["mean_executed_flops"] - fp["mean_executed_flops"],
                       "exit_fraction_delta": [y - x for x, y in zip(fp["exit_fractions"], q["exit_fractions"])],
                       "routing_switch_rate": float((a != b).float().mean()),
                       "routing_transition_counts_fp32_rows_int8_columns": transition})
    return {"delta_definition": "INT8 minus FP32",
            "size_reduction_percent": 100 * (1 - int8_bytes / fp32_bytes),
            "compression_ratio": fp32_bytes / int8_bytes,
            "fp32_exit3_macro_auroc": fp32["exit_metrics"]["exit3_logits"]["macro_auroc"],
            "int8_exit3_macro_auroc": int8["exit_metrics"]["exit3_logits"]["macro_auroc"],
            "exit3_macro_auroc_delta": delta(fp32["exit_metrics"]["exit3_logits"]["macro_auroc"],
                                             int8["exit_metrics"]["exit3_logits"]["macro_auroc"]),
            "operating_points": points}


def csv_text(fieldnames: tuple[str, ...], rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
