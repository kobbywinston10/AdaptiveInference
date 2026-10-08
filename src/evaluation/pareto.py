"""Validation threshold candidates and programmatic Pareto dominance."""

import math

from src.routing.policy import RoutingPolicy


def policy_candidates(artifact: dict, modes=("calibrated", "uncalibrated")) -> list[tuple[str, RoutingPolicy]]:
    candidates = []
    for mode in modes:
        entry = artifact["policies"][mode]
        base = RoutingPolicy.from_dict(entry["policy"])
        for index, point in enumerate(entry["threshold_sweep"]):
            candidates.append((f"{mode}_{index:03d}", RoutingPolicy(
                point["threshold1"], point["threshold2"], base.temperatures, base.aggregation,
            )))
    return candidates


def pareto_ids(points: list[dict], cost_key: str, score_key: str) -> set[str]:
    """Maximize score and minimize cost; equal points are both retained."""
    valid = [point for point in points if point.get(cost_key) is not None and point.get(score_key) is not None]
    for point in valid:
        if not math.isfinite(float(point[cost_key])) or not math.isfinite(float(point[score_key])):
            raise ValueError("Pareto coordinates must be finite")
    return {point["id"] for point in valid if not any(
        other["id"] != point["id"]
        and other[cost_key] <= point[cost_key]
        and other[score_key] >= point[score_key]
        and (other[cost_key] < point[cost_key] or other[score_key] > point[score_key])
        for other in valid
    )}
