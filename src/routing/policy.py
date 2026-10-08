"""Entropy-based early-exit routing, with validation-selected thresholds."""

import math
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from src.routing.calibration import _check_arrays, binary_ece
from src.data.chestxray import LABELS
from src.utils.run import file_sha256


def binary_entropy(probabilities: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    if epsilon <= 0 or epsilon >= 0.5:
        raise ValueError("epsilon must lie between zero and 0.5")
    p = probabilities.float().clamp(epsilon, 1 - epsilon)
    return -p * p.log() - (1 - p) * torch.log1p(-p)


def uncertainty(logits: torch.Tensor, temperature: float = 1.0, aggregation: str = "max") -> torch.Tensor:
    if logits.shape[-1] != 5 or temperature <= 0:
        raise ValueError("Expected five logits and a positive temperature")
    entropies = binary_entropy(torch.sigmoid(logits.float() / temperature))
    if aggregation == "max":
        return entropies.max(dim=-1).values
    if aggregation == "mean":
        return entropies.mean(dim=-1)
    raise ValueError("aggregation must be max or mean")


@dataclass(frozen=True)
class RoutingPolicy:
    threshold1: float
    threshold2: float
    temperatures: tuple[float, float, float] = (1.0, 1.0, 1.0)
    aggregation: str = "max"

    def __post_init__(self):
        if self.aggregation not in ("max", "mean") or len(self.temperatures) != 3:
            raise ValueError("Expected max/mean aggregation and three temperatures")
        if not all(math.isfinite(t) and t > 0 for t in self.temperatures):
            raise ValueError("Temperatures must be finite and positive")
        if not all(math.isfinite(t) and -1 <= t <= math.log(2) + 1e-6 for t in (self.threshold1, self.threshold2)):
            raise ValueError("Thresholds must lie in [-1, log(2)]")

    def should_exit(self, index: int, logits: torch.Tensor) -> bool:
        if index not in (1, 2) or logits.shape != (1, 5):
            raise ValueError("The conditional callback expects one image at Exit 1 or 2")
        threshold = self.threshold1 if index == 1 else self.threshold2
        return bool(uncertainty(logits, self.temperatures[index - 1], self.aggregation).item() <= threshold)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        return cls(value["threshold1"], value["threshold2"], tuple(value["temperatures"]), value["aggregation"])


def route_all(exits: tuple[torch.Tensor, ...], policy: RoutingPolicy) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline validation routing from all exits; returns scaled logits and exit IDs."""
    if len(exits) != 3 or any(z.ndim != 2 or z.shape[1] != 5 or z.shape != exits[0].shape for z in exits):
        raise ValueError("Expected three matching [images, 5] logit tensors")
    u1 = uncertainty(exits[0], policy.temperatures[0], policy.aggregation)
    u2 = uncertainty(exits[1], policy.temperatures[1], policy.aggregation)
    first = u1 <= policy.threshold1
    second = ~first & (u2 <= policy.threshold2)
    indices = torch.where(first, 1, torch.where(second, 2, 3))
    scaled = torch.stack([z.float() / t for z, t in zip(exits, policy.temperatures)], dim=1)
    chosen = scaled[torch.arange(len(indices), device=indices.device), indices - 1]
    return chosen, indices


@torch.no_grad()
def route_one(model, image: torch.Tensor, policy: RoutingPolicy) -> tuple[torch.Tensor, int]:
    """Real conditional inference: later ResNet layers are skipped."""
    model.eval()
    logits, index = model.forward_adaptive(image, policy.should_exit)
    return torch.sigmoid(logits.float() / policy.temperatures[index - 1]), index

def macro_auroc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro AUROC across labels with both positive and negative examples."""
    _check_arrays(logits, labels)

    from sklearn.metrics import roc_auc_score

    probabilities = torch.sigmoid(logits.float()).cpu().numpy()
    truth = labels.int().cpu().numpy()

    scores = []
    for i in range(len(LABELS)):
        if len(set(truth[:, i])) == 2:
            scores.append(float(roc_auc_score(truth[:, i], probabilities[:, i])))

    if not scores:
        raise ValueError("Macro AUROC is undefined for all labels")

    return sum(scores) / len(scores)


def threshold_sweep(exits: tuple[torch.Tensor, ...], labels: torch.Tensor, temperatures: tuple[float, float, float], aggregation: str = "max", quantile_steps: int = 11) -> tuple[list[dict], float]:
    """Sweep validation uncertainty quantiles; score Bernoulli BCE and depth."""
    if quantile_steps < 2:
        raise ValueError("quantile_steps must be at least 2")
    if len(exits) != 3 or len(temperatures) != 3:
        raise ValueError("Expected three exits and temperatures")
    for z in exits:
        _check_arrays(z, labels)
    if len(labels) == 0:
        raise ValueError("Validation set is empty")
    baseline_bce = float(F.binary_cross_entropy_with_logits(exits[2].float() / temperatures[2], labels.float()))
    grids = []
    for i in range(2):
        values = uncertainty(exits[i], temperatures[i], aggregation)
        quantiles = torch.quantile(values, torch.linspace(0, 1, quantile_steps, device=values.device))
        grids.append(sorted({-1.0, *[min(float(q), math.log(2)) for q in quantiles]}))
    sweep = []
    for t1 in grids[0]:
        for t2 in grids[1]:
            policy = RoutingPolicy(t1, t2, temperatures, aggregation)
            selected, indices = route_all(exits, policy)
            counts = [(indices == i).sum().item() for i in (1, 2, 3)]

            sweep.append({
                "threshold1": t1,
                "threshold2": t2,
                "validation_bce": float(
                    F.binary_cross_entropy_with_logits(selected, labels.float())
                ),
                "validation_macro_auroc": macro_auroc(selected, labels),
                "average_exit_depth": float(indices.float().mean()),
                "exit_counts": counts,
            })
    return sweep, baseline_bce


def select_thresholds(
    exits: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    temperatures: tuple[float, float, float],
    aggregation: str = "max",
    quantile_steps: int = 11,
    max_bce_increase: float = 0.02,
    max_auroc_drop: float = 0.01,
) -> tuple[RoutingPolicy, dict, list[dict]]:
    """Choose shallowest validation policy subject to BCE and AUROC constraints."""
    if max_bce_increase < 0:
        raise ValueError("max_bce_increase must be nonnegative")
    if max_auroc_drop < 0:
        raise ValueError("max_auroc_drop must be nonnegative")

    sweep, baseline_bce = threshold_sweep(
        exits,
        labels,
        temperatures,
        aggregation,
        quantile_steps,
    )

    final_logits = exits[2].float() / temperatures[2]
    baseline_auroc = macro_auroc(final_logits, labels)

    feasible = [
        point
        for point in sweep
        if (
            point["validation_bce"]
            <= baseline_bce + max_bce_increase + 1e-9
            and point["validation_macro_auroc"]
            >= baseline_auroc - max_auroc_drop - 1e-9
        )
    ]

    if not feasible:
        raise RuntimeError(
            "No routing policy satisfies the validation BCE and AUROC constraints"
        )

    chosen = min(
        feasible,
        key=lambda point: (
            point["average_exit_depth"],
            -point["validation_macro_auroc"],
            point["validation_bce"],
            point["threshold1"],
            point["threshold2"],
        ),
    )

    policy = RoutingPolicy(
        chosen["threshold1"],
        chosen["threshold2"],
        temperatures,
        aggregation,
    )

    selection = {
        **chosen,
        "final_exit_validation_bce": baseline_bce,
        "final_exit_validation_macro_auroc": baseline_auroc,
        "max_bce_increase": max_bce_increase,
        "max_auroc_drop": max_auroc_drop,
    }

    return policy, selection, sweep

def validation_summary(exits: tuple[torch.Tensor, ...], labels: torch.Tensor, policy: RoutingPolicy) -> dict:
    scaled, indices = route_all(exits, policy)
    return selected_metrics(scaled, labels, indices)


def selected_metrics(scaled: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> dict:
    _check_arrays(scaled, labels)
    if indices.shape != (len(labels),):
        raise ValueError("Exit IDs must match the validation batch")
    probabilities = torch.sigmoid(scaled)
    predictions = probabilities >= 0.5
    per_label_auroc = {}
    from sklearn.metrics import roc_auc_score, f1_score
    truth = labels.int().cpu().numpy()
    probs = probabilities.cpu().numpy()
    for i, label in enumerate(LABELS):
        per_label_auroc[label] = float(roc_auc_score(truth[:, i], probs[:, i])) if len(set(truth[:, i])) == 2 else None
    defined = [value for value in per_label_auroc.values() if value is not None]
    return {
        "binary_bce": float(F.binary_cross_entropy_with_logits(scaled, labels.float())),
        "binary_ece": binary_ece(probabilities, labels),
        "macro_auroc": sum(defined) / len(defined) if defined else None,
        "per_label_auroc": per_label_auroc,
        "macro_f1": float(f1_score(truth, predictions.int().cpu().numpy(), average="macro", zero_division=0)),
        "subset_accuracy": float((predictions == labels.bool()).all(dim=1).float().mean()),
        "exit_counts": [(indices == i).sum().item() for i in (1, 2, 3)],
        "average_exit_depth": float(indices.float().mean()),
    }


def load_policy(artifact_path: Path, mode: str, checkpoint_path: Path) -> RoutingPolicy:
    """Read a validation-fitted policy for its exact model checkpoint."""
    if mode not in ("calibrated", "uncalibrated"):
        raise ValueError("mode must be calibrated or uncalibrated")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1 or artifact.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise ValueError("Calibration artifact does not match the model checkpoint")
    return RoutingPolicy.from_dict(artifact["policies"][mode]["policy"])
