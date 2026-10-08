"""Structural LOW/MEDIUM/HIGH caps over a validation-fitted routing policy."""

from enum import Enum

import torch

from src.routing.policy import RoutingPolicy, route_all, selected_metrics


class Budget(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


MAX_EXIT = {Budget.LOW: 1, Budget.MEDIUM: 2, Budget.HIGH: 3}


def _budget(value: Budget | str) -> Budget:
    try:
        return Budget(value)
    except ValueError as error:
        raise ValueError("budget must be LOW, MEDIUM, or HIGH") from error


def constrain_exit(preferred: torch.Tensor, budget: Budget | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return actual exit and whether the budget forced an earlier exit."""
    if preferred.ndim != 1 or not ((preferred >= 1) & (preferred <= 3)).all():
        raise ValueError("Preferred exits must be a vector of IDs 1, 2, or 3")
    maximum = MAX_EXIT[_budget(budget)]
    actual = preferred.clamp(max=maximum)
    return actual, preferred > maximum


def evaluate_budget_from_all(exits: tuple[torch.Tensor, ...], labels: torch.Tensor, policy: RoutingPolicy, budget: Budget | str) -> dict:
    """Exact diagnostic using all exits; this pass is not budgeted inference."""
    _, preferred = route_all(exits, policy)
    actual, forced = constrain_exit(preferred, budget)
    scaled = torch.stack([z.float() / t for z, t in zip(exits, policy.temperatures)], dim=1)
    chosen = scaled[torch.arange(len(actual), device=actual.device), actual - 1]
    metrics = selected_metrics(chosen, labels, actual)
    return {
        "budget": _budget(budget).value,
        "preferred_exit": preferred,
        "actual_exit": actual,
        "budget_forced": forced,
        "budget_forced_exit_rate": float(forced.float().mean()),
        "scaled_logits": chosen,
        "metrics": metrics,
    }


@torch.no_grad()
def route_one_budgeted(model, image: torch.Tensor, policy: RoutingPolicy, budget: Budget | str) -> dict:
    """Run only permitted layers; exact preferred exit is unknown if forced."""
    cap = MAX_EXIT[_budget(budget)]
    decision = {"forced": False}

    def should_exit(index, logits):
        preferred_here = policy.should_exit(index, logits)
        if index >= cap:
            decision["forced"] = not preferred_here
            return True
        return preferred_here

    model.eval()
    logits, actual = model.forward_adaptive(image, should_exit)
    probabilities = torch.sigmoid(logits.float() / policy.temperatures[actual - 1])
    return {
        "probabilities": probabilities,
        "actual_exit": actual,
        "preferred_exit": None if decision["forced"] else actual,
        "budget_forced": decision["forced"],
        "budget": _budget(budget).value,
    }
