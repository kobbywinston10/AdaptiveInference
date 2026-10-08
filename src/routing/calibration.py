"""Validation-only scalar temperature fitting for independent labels."""

import math

import torch
from torch.nn import functional as F


def _check_arrays(logits: torch.Tensor, labels: torch.Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] != 5 or labels.shape != logits.shape:
        raise ValueError("Logits and labels must both have shape [images, 5]")
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
        raise ValueError("Logits and labels must be finite")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Labels must be binary")


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Minimize validation Bernoulli NLL with one bounded scalar temperature."""
    _check_arrays(logits, labels)
    if len(logits) == 0:
        raise ValueError("Cannot calibrate an empty validation set")
    z = logits.detach().to(device="cpu", dtype=torch.float64)
    y = labels.detach().to(device="cpu", dtype=torch.float64)
    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_t.clamp(math.log(0.05), math.log(20.0)).exp()
        loss = F.binary_cross_entropy_with_logits(z / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.detach().clamp(math.log(0.05), math.log(20.0)).exp())
    if not math.isfinite(temperature):
        raise ValueError("Temperature fitting produced a non-finite result")
    if F.binary_cross_entropy_with_logits(z / temperature, y) > F.binary_cross_entropy_with_logits(z, y):
        return 1.0
    return temperature


def fit_exit_temperatures(exits: tuple[torch.Tensor, ...], labels: torch.Tensor) -> tuple[float, float, float]:
    if len(exits) != 3:
        raise ValueError("Expected exactly three exits")
    return tuple(fit_temperature(logits, labels) for logits in exits)


def binary_ece(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    """Equal-width ECE over all image-label Bernoulli predictions."""
    _check_arrays(probabilities, labels)
    if bins < 1 or probabilities.numel() == 0:
        raise ValueError("bins and validation set must be nonempty")
    p = probabilities.detach().float().flatten()
    y = labels.detach().float().flatten()
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")
    boundaries = torch.linspace(0, 1, bins + 1, device=p.device)
    index = torch.bucketize(p, boundaries[1:-1])
    error = p.new_zeros(())
    for bucket in range(bins):
        mask = index == bucket
        if mask.any():
            error += mask.float().mean() * (p[mask].mean() - y[mask].mean()).abs()
    return float(error)
