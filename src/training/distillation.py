"""Auxiliary-head training with a detached, full-depth Bernoulli teacher."""

from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from src.models.adaptive import AdaptiveResNet50


@dataclass(frozen=True)
class DistillationConfig:
    temperature: float = 2.0
    hard_weight: float = 1.0
    kd_weight: float = 1.0
    exit_weights: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self):
        if self.temperature <= 0 or min(self.hard_weight, self.kd_weight, *self.exit_weights) < 0:
            raise ValueError("Temperature must be positive and loss weights nonnegative")


def bernoulli_kd(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    """Binary soft-target cross entropy, scaled by T² as in distillation."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    target = torch.sigmoid(teacher.detach().float() / temperature)
    return F.binary_cross_entropy_with_logits(student.float() / temperature, target) * temperature**2


def auxiliary_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    config: DistillationConfig,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if labels.ndim != 2 or labels.shape[1] != 5:
        raise ValueError("Labels must have shape [batch, 5]")
    teacher = outputs[2].detach()
    total = labels.new_zeros(())
    for weight, logits in zip(config.exit_weights, outputs[:2]):
        hard = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=pos_weight)
        kd = bernoulli_kd(logits, teacher, config.temperature) if config.kd_weight else hard.new_zeros(())
        total = total + weight * (config.hard_weight * hard + config.kd_weight * kd)
    return total


def freeze_baseline(model: AdaptiveResNet50) -> None:
    """Keep a trained baseline fixed while learning the two auxiliary heads."""
    for module in (model.backbone, model.final):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    for module in (model.exit1, model.exit2):
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def train_auxiliary_epoch(
    model: AdaptiveResNet50,
    batches,
    optimizer: torch.optim.Optimizer,
    config: DistillationConfig,
    pos_weight: torch.Tensor | None = None,
    scaler=None,
    mixed_precision: bool = False,
) -> float:
    """Train heads on train batches only; caller owns split and checkpoint selection."""
    freeze_baseline(model)
    total_loss, total_samples = 0.0, 0
    device = next(model.parameters()).device
    use_amp = device.type == "cuda" and mixed_precision
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    for images, labels in batches:
        images = images.to(device, non_blocking=use_amp)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=use_amp)
        weights = pos_weight.to(device) if pos_weight is not None else None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model.forward_all_exits(images)
            loss = auxiliary_loss(outputs, labels, config, weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(images)
        total_samples += len(images)
    if not total_samples:
        raise ValueError("Training loader is empty")
    return total_loss / total_samples


def evaluate_exit_losses(model: AdaptiveResNet50, batches) -> tuple[float, float]:
    """Validation-only per-exit BCE, useful for a with/without KD comparison."""
    model.eval()
    sums = [0.0, 0.0]
    count = 0
    with torch.no_grad():
        for images, labels in batches:
            device = next(model.parameters()).device
            images, labels = images.to(device), labels.to(device=device, dtype=torch.float32)
            outputs = model.forward_all_exits(images)
            for i in range(2):
                sums[i] += F.binary_cross_entropy_with_logits(outputs[i], labels, reduction="sum").item()
            count += labels.numel()
    if not count:
        raise ValueError("Validation loader is empty")
    return sums[0] / count, sums[1] / count


def compare_kd(
    baseline: AdaptiveResNet50,
    train_batches,
    validation_batches,
    epochs: int,
    learning_rate: float,
    config: DistillationConfig,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, tuple[float, float]]:
    """Train independent heads with identical starts; return validation BCE per exit.

    Inputs must be re-iterable train and validation loaders. No test data belongs here.
    """
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    results = {}
    for name, kd_weight in (("without_kd", 0.0), ("with_kd", config.kd_weight)):
        model = deepcopy(baseline)
        freeze_baseline(model)
        optimizer = torch.optim.Adam(
            list(model.exit1.parameters()) + list(model.exit2.parameters()), lr=learning_rate
        )
        variant = DistillationConfig(
            config.temperature, config.hard_weight, kd_weight, config.exit_weights
        )
        for _ in range(epochs):
            train_auxiliary_epoch(model, train_batches, optimizer, variant, pos_weight)
        results[name] = evaluate_exit_losses(model, validation_batches)
    return results
