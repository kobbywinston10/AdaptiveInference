"""Deterministic, evaluation-only blur/noise on normalized grayscale RGB tensors."""

import hashlib
import math

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


KINDS = ("blur", "noise")
SEVERITIES = ("clean", "mild", "severe")


def gaussian_blur(image: torch.Tensor, sigma_pixels: float) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3 or sigma_pixels <= 0:
        raise ValueError("Expected [3, height, width] image and positive blur sigma")
    radius = math.ceil(3 * sigma_pixels)
    positions = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel_1d = torch.exp(-0.5 * (positions / sigma_pixels) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)
    weights = kernel.expand(3, 1, -1, -1)
    padded = F.pad(image.unsqueeze(0), (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, weights, groups=3).squeeze(0)


def gaussian_noise(image: torch.Tensor, std_normalized: float, seed: int) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3 or std_normalized <= 0:
        raise ValueError("Expected [3, height, width] image and positive noise std")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, *image.shape[-2:]), generator=generator, dtype=image.dtype)
    return (image + noise.expand_as(image) * std_normalized).clamp(-1, 1)


class DegradedDataset(Dataset):
    """Wrap an evaluation split; noise depends on image identity, not worker order."""

    def __init__(self, base: Dataset, kind: str, severity: str, parameters: dict, seed: int):
        if kind not in KINDS or severity not in SEVERITIES:
            raise ValueError("kind must be blur/noise and severity clean/mild/severe")
        self.base = base
        self.kind = kind
        self.severity = severity
        self.parameters = parameters
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, labels = self.base[index]
        if self.severity == "clean":
            return image, labels
        if self.kind == "blur":
            return gaussian_blur(image, self.parameters["blur_sigma_pixels"][self.severity]), labels
        identity = self.base.rows[index]["image"] if hasattr(self.base, "rows") else str(index)
        # Pair mild/severe by scaling the same draw for each image.
        digest = hashlib.sha256(f"{self.seed}|{identity}".encode()).digest()
        sample_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
        return gaussian_noise(image, self.parameters["noise_std_normalized"][self.severity], sample_seed), labels


def confident_wrong_labels(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float = 0.9) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flag images with at least one wrong label decided at high confidence."""
    if probabilities.shape != labels.shape or probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("Expected matching [images, 5] probabilities and labels")
    if not 0.5 < threshold <= 1:
        raise ValueError("Confidence threshold must lie in (0.5, 1]")
    predicted = probabilities >= 0.5
    wrong = predicted != labels.bool()
    confidence = torch.where(predicted, probabilities, 1 - probabilities)
    maximum_wrong = torch.where(wrong, confidence, torch.zeros_like(confidence)).max(dim=1).values
    return maximum_wrong >= threshold, wrong, maximum_wrong


def average_flops(exit_indices: torch.Tensor, measured_costs: dict[str, float]) -> float:
    """Mean executed FLOPs per image for actual adaptive exits.

    Each image contributes the measured ``exit1``, ``exit2``, or ``exit3``
    path cost, including the earlier heads needed for routing. Static path
    costs (``fixed_exit1``, ``fixed_exit2``, ``full``) are not used.
    """
    required = ("exit1", "exit2", "exit3")
    if exit_indices.ndim != 1 or len(exit_indices) == 0 or not ((exit_indices >= 1) & (exit_indices <= 3)).all():
        raise ValueError("Expected nonempty exit IDs 1, 2, or 3")
    try:
        costs = [float(measured_costs[key]) for key in required]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Measured costs need exit1, exit2, exit3") from error
    if not all(math.isfinite(cost) and cost > 0 for cost in costs) or not costs[0] < costs[1] < costs[2]:
        raise ValueError("Measured exit FLOPs must be finite, positive, and increasing")
    table = torch.tensor([0, *costs], dtype=torch.float64, device=exit_indices.device)
    return float(table[exit_indices.long()].mean())
