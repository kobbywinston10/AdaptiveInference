"""Three shared-weight ResNet50 exits for five independent chest X-ray labels."""

from collections.abc import Callable

import torch
from torch import nn

from .resnet import ResNet50Backbone


class AdaptiveResNet50(nn.Module):
    def __init__(self, num_labels: int = 5):
        super().__init__()
        if num_labels != 5:
            raise ValueError("This project requires exactly five labels")
        self.backbone = ResNet50Backbone()
        self.exit1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 5))
        self.exit2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, 5))
        self.final = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2048, 5))

    def forward_all_exits(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.backbone.layer1(self.backbone.stem(x))
        x = self.backbone.layer2(x)
        first = self.exit1(x)
        x = self.backbone.layer3(x)
        second = self.exit2(x)
        x = self.backbone.layer4(x)
        return first, second, self.final(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.forward_all_exits(x)

    def forward_final(self, x: torch.Tensor) -> torch.Tensor:
        """Static full-depth baseline without computing auxiliary heads."""
        x = self.backbone.layer1(self.backbone.stem(x))
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        return self.final(self.backbone.layer4(x))

    def forward_adaptive(
        self,
        x: torch.Tensor,
        should_exit: Callable[[int, torch.Tensor], bool],
    ) -> tuple[torch.Tensor, int]:
        """Conditionally run later layers; the callback sees logits at each early exit.

        This batch-wide interface requires a single image. Phase 5 can supply an
        entropy/threshold callback, while fixed-exit baselines use a fixed callback.
        """
        if x.shape[0] != 1:
            raise ValueError("forward_adaptive requires batch size one")
        x = self.backbone.layer1(self.backbone.stem(x))
        x = self.backbone.layer2(x)
        logits = self.exit1(x)
        if should_exit(1, logits):
            return logits, 1
        x = self.backbone.layer3(x)
        logits = self.exit2(x)
        if should_exit(2, logits):
            return logits, 2
        return self.final(self.backbone.layer4(x)), 3


def load_radimagenet_backbone(model: AdaptiveResNet50, path: str) -> None:
    """Load the notebook's `backbone.0..8` checkpoint with strict shape checks."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    while isinstance(checkpoint, dict) and any(isinstance(checkpoint.get(key), dict) for key in ("state_dict", "model_state", "model", "weights")):
        checkpoint = next(checkpoint[key] for key in ("state_dict", "model_state", "model", "weights") if isinstance(checkpoint.get(key), dict))
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint state dictionary")
    keys = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
    indices = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
    mapped = {}
    ignored = []
    for key, value in checkpoint.items():
        key = key.removeprefix("module.")
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        first, _, rest = key.partition(".")
        if first in indices:
            key = indices[first] + "." + rest
        if key.startswith(keys) and isinstance(value, torch.Tensor):
            mapped[key] = value
        elif isinstance(value, torch.Tensor) and not key.startswith(("fc.", "classifier.")):
            ignored.append(key)
    if ignored:
        raise ValueError(f"Unrecognized checkpoint tensors: {ignored[:8]}")
    missing, unexpected = model.backbone.load_state_dict(mapped, strict=False)
    # avgpool has no parameters; every trainable backbone tensor must be present.
    if missing or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
