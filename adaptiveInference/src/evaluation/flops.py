"""Consistent module-hook arithmetic count for all inference paths."""

from collections.abc import Callable

import torch
from torch import nn

from src.models.resnet import Bottleneck


METHOD = "module_hooks_2_flops_per_mac_plus_bn_relu_pool_residual_v1"


def count_flops(model: nn.Module, forward: Callable[[], object]) -> int:
    """Count executed arithmetic operations; excludes tensor indexing and entropy."""
    total = 0

    def hook(module, inputs, output):
        nonlocal total
        x = inputs[0]
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            total += output.numel() * (2 * (module.in_channels // module.groups) * kernel + int(module.bias is not None))
        elif isinstance(module, nn.Linear):
            total += output.numel() * (2 * module.in_features + int(module.bias is not None))
        elif isinstance(module, nn.BatchNorm2d):
            total += 2 * output.numel()
        elif isinstance(module, nn.ReLU):
            total += output.numel()
        elif isinstance(module, nn.MaxPool2d):
            size = module.kernel_size
            area = size * size if isinstance(size, int) else size[0] * size[1]
            total += (area - 1) * output.numel()
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            total += x.numel() + output.numel()
        elif isinstance(module, Bottleneck):
            total += output.numel()  # residual add

    types = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, Bottleneck)
    handles = [module.register_forward_hook(hook) for module in model.modules() if isinstance(module, types)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            forward()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total


def count_exit_flops(model, image: torch.Tensor) -> dict[str, int]:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("FLOP counting requires one three-channel image")
    result = {}
    for index in (1, 2, 3):
        result[f"fixed_exit{index}" if index < 3 else "full"] = count_flops(model, lambda index=index: model.forward_to_exit(image, index))
        result[f"exit{index}"] = count_flops(model, lambda index=index: model.forward_adaptive(image, lambda candidate, _: candidate == index))
    if not (0 < result["exit1"] < result["exit2"] < result["exit3"]):
        raise ValueError("Adaptive exit FLOPs are not strictly increasing")
    return result
