"""Batch-one warmed latency with explicit device synchronization."""

from collections.abc import Callable
from time import perf_counter_ns

import numpy as np
import torch


def measure_latency(run: Callable[[int], object], device: torch.device, warmup: int, measurements: int) -> dict[str, float]:
    if warmup < 0 or measurements < 1:
        raise ValueError("warmup must be nonnegative and measurements positive")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times = []
    with torch.inference_mode():
        for index in range(warmup):
            run(index)
        synchronize()
        for index in range(measurements):
            synchronize()
            start = perf_counter_ns()
            run(index)
            synchronize()
            times.append((perf_counter_ns() - start) / 1e6)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }
