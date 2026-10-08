"""Training-only calibration and graph diagnostics for static ONNX PTQ."""

import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader

from src.data.chestxray import ChestXrayDataset
from src.deployment.onnx_utils import INPUT_NAME


TARGET_OPS = ("Conv", "Gemm")


def training_reader(config: dict, samples: int, seed: int):
    """Sample training rows without replacement, then read in manifest order."""
    if not isinstance(samples, int) or samples < 1 or not isinstance(seed, int):
        raise ValueError("Calibration samples must be positive and seed must be an integer")
    with (config["split_dir"] / "train.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if samples > len(rows):
        raise ValueError(f"Requested {samples} calibration images, but training split has {len(rows)}")
    indices = sorted(random.Random(seed).sample(range(len(rows)), samples))
    dataset = ChestXrayDataset(rows, config["dataset_root"], config["image_size"])
    return TrainingCalibrationReader(dataset, indices), indices


class TrainingCalibrationReader(CalibrationDataReader):
    def __init__(self, dataset: ChestXrayDataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        self.position = 0

    def get_next(self):
        if self.position >= len(self.indices):
            return None
        image, _ = self.dataset[self.indices[self.position]]
        self.position += 1
        return {INPUT_NAME: np.asarray(image.unsqueeze(0).numpy(), dtype=np.float32)}

    def rewind(self):
        self.position = 0


def graph_summary(path: Path) -> dict:
    model = onnx.load(str(path))
    operators = dict(sorted(Counter(node.op_type for node in model.graph.node).items()))
    producers = {output: node.op_type for node in model.graph.node for output in node.output}
    qdq_wrapped = {name: sum(node.op_type == name and any(
        producers.get(value) == "DequantizeLinear" for value in node.input)
        for node in model.graph.node) for name in TARGET_OPS}
    return {"operator_counts": operators,
            "quantized_target_operators": {name: operators.get(name, 0) for name in TARGET_OPS},
            "qdq_wrapped_target_operators": qdq_wrapped,
            "other_operators": {name: count for name, count in operators.items() if name not in TARGET_OPS},
            "int8_initializers": sum(tensor.data_type == onnx.TensorProto.INT8 for tensor in model.graph.initializer),
            "opset": {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}}
