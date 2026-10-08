"""Load a frozen Phase 4 model and its matching validation-fitted policy."""

import json
from pathlib import Path

import torch

from src.data.chestxray import load_splits
from src.models.adaptive import AdaptiveResNet50
from src.routing.policy import load_policy
from src.utils.run import split_hashes


def load_evaluation_context(config: dict, checkpoint_path: Path, calibration_path: Path, mode: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    digest = split_hashes(config)
    if checkpoint.get("split_hashes") != digest:
        raise ValueError("Checkpoint and split manifests differ")
    artifact = json.loads(calibration_path.read_text(encoding="utf-8"))
    if artifact.get("split_hashes") != digest:
        raise ValueError("Calibration artifact and split manifests differ")
    policy = load_policy(calibration_path, mode, checkpoint_path)
    splits = load_splits(config["split_dir"], config["dataset_root"])
    model = AdaptiveResNet50()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), policy, splits
