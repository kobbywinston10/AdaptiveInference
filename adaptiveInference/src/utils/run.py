"""Run identity, hardware metadata, and durable training history."""

import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from src.data.chestxray import LABELS


HISTORY_FIELDS = ("epoch", "train_loss", "val_macro_auroc", "val_macro_f1", "val_subset_accuracy", "best_epoch", "best_macro_auroc") + tuple(f"val_auroc_{label}" for label in LABELS)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check DAS-6 GPU allocation and CUDA-enabled PyTorch")
    return torch.device(requested)


def use_amp(config: dict, device: torch.device) -> bool:
    return bool(config["mixed_precision"] and device.type == "cuda")


def serializable_config(config: dict) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}


def run_signature(config: dict) -> str:
    """Reject resuming with different data or training settings; epochs may grow."""
    relevant = serializable_config(config).copy()
    relevant.pop("epochs", None)
    relevant.pop("adaptive_epochs", None)
    relevant.pop("distillation", None)
    relevant.pop("adaptive_learning_rate", None)
    payload = json.dumps({"config": relevant, "splits": split_hashes(config)}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def split_hashes(config: dict) -> dict[str, str]:
    split_dir = config["split_dir"]
    return {split: hashlib.sha256((split_dir / f"{split}.csv").read_bytes()).hexdigest() for split in ("train", "validation", "test")}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_history(path: Path, rows: list[dict]) -> None:
    import io
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HISTORY_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def write_run_metadata(run_dir: Path, config: dict, device: torch.device, signature: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run_dir / "config.yaml", yaml.safe_dump(serializable_config(config), sort_keys=True))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": commit,
        "run_signature": signature,
        "seed": config["seed"],
    }
    atomic_write_text(run_dir / "environment.json", json.dumps(info, indent=2))
