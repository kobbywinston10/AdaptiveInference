"""Repository-relative configuration and reproducible paths."""

from pathlib import Path
import re

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    required = {"dataset_root", "metadata_csv", "pretrained_checkpoint", "split_dir", "seed", "image_size", "batch_size", "epochs", "learning_rate", "run_name", "device", "mixed_precision", "num_workers", "pin_memory", "checkpoint_root", "results_root", "adaptive_epochs", "adaptive_learning_rate", "distillation"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    for key in ("dataset_root", "metadata_csv", "pretrained_checkpoint", "split_dir", "checkpoint_root", "results_root"):
        config[key] = resolve_project_path(config[key])
    if config["batch_size"] < 1 or config["image_size"] < 32 or config["epochs"] < 1 or config["learning_rate"] <= 0 or config["adaptive_epochs"] < 1 or config["adaptive_learning_rate"] <= 0:
        raise ValueError("Invalid image size, batch size, epochs, or learning rate")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if not isinstance(config["mixed_precision"], bool) or not isinstance(config["pin_memory"], bool):
        raise ValueError("mixed_precision and pin_memory must be booleans")
    if not isinstance(config["num_workers"], int) or config["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    validate_run_name(config["run_name"])
    kd = config["distillation"]
    if not isinstance(kd, dict) or set(kd) != {"temperature", "hard_weight", "kd_weight", "exit_weights"}:
        raise ValueError("distillation needs temperature, hard_weight, kd_weight, exit_weights")
    if not isinstance(kd["exit_weights"], list) or len(kd["exit_weights"]) != 2 or kd["temperature"] <= 0 or min(kd["hard_weight"], kd["kd_weight"], *kd["exit_weights"]) < 0:
        raise ValueError("Invalid distillation parameters")
    degradation = config.get("degradation")
    if not isinstance(degradation, dict) or set(degradation) != {"blur_sigma_pixels", "noise_std_normalized"}:
        raise ValueError("degradation needs blur_sigma_pixels and noise_std_normalized")
    for name, values in degradation.items():
        if not isinstance(values, dict) or set(values) != {"mild", "severe"} or not all(isinstance(value, (int, float)) and value > 0 for value in values.values()) or values["mild"] >= values["severe"]:
            raise ValueError(f"Invalid degradation severities for {name}")
    config.setdefault("max_images", None)
    if config["max_images"] is not None and (not isinstance(config["max_images"], int) or config["max_images"] < 1):
        raise ValueError("max_images must be null or a positive integer")
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_run_name(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None:
        raise ValueError("run_name must contain only letters, digits, underscore, or hyphen")
