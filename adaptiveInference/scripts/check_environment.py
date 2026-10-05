"""Check the selected configuration, local data paths, and checkpoint."""

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils.config import load_config
from src.models.adaptive import AdaptiveResNet50, load_radimagenet_backbone
from src.utils.run import select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--check-checkpoint", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    selected = args.device or config["device"]
    device = select_device(selected)
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
        "mixed_precision_for_training": bool(config["mixed_precision"] and device.type == "cuda"),
        "dataset_root": str(config["dataset_root"]),
        "metadata_exists": config["metadata_csv"].is_file(),
        "checkpoint_exists": config["pretrained_checkpoint"].is_file(),
        "local_png_count": sum(1 for _ in config["dataset_root"].rglob("*.png")) if config["dataset_root"].is_dir() else 0,
    }
    if args.check_checkpoint and report["checkpoint_exists"]:
        load_radimagenet_backbone(AdaptiveResNet50(), str(config["pretrained_checkpoint"]))
        report["checkpoint_load_valid"] = True
    print(json.dumps(report, indent=2))
    if not all((config["dataset_root"].is_dir(), report["metadata_exists"], report["checkpoint_exists"], report["local_png_count"] > 0)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
