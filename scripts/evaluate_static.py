"""Evaluate a saved baseline on validation or the untouched final test split."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.chestxray import load_splits, make_loaders
from src.training.baseline import evaluate_baseline, load_baseline
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    config = load_config(args.config)
    splits = load_splits(config["split_dir"], config["dataset_root"])
    loaders = make_loaders(splits, config["dataset_root"], config["image_size"], config["batch_size"], config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_baseline(args.checkpoint, device)
    print(json.dumps({"split": args.split, "metrics": evaluate_baseline(model, loaders[args.split], device)}, indent=2))


if __name__ == "__main__":
    main()
