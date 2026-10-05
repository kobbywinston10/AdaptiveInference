"""Train Phase 4 auxiliary heads from an immutable Phase 3 best checkpoint."""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.chestxray import load_splits, make_loaders, positive_weights
from src.training.baseline import atomic_torch_save, load_baseline
from src.training.distillation import DistillationConfig, evaluate_exit_losses, freeze_baseline, train_auxiliary_epoch
from src.utils.config import load_config, resolve_project_path, validate_run_name
from src.utils.run import file_sha256, run_signature, select_device, split_hashes, use_amp, write_run_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--kd-weight", type=float, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_run_name(args.run_name)
    if args.kd_weight < 0:
        raise ValueError("kd_weight must be nonnegative")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    config["run_name"] = args.run_name
    config["distillation"]["kd_weight"] = args.kd_weight
    if args.device is not None:
        config["device"] = args.device
    for name in ("dataset_root", "split_dir"):
        value = getattr(args, name)
        if value is not None:
            config[name] = resolve_project_path(value)
    baseline_path = resolve_project_path(args.baseline_checkpoint)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Trained Phase 3 best checkpoint absent: {baseline_path}")
    baseline_header = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if baseline_header.get("split_hashes") != split_hashes(config):
        raise ValueError("Baseline checkpoint and Phase 4 split manifests differ")
    del baseline_header
    config["baseline_checkpoint"] = baseline_path
    config["baseline_checkpoint_sha256"] = file_sha256(baseline_path)
    device = select_device(config["device"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    splits = load_splits(config["split_dir"], config["dataset_root"])
    loaders = make_loaders(splits, config["dataset_root"], config["image_size"], config["batch_size"], config["seed"], config["num_workers"], config["pin_memory"] and device.type == "cuda")
    output = config["checkpoint_root"] / config["run_name"] / "best.pt"
    run_dir = config["results_root"] / config["run_name"]
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Phase 4 output exists: {output}")
    model = load_baseline(baseline_path, device)
    freeze_baseline(model)
    optimizer = torch.optim.Adam(list(model.exit1.parameters()) + list(model.exit2.parameters()), lr=config["adaptive_learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp(config, device))
    kd = config["distillation"]
    loss_config = DistillationConfig(kd["temperature"], kd["hard_weight"], kd["kd_weight"], tuple(kd["exit_weights"]))
    pos_weight = positive_weights(splits["train"]).to(device)
    write_run_metadata(run_dir, config, device, run_signature(config))
    best = float("inf")
    history = []
    epochs = args.epochs or config["adaptive_epochs"]
    for epoch in range(1, epochs + 1):
        loss = train_auxiliary_epoch(model, loaders["train"], optimizer, loss_config, pos_weight, scaler, use_amp(config, device))
        exit1_bce, exit2_bce = evaluate_exit_losses(model, loaders["validation"])
        row = {"epoch": epoch, "train_loss": loss, "exit1_val_bce": exit1_bce, "exit2_val_bce": exit2_bce}
        history.append(row)
        score = (exit1_bce + exit2_bce) / 2
        if score < best:
            best = score
            atomic_torch_save({"model_state": model.state_dict(), "epoch": epoch, "validation_exit_bce": (exit1_bce, exit2_bce), "baseline_checkpoint": str(baseline_path), "kd_weight": args.kd_weight}, output)
        print(json.dumps(row))
    from src.utils.run import atomic_write_text
    atomic_write_text(run_dir / "history.json", json.dumps(history, indent=2))
    print(json.dumps({"best_checkpoint": str(output), "history": str(run_dir / 'history.json')}))


if __name__ == "__main__":
    main()
