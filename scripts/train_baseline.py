"""Train/resume the static final exit; --smoke runs one real forward only."""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.chestxray import ChestXrayDataset, LABELS, load_splits, make_loaders, positive_weights, training_class_stats
from src.models.adaptive import AdaptiveResNet50, load_radimagenet_backbone
from src.training.baseline import evaluate_baseline, resume_baseline, save_baseline, save_last, train_baseline_epoch
from src.utils.config import load_config, resolve_project_path, validate_run_name
from src.utils.run import run_signature, select_device, split_hashes, use_amp, write_history, write_run_metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, help="total epochs, including completed epochs on resume")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume from this run's last.pt")
    parser.add_argument("--smoke", action="store_true", help="one real image forward; no training or output")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run; cannot be combined with --resume")
    return parser.parse_args()


def configured_args(args):
    config = load_config(args.config)
    for name in ("dataset_root", "metadata_csv", "split_dir", "pretrained_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            config[name] = resolve_project_path(value)
    for name in ("run_name", "device", "epochs", "batch_size", "num_workers"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.no_amp:
        config["mixed_precision"] = False
    validate_run_name(config["run_name"])
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["num_workers"] < 0:
        raise ValueError("epochs, batch_size, and num_workers are invalid")
    return config


def main():
    args = parse_args()
    if args.resume and (args.overwrite or args.smoke):
        raise ValueError("--resume cannot be combined with --overwrite or --smoke")
    config = configured_args(args)
    device = select_device(config["device"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    splits = load_splits(config["split_dir"], config["dataset_root"])
    model = AdaptiveResNet50()
    checkpoint_dir = config["checkpoint_root"] / config["run_name"]
    run_dir = config["results_root"] / config["run_name"]
    best_path, last_path = checkpoint_dir / "best.pt", checkpoint_dir / "last.pt"
    if args.smoke:
        load_radimagenet_backbone(model, str(config["pretrained_checkpoint"]))
        model = model.to(device).eval()
        image, label = ChestXrayDataset(splits["train"], config["dataset_root"], config["image_size"])[0]
        with torch.no_grad():
            logits = model.forward_final(image.unsqueeze(0).to(device))
        print(json.dumps({"device": str(device), "amp_enabled_for_training": use_amp(config, device), "image_shape": list(image.shape), "label_shape": list(label.shape), "logits_shape": list(logits.shape), "finite_logits": bool(torch.isfinite(logits).all())}))
        return
    loaders = make_loaders(splits, config["dataset_root"], config["image_size"], config["batch_size"], config["seed"], config["num_workers"], config["pin_memory"] and device.type == "cuda")
    signature = run_signature(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(list(model.backbone.parameters()) + list(model.final.parameters()), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp(config, device))
    pos_weight = positive_weights(splits["train"]).to(device)
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint absent: {last_path}")
        state = resume_baseline(last_path, model, optimizer, scaler, signature, loaders["train"])
        start_epoch = state["epoch"] + 1
        best_score, best_epoch, history = state["best_score"], state["best_epoch"], state["history"]
        if best_epoch == state["epoch"]:
            best_file_epoch = None
            if best_path.is_file():
                best_file_epoch = torch.load(best_path, map_location="cpu", weights_only=True)["epoch"]
            if best_file_epoch != best_epoch:
                save_baseline(best_path, model, best_epoch, state["validation_metrics"], config["run_name"], split_hashes(config))
        write_history(run_dir / "history.csv", history)
    else:
        if not args.overwrite and any(p.exists() for p in (best_path, last_path, run_dir / "history.csv")):
            raise FileExistsError(f"Run already exists: {config['run_name']}; use --resume or a new --run-name")
        load_radimagenet_backbone(model, str(config["pretrained_checkpoint"]))
        start_epoch, best_score, best_epoch, history = 1, float("-inf"), None, []
        write_run_metadata(run_dir, config, device, signature)
        print(json.dumps({"train_class_stats": training_class_stats(splits["train"]), "device": str(device), "mixed_precision": use_amp(config, device)}))
    if start_epoch > config["epochs"]:
        raise ValueError(f"Checkpoint completed epoch {start_epoch - 1}; --epochs must be larger")
    for epoch in range(start_epoch, config["epochs"] + 1):
        train_loss = train_baseline_epoch(model, loaders["train"], optimizer, pos_weight, device, scaler, use_amp(config, device))
        metrics = evaluate_baseline(model, loaders["validation"], device)
        score = metrics["macro_auroc"]
        if score is None:
            raise ValueError("Validation AUROC undefined for every label; choose a larger split")
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        row = {"epoch": epoch, "train_loss": train_loss, "val_macro_auroc": score, "val_macro_f1": metrics["macro_f1"], "val_subset_accuracy": metrics["subset_accuracy"], "best_epoch": best_epoch, "best_macro_auroc": best_score}
        row.update({f"val_auroc_{label}": metrics["per_label_auroc"][label] for label in LABELS})
        history.append(row)
        save_last(last_path, model, optimizer, scaler, epoch, best_score, best_epoch, history, metrics, signature, loaders["train"])
        if improved:
            save_baseline(best_path, model, epoch, metrics, config["run_name"], split_hashes(config))
        write_history(run_dir / "history.csv", history)
        print(json.dumps({"epoch": epoch, "train_loss": train_loss, "validation": metrics, "best_epoch": best_epoch, "best_macro_auroc": best_score}))
    print(json.dumps({"best_checkpoint": str(best_path), "resume_checkpoint": str(last_path), "history": str(run_dir / 'history.csv')}))


if __name__ == "__main__":
    main()
