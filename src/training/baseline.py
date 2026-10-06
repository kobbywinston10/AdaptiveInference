"""Static five-label baseline training and held-out evaluation."""

from pathlib import Path
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn import functional as F

from src.data.chestxray import LABELS
from src.models.adaptive import AdaptiveResNet50


def evaluate_baseline(model: AdaptiveResNet50, loader, device: torch.device) -> dict:
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits.append(model.forward_final(images.to(device)).cpu())
            targets.append(labels.cpu())
    if not logits:
        raise ValueError("Evaluation loader is empty")
    probabilities = torch.sigmoid(torch.cat(logits)).numpy()
    truth = torch.cat(targets).numpy().astype(int)
    predictions = (probabilities >= 0.5).astype(int)
    per_label = {}
    for i, label in enumerate(LABELS):
        per_label[label] = float(roc_auc_score(truth[:, i], probabilities[:, i])) if len(np.unique(truth[:, i])) == 2 else None
    defined = [value for value in per_label.values() if value is not None]
    return {
        "macro_auroc": float(np.mean(defined)) if defined else None,
        "per_label_auroc": per_label,
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(truth == predictions, axis=1))),
        "evaluated_images": len(truth),
    }


def train_baseline_epoch(model, loader, optimizer, pos_weight, device, scaler=None, mixed_precision=False) -> float:
    model.train()
    use_amp = device.type == "cuda" and mixed_precision
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    total_loss, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=use_amp), labels.to(device, non_blocking=use_amp)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model.forward_final(images)
            # BCEWithLogitsLoss is evaluated in float32 for stable multi-label loss.
            loss = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=pos_weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(images)
        count += len(images)
    if not count:
        raise ValueError("Training loader is empty")
    return total_loss / count


def save_baseline(path: Path, model: AdaptiveResNet50, epoch: int, metrics: dict, config_name: str, split_hashes: dict | None = None) -> None:
    atomic_torch_save({"model_state": model.state_dict(), "epoch": epoch, "validation_metrics": metrics, "config_name": config_name, "split_hashes": split_hashes}, path)


def load_baseline(path: Path, device: torch.device) -> AdaptiveResNet50:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = AdaptiveResNet50()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng(loader) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": loader.generator.get_state(),
    }


def restore_rng(state: dict, loader) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    loader.generator.set_state(state["loader"])


def save_last(path: Path, model, optimizer, scaler, epoch, best_score, best_epoch, history, validation_metrics, run_signature, loader) -> None:
    atomic_torch_save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "history": history,
        "validation_metrics": validation_metrics,
        "run_signature": run_signature,
        "rng_state": capture_rng(loader),
    }, path)


def resume_baseline(path: Path, model, optimizer, scaler, run_signature, loader) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["run_signature"] != run_signature:
        raise ValueError("Resume config or split manifests differ from the checkpoint")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    restore_rng(checkpoint["rng_state"], loader)
    return checkpoint
