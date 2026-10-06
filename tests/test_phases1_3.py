import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.chestxray import (
    ChestXrayDataset,
    assert_no_patient_leakage,
    available_records,
    load_splits,
    parse_labels,
    positive_weights,
    save_splits,
    split_patients,
    training_class_stats,
)
from src.models.adaptive import AdaptiveResNet50
from src.training.baseline import evaluate_baseline, load_baseline, resume_baseline, save_baseline, save_last, train_baseline_epoch
from src.training.distillation import DistillationConfig, freeze_baseline, train_auxiliary_epoch
from src.utils.config import load_config
from src.utils.run import run_signature, select_device, use_amp, write_history


def test_config_loads_repository_relative_paths():
    config = load_config("configs/dev.yaml")
    assert config["dataset_root"].is_absolute()
    assert config["metadata_csv"].is_absolute()
    assert config["pretrained_checkpoint"].is_absolute()
    full = load_config("configs/full.yaml")
    assert full["device"] == "cuda" and full["mixed_precision"]
    assert full["dataset_root"].is_absolute() and full["checkpoint_root"].is_absolute()


def test_labels_are_multilabel():
    assert parse_labels("Atelectasis|Effusion") == (1, 1, 0, 0, 0)
    assert parse_labels("No Finding") == (0, 0, 0, 0, 0)


def test_local_filter_split_and_dataset(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    metadata = tmp_path / "metadata.csv"
    rows = []
    labels = ("Atelectasis", "Effusion", "Mass", "Nodule", "Pneumothorax")
    for i in range(20):
        name = f"{i}.png"
        Image.fromarray(np.full((12, 12), i, dtype=np.uint8)).save(root / name)
        rows.append({"Image Index": name, "Patient ID": str(i), "Finding Labels": labels[i % 5]})
    rows.append({"Image Index": "missing.png", "Patient ID": "missing", "Finding Labels": "Mass"})
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    records = available_records(metadata, root)
    assert len(records) == 20
    splits = split_patients(records, 42)
    assert splits == split_patients(records, 42)
    assert_no_patient_leakage(splits)
    split_dir = tmp_path / "splits"
    save_splits(splits, split_dir)
    loaded = load_splits(split_dir, root)
    assert sum(map(len, loaded.values())) == 20
    image, target = ChestXrayDataset(loaded["train"], root, 32)[0]
    assert image.shape == (3, 32, 32) and target.shape == (5,)
    assert image.min() >= -1 and image.max() <= 1
    assert torch.equal(image[0], image[1])
    with pytest.raises(FileExistsError):
        save_splits(splits, split_dir)
    corrupted = dict(loaded)
    corrupted["test"] = loaded["test"] + loaded["train"][:1]
    with pytest.raises(ValueError, match="Patient leakage"):
        assert_no_patient_leakage(corrupted)


def test_weights_use_training_rows_only():
    rows = [
        {"finding_labels": "Atelectasis|Effusion|Mass|Nodule|Pneumothorax"},
        {"finding_labels": "No Finding"},
        {"finding_labels": "No Finding"},
    ]
    assert torch.equal(positive_weights(rows), torch.full((5,), 2.0))
    stats = training_class_stats(rows)
    assert stats["Mass"]["prevalence"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="No positive"):
        positive_weights(rows[1:])


def test_static_training_metrics_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    model = AdaptiveResNet50()
    optimizer = torch.optim.SGD(list(model.backbone.parameters()) + list(model.final.parameters()), lr=0.001)
    batch = [(torch.randn(2, 3, 64, 64), torch.tensor([[1., 0., 1., 0., 1.], [0., 1., 0., 1., 0.]]))]
    before = model.final[2].weight.detach().clone()
    loss = train_baseline_epoch(model, batch, optimizer, torch.ones(5), torch.device("cpu"))
    assert loss > 0 and not torch.equal(model.final[2].weight, before)
    metrics = evaluate_baseline(model, batch, torch.device("cpu"))
    assert metrics["evaluated_images"] == 2
    assert set(metrics["per_label_auroc"]) == {"Atelectasis", "Effusion", "Mass", "Nodule", "Pneumothorax"}
    assert 0 <= metrics["subset_accuracy"] <= 1
    checkpoint = tmp_path / "baseline.pt"
    save_baseline(checkpoint, model, 1, metrics, "synthetic_test")
    restored = load_baseline(checkpoint, torch.device("cpu"))
    assert torch.equal(restored.final[2].weight, model.final[2].weight)
    freeze_baseline(restored)
    head_optimizer = torch.optim.SGD(list(restored.exit1.parameters()) + list(restored.exit2.parameters()), lr=0.001)
    teacher_before = restored.final[2].weight.detach().clone()
    assert train_auxiliary_epoch(restored, batch, head_optimizer, DistillationConfig(kd_weight=0)) > 0
    assert torch.equal(restored.final[2].weight, teacher_before)
    loader = type("Loader", (), {"generator": torch.Generator().manual_seed(42)})()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history = [{"epoch": 1, "train_loss": loss, "val_macro_auroc": metrics["macro_auroc"], "val_macro_f1": metrics["macro_f1"], "val_subset_accuracy": metrics["subset_accuracy"], "best_epoch": 1, "best_macro_auroc": metrics["macro_auroc"]}]
    last_path = tmp_path / "last.pt"
    save_last(last_path, model, optimizer, scaler, 1, metrics["macro_auroc"], 1, history, metrics, "same-data", loader)
    resumed_optimizer = torch.optim.SGD(list(restored.backbone.parameters()) + list(restored.final.parameters()), lr=0.001)
    state = resume_baseline(last_path, restored, resumed_optimizer, scaler, "same-data", loader)
    assert state["epoch"] == 1 and state["best_epoch"] == 1
    assert state["history"] == history
    with pytest.raises(ValueError, match="differ"):
        resume_baseline(last_path, restored, resumed_optimizer, scaler, "changed-data", loader)
    write_history(tmp_path / "history.csv", state["history"])
    with (tmp_path / "history.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_gpu_policy_and_split_signature(tmp_path):
    config = load_config("configs/dev.yaml")
    config["split_dir"] = tmp_path
    for name in ("train", "validation", "test"):
        (tmp_path / f"{name}.csv").write_text(name)
    first = run_signature(config)
    config["epochs"] += 1
    assert run_signature(config) == first
    (tmp_path / "train.csv").write_text("changed")
    assert run_signature(config) != first
    assert not use_amp(config, torch.device("cpu"))
    assert select_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA requested"):
            select_device("cuda")
