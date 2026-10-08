"""Small synthetic run checks CLI artifact handoff; no real dataset training."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from scripts.train_adaptive import main as adaptive_main
from scripts.train_baseline import main as baseline_main
from scripts.calibrate import main as calibrate_main
from scripts.evaluate_adaptive import main as budget_eval_main
from scripts.run_degradation_experiment import main as degradation_main
from src.data.chestxray import save_splits
from src.routing.policy import load_policy


def test_baseline_resume_and_phase4_handoff(tmp_path, monkeypatch, capsys):
    torch.set_num_threads(2)
    root = tmp_path / "images"
    root.mkdir()
    rows = {name: [] for name in ("train", "validation", "test")}
    for split in rows:
        for index, finding in enumerate(("Atelectasis|Effusion|Mass|Nodule|Pneumothorax", "No Finding")):
            name = f"{split}_{index}.png"
            Image.fromarray(np.full((32, 32), 40 + index * 100, dtype=np.uint8)).save(root / name)
            rows[split].append({"image": name, "patient_id": f"{split}_{index}", "finding_labels": finding})
    split_dir = tmp_path / "splits"
    save_splits(rows, split_dir)
    repo = Path(__file__).parents[1]
    if not (repo / "weights/ResNet50.pt").is_file():
        pytest.skip("Optional local RadImageNet checkpoint is absent")
    config = yaml.safe_load((repo / "configs/dev.yaml").read_text(encoding="utf-8"))
    config.update({
        "dataset_root": str(root),
        "metadata_csv": str(tmp_path / "unused.csv"),
        "pretrained_checkpoint": str(repo / "weights/ResNet50.pt"),
        "split_dir": str(split_dir),
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "results_root": str(tmp_path / "results"),
        "run_name": "synthetic_baseline",
        "device": "cpu",
        "num_workers": 0,
        "pin_memory": False,
        "image_size": 32,
        "batch_size": 2,
        "epochs": 1,
        "adaptive_epochs": 1,
    })
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_baseline.py", "--config", str(config_path)])
    baseline_main()
    best = tmp_path / "checkpoints/synthetic_baseline/best.pt"
    last = tmp_path / "checkpoints/synthetic_baseline/last.pt"
    assert best.is_file() and last.is_file()
    assert torch.load(last, weights_only=True)["epoch"] == 1
    monkeypatch.setattr(sys, "argv", ["train_baseline.py", "--config", str(config_path), "--resume", "--epochs", "2"])
    baseline_main()
    assert torch.load(last, weights_only=True)["epoch"] == 2
    with (tmp_path / "results/synthetic_baseline/history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == 2
    best_epoch = torch.load(best, weights_only=True)["epoch"]
    assert float(history[best_epoch - 1]["val_macro_auroc"]) == max(float(row["val_macro_auroc"]) for row in history)
    for value, name in (("0", "synthetic_no_kd"), ("1", "synthetic_with_kd")):
        monkeypatch.setattr(sys, "argv", ["train_adaptive.py", "--config", str(config_path), "--baseline-checkpoint", str(best), "--kd-weight", value, "--run-name", name, "--epochs", "1"])
        adaptive_main()
        phase4 = torch.load(tmp_path / "checkpoints" / name / "best.pt", weights_only=True)
        assert phase4["kd_weight"] == float(value)
        assert phase4["baseline_checkpoint"] == str(best)
    adaptive_checkpoint = tmp_path / "checkpoints/synthetic_with_kd/best.pt"
    monkeypatch.setattr(sys, "argv", ["calibrate.py", "--config", str(config_path), "--checkpoint", str(adaptive_checkpoint), "--device", "cpu", "--quantile-steps", "3"])
    calibrate_main()
    artifact = tmp_path / "results/synthetic_with_kd/calibration.json"
    assert artifact.is_file()
    assert load_policy(artifact, "calibrated", adaptive_checkpoint).temperatures[0] > 0
    assert load_policy(artifact, "uncalibrated", adaptive_checkpoint).temperatures == (1., 1., 1.)
    monkeypatch.setattr(sys, "argv", ["evaluate_adaptive.py", "--config", str(config_path), "--checkpoint", str(adaptive_checkpoint), "--device", "cpu"])
    budget_eval_main()
    import json
    budget_report = json.loads((tmp_path / "results/synthetic_with_kd/budget_validation_calibrated.json").read_text())
    assert set(budget_report["budgets"]) == {"LOW", "MEDIUM", "HIGH"}
    assert len(budget_report["budgets"]["LOW"]["records"]) == 2
    monkeypatch.setattr(sys, "argv", ["run_degradation_experiment.py", "--config", str(config_path), "--checkpoint", str(adaptive_checkpoint), "--device", "cpu", "--degradation", "both"])
    degradation_main()
    experiment = json.loads((tmp_path / "results/synthetic_with_kd/degradation_validation_calibrated.json").read_text())
    assert len(experiment["conditions"]) == 18
    assert {row["budget"] for row in experiment["conditions"]} == {"LOW", "MEDIUM", "HIGH"}
    assert all(row["average_flops_per_image"] is None for row in experiment["conditions"])
    clean_uncertainty = [row["mean_exit1_uncertainty"] for row in experiment["conditions"] if row["severity"] == "clean"]
    assert len(set(clean_uncertainty)) == 1
    assert (tmp_path / "results/synthetic_with_kd/degradation_validation_calibrated_confident_errors.csv").is_file()
    capsys.readouterr()
