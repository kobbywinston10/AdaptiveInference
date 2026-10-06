"""CPU-only Phase 8 checks with synthetic data, never experimental results."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from scripts.benchmark_flops import main as flops_main
from scripts.benchmark_latency import main as latency_main
from scripts.calibrate import main as calibrate_main
from scripts.generate_pareto import main as pareto_main
from src.data.chestxray import save_splits
from src.evaluation.flops import count_exit_flops
from src.evaluation.latency import measure_latency
from src.evaluation.pareto import pareto_ids
from src.models.adaptive import AdaptiveResNet50
from src.utils.run import split_hashes


def test_flops_and_pareto_dominance():
    torch.set_num_threads(2)
    model = AdaptiveResNet50()
    counts = count_exit_flops(model, torch.zeros(1, 3, 32, 32))
    assert counts["fixed_exit1"] == counts["exit1"]
    assert counts["fixed_exit2"] < counts["exit2"]
    assert counts["full"] < counts["exit3"]
    assert counts["exit1"] < counts["exit2"] < counts["exit3"]
    points = [
        {"id": "a", "cost": 1, "score": 0.5},
        {"id": "b", "cost": 2, "score": 0.6},
        {"id": "c", "cost": 2, "score": 0.4},
        {"id": "d", "cost": 1, "score": 0.5},
    ]
    assert pareto_ids(points, "cost", "score") == {"a", "b", "d"}
    assert measure_latency(lambda _: None, torch.device("cpu"), 1, 2)["mean_ms"] >= 0
    with pytest.raises(ValueError):
        measure_latency(lambda _: None, torch.device("cpu"), 0, 0)


def test_phase8_synthetic_artifact_handoff(tmp_path, monkeypatch, capsys):
    torch.set_num_threads(2)
    root = tmp_path / "images"
    root.mkdir()
    rows = {split: [] for split in ("train", "validation", "test")}
    for split in rows:
        for index, finding in enumerate(("Atelectasis|Effusion|Mass|Nodule|Pneumothorax", "No Finding")):
            name = f"{split}_{index}.png"
            Image.fromarray(np.full((32, 32), 50 + 100 * index, dtype=np.uint8)).save(root / name)
            rows[split].append({"image": name, "patient_id": name, "finding_labels": finding})
    split_dir = tmp_path / "splits"
    save_splits(rows, split_dir)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/dev.yaml").read_text())
    config.update({"dataset_root": str(root), "split_dir": str(split_dir),
                   "checkpoint_root": str(tmp_path / "checkpoints"), "results_root": str(tmp_path / "results"),
                   "device": "cpu", "image_size": 32, "batch_size": 2, "num_workers": 0, "pin_memory": False})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = tmp_path / "checkpoints/synthetic/best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model_state": AdaptiveResNet50().state_dict(), "split_hashes": split_hashes({"split_dir": split_dir})}, checkpoint)
    monkeypatch.setattr(sys, "argv", ["calibrate.py", "--config", str(config_path), "--checkpoint", str(checkpoint), "--quantile-steps", "2"])
    calibrate_main()
    monkeypatch.setattr(sys, "argv", ["benchmark_flops.py", "--config", str(config_path), "--checkpoint", str(checkpoint)])
    flops_main()
    monkeypatch.setattr(sys, "argv", ["benchmark_latency.py", "--config", str(config_path), "--checkpoint", str(checkpoint), "--samples", "1", "--warmup", "0", "--measurements", "1"])
    latency_main()
    monkeypatch.setattr(sys, "argv", ["generate_pareto.py", "--config", str(config_path), "--checkpoint", str(checkpoint)])
    pareto_main()
    out = tmp_path / "results/synthetic"
    with (out / "pareto_points.csv").open(newline="", encoding="utf-8") as handle:
        points = list(csv.DictReader(handle))
    assert len(points) == len(json.loads((out / "latency.json").read_text())["points"])
    assert all(float(point["average_flops"]) > 0 for point in points)
    assert {point["kind"] for point in points} == {"static", "adaptive"}
    assert all((out / name).is_file() for name in (
        "plots/subset_accuracy_vs_flops.png", "plots/macro_auroc_vs_flops.png", "plots/subset_accuracy_vs_latency.png",
    ))
    assert (out / "flops.csv").is_file() and (out / "latency.csv").is_file()
    capsys.readouterr()
