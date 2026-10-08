"""The local runner's orchestration is tested without starting training."""

import json
import sys

import numpy as np
import yaml
from PIL import Image

from scripts import run_local_e2e
from src.data.chestxray import save_splits


def test_local_e2e_plan_and_command_sequence(tmp_path, monkeypatch, capsys):
    images = tmp_path / "images"
    images.mkdir()
    rows = {}
    for split in ("train", "validation", "test"):
        rows[split] = []
        for index, finding in enumerate(("Atelectasis|Effusion|Mass|Nodule|Pneumothorax", "No Finding")):
            filename = f"{split}_{index}.png"
            Image.fromarray(np.full((32, 32), 80 + index * 50, dtype=np.uint8)).save(images / filename)
            rows[split].append({"image": filename, "patient_id": filename, "finding_labels": finding})
    splits = tmp_path / "splits"
    save_splits(rows, splits)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"preflight existence check only")
    config = yaml.safe_load((run_local_e2e.PROJECT_ROOT / "configs/tiny.yaml").read_text())
    config.update({"dataset_root": str(images), "metadata_csv": str(tmp_path / "metadata.csv"),
                   "pretrained_checkpoint": str(weights), "split_dir": str(splits),
                   "checkpoint_root": str(tmp_path / "checkpoints"), "results_root": str(tmp_path / "results")})
    (tmp_path / "metadata.csv").write_text("synthetic fixture\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["run_local_e2e.py", "--config", str(config_path)])
    run_local_e2e.main()
    assert "\"training_epochs_per_stage\": 1" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()
    calls = []

    def fake_command(script, *arguments, capture=False):
        calls.append((script, arguments))
        return json.dumps({"split": "validation", "metrics": {}}) if capture else None

    monkeypatch.setattr(run_local_e2e, "command", fake_command)
    monkeypatch.setattr(sys, "argv", ["run_local_e2e.py", "--config", str(config_path), "--run"])
    run_local_e2e.main()
    assert [script for script, _ in calls] == [
        "train_baseline.py", "train_baseline.py", "evaluate_static.py", "train_adaptive.py",
        "calibrate.py", "evaluate_adaptive.py", "benchmark_flops.py",
        "run_degradation_experiment.py", "benchmark_latency.py", "generate_pareto.py",
    ]
    assert all("test" not in arguments for _, arguments in calls)
    summary = json.loads((tmp_path / "results/local_e2e_baseline/local_e2e_summary.json").read_text())
    assert summary["status"] == "completed_smoke_test"
