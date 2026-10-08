"""Synthetic validation-only operating-point selection and frozen evaluation."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from scripts.evaluate_operating_points import main as evaluate_main
from scripts.evaluate_operating_points import evaluate_frozen
from scripts.select_operating_points import main as select_main
from src.data.chestxray import save_splits
from src.evaluation.operating_points import mean_executed_flops, select_operating_points
from src.models.adaptive import AdaptiveResNet50
from src.routing.policy import RoutingPolicy
from src.utils.run import file_sha256, split_hashes


def synthetic_artifacts():
    base = RoutingPolicy(0.03, 0.06, (0.7, 0.8, 1.0), "max")
    sweep = []
    for index in range(7):
        counts = [90 - 10 * index, 10, 10 * index]
        sweep.append({"threshold1": 0.01 * index, "threshold2": 0.02 * index,
                      "validation_macro_auroc": 0.60 + 0.01 * index,
                      "validation_bce": 0.6 - 0.01 * index,
                      "average_exit_depth": sum((i + 1) * n for i, n in enumerate(counts)) / 100,
                      "exit_counts": counts})
    sweep.append(dict(sweep[2]))  # Duplicate Pareto coordinate.
    calibration = {"fit_split": "validation", "threshold_selection_split": "validation",
                   "checkpoint_sha256": "synthetic", "policies": {"calibrated": {
                       "policy": base.to_dict(), "threshold_sweep": sweep},
                       "uncalibrated": {"threshold_sweep": [{"test_only": True}]}}}
    flops = {"checkpoint_sha256": "synthetic", "fixed_exit1": 0.5,
             "fixed_exit2": 1.5, "full": 2.5, "exit1": 1.0, "exit2": 2.0, "exit3": 3.0}
    return calibration, flops


def test_calibrated_validation_selection_is_deterministic_and_unique():
    calibration, flops = synthetic_artifacts()
    first = select_operating_points(calibration, flops, 5)
    assert first == select_operating_points(calibration, flops, 5)
    assert first["candidate_count"] == 8
    assert len(first["points"]) == len({p["id"] for p in first["points"]}) == 5
    assert [p["id"] for p in first["points"]] == [f"adaptive_op{i}" for i in range(1, 6)]
    assert all(a["mean_executed_flops"] < b["mean_executed_flops"]
               for a, b in zip(first["points"], first["points"][1:]))
    assert first["frozen_policy"]["included"]
    assert any(p["original_threshold_sweep_index"] == 3 for p in first["points"])
    for point in first["points"]:
        assert point["mean_executed_flops"] == pytest.approx(
            sum(n * flops[f"exit{i}"] for i, n in enumerate(point["exit_counts"], 1)) / 100)
        assert point["mean_executed_flops"] != pytest.approx(
            sum(n * flops[k] for n, k in zip(point["exit_counts"], ("fixed_exit1", "fixed_exit2", "full"))) / 100)
    changed = copy.deepcopy(calibration)
    changed["test_metrics"] = {"macro_auroc": -999}
    changed["policies"]["uncalibrated"]["threshold_sweep"] = [{"test_only": False}]
    assert select_operating_points(changed, flops, 5) == first
    changed["fit_split"] = "test"
    with pytest.raises(ValueError, match="validation only"):
        select_operating_points(changed, flops, 5)


def test_frozen_policies_reload_and_evaluate_without_reselection():
    calibration, flops = synthetic_artifacts()
    frozen = json.loads(json.dumps(select_operating_points(calibration, flops, 5)))
    assert all(RoutingPolicy.from_dict(point).aggregation == "max" for point in frozen["points"])
    logits = torch.tensor([[4., -4., 4., -4., 4.], [-4., 4., -4., 4., -4.],
                           [1., -1., 1., -1., 1.], [-1., 1., -1., 1., -1.]])
    labels = torch.tensor([[1., 0., 1., 0., 1.], [0., 1., 0., 1., 0.],
                           [1., 0., 1., 0., 1.], [0., 1., 0., 1., 0.]])
    exits = (logits, logits * 1.1, logits * 1.2)
    results = evaluate_frozen(exits, labels, frozen["points"], flops)
    assert [r["id"] for r in results] == [p["id"] for p in frozen["points"]]
    assert all(r["mean_executed_flops"] == pytest.approx(mean_executed_flops(r["exit_counts"], flops)) for r in results)
    assert all(r["macro_auroc"] is not None for r in results)


def test_synthetic_cli_freeze_then_evaluate_validation_and_test(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    images = tmp_path / "images"
    images.mkdir()
    rows = {}
    for split in ("train", "validation", "test"):
        rows[split] = []
        for index, finding in enumerate(("Atelectasis|Effusion|Mass|Nodule|Pneumothorax", "No Finding")):
            name = f"{split}_{index}.png"
            Image.fromarray(np.full((32, 32), 50 + 100 * index, dtype=np.uint8)).save(images / name)
            rows[split].append({"image": name, "patient_id": name, "finding_labels": finding})
    split_dir = tmp_path / "splits"
    save_splits(rows, split_dir)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/dev.yaml").read_text())
    config.update({"dataset_root": str(images), "split_dir": str(split_dir),
                   "checkpoint_root": str(tmp_path / "checkpoints"), "results_root": str(tmp_path / "results"),
                   "device": "cpu", "image_size": 32, "batch_size": 2, "num_workers": 0, "pin_memory": False})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = tmp_path / "checkpoints/synthetic/best.pt"
    checkpoint.parent.mkdir(parents=True)
    hashes = split_hashes({"split_dir": split_dir})
    torch.save({"model_state": AdaptiveResNet50().state_dict(), "split_hashes": hashes}, checkpoint)
    calibration, flops = synthetic_artifacts()
    digest = file_sha256(checkpoint)
    calibration.update({"schema_version": 1, "split_hashes": hashes, "checkpoint_sha256": digest})
    flops.update({"checkpoint_sha256": digest, "image_size": 32, "method": "synthetic_test_costs"})
    out = tmp_path / "results/synthetic"
    out.mkdir(parents=True)
    (out / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    (out / "flops.json").write_text(json.dumps(flops), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["select_operating_points.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint)])
    select_main()
    frozen = json.loads((out / "operating_points.json").read_text())
    assert frozen["selection_split"] == "validation" and len(frozen["points"]) == 5
    for split in ("validation", "test"):
        monkeypatch.setattr(sys, "argv", ["evaluate_operating_points.py", "--config", str(config_path),
                                           "--checkpoint", str(checkpoint), "--split", split])
        evaluate_main()
        report = json.loads((out / f"operating_points_{split}.json").read_text())
        assert [p["id"] for p in report["adaptive_points"]] == [p["id"] for p in frozen["points"]]
        assert [p["id"] for p in report["fixed_points"]] == ["fixed_exit1", "fixed_exit2", "full"]
        assert (out / f"operating_points_{split}.png").is_file()
