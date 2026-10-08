"""CPU ONNX smoke test with synthetic images and an untrained model."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from scripts.evaluate_onnx import main as evaluate_main
from scripts.export_onnx import main as export_main
from src.data.chestxray import LABELS, save_splits
from src.deployment.onnx_utils import OUTPUT_NAMES, load_session
from src.models.adaptive import AdaptiveResNet50
from src.routing.policy import RoutingPolicy
from src.utils.run import file_sha256, split_hashes


def test_fp32_shared_graph_parity_and_offline_evaluation(tmp_path, monkeypatch, capsys):
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
    calibration = {"schema_version": 1, "fit_split": "validation", "threshold_selection_split": "validation",
                   "checkpoint_sha256": file_sha256(checkpoint), "split_hashes": hashes,
                   "policies": {"calibrated": {"policy": RoutingPolicy(-1, -1).to_dict()}}}
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["export_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--validation-samples", "2"])
    export_main()
    deployment = tmp_path / "results/synthetic/deployment"
    onnx_path = deployment / "adaptive_fp32.onnx"
    report = json.loads((deployment / "fp32_export_report.json").read_text())
    assert onnx_path.is_file() and report["onnx_size_bytes"] == onnx_path.stat().st_size
    assert report["input_shape"] == [1, 3, 32, 32]
    assert report["output_names"] == list(OUTPUT_NAMES)
    assert report["validation_parity"]["passed"]
    assert all(row["passed"] for row in report["validation_parity"]["per_exit"].values())
    assert len(load_session(onnx_path).get_outputs()) == 3
    monkeypatch.setattr(sys, "argv", ["evaluate_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--split", "validation",
                                       "--calibration", str(calibration_path), "--parity-samples", "2"])
    evaluate_main()
    evaluation = json.loads((deployment / "onnx_validation_report.json").read_text())
    assert evaluation["validation_parity"]["passed"]
    assert evaluation["frozen_calibrated_policy"]["metrics"]["exit_counts"] == [0, 0, 2]
    assert list(evaluation["exit_metrics"]) == list(OUTPUT_NAMES)
    with np.load(deployment / "onnx_validation_logits.npz") as logits:
        assert all(logits[name].shape == (2, 5) for name in OUTPUT_NAMES)
        assert tuple(logits["label_names"]) == LABELS
    monkeypatch.setattr(sys, "argv", ["evaluate_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--split", "test",
                                       "--calibration", str(calibration_path)])
    evaluate_main()
    test_evaluation = json.loads((deployment / "onnx_test_report.json").read_text())
    assert test_evaluation["validation_parity"] is None
    assert test_evaluation["frozen_calibrated_policy"]["policy"] == json.loads(
        json.dumps(calibration["policies"]["calibrated"]["policy"]))
    capsys.readouterr()
