"""Synthetic CPU smoke test for training-only static ONNX quantization."""

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

from scripts.benchmark_onnx import benchmark_pair, main as benchmark_main
from scripts.evaluate_onnx import main as evaluate_main
from scripts.export_onnx import main as export_main
from scripts.quantize_onnx import main as quantize_main
from src.data.chestxray import save_splits
from src.deployment.onnx_utils import load_session, model_footprint, run_all_exits
from src.deployment.quantization import training_reader
from src.evaluation.onnx_comparison import compare_variants
from src.models.adaptive import AdaptiveResNet50
from src.routing.policy import RoutingPolicy
from src.utils.run import file_sha256, split_hashes


def test_static_ptq_uses_training_only_and_loads_three_exits(tmp_path, monkeypatch, capsys):
    torch.set_num_threads(2)
    images = tmp_path / "images"
    images.mkdir()
    rows = {split: [] for split in ("train", "validation", "test")}
    for split, count in (("train", 4), ("validation", 2), ("test", 1)):
        for index in range(count):
            name = f"{split}_{index}.png"
            if split != "test":  # A missing test image catches accidental test access.
                Image.fromarray(np.full((32, 32), 35 + 35 * index, dtype=np.uint8)).save(images / name)
            rows[split].append({"image": name, "patient_id": name,
                                "finding_labels": "Atelectasis" if index % 2 == 0 else "No Finding"})
    split_dir = tmp_path / "splits"
    save_splits(rows, split_dir)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/dev.yaml").read_text())
    config.update({"dataset_root": str(images), "split_dir": str(split_dir),
                   "checkpoint_root": str(tmp_path / "checkpoints"), "results_root": str(tmp_path / "results"),
                   "device": "cpu", "image_size": 32, "batch_size": 1, "num_workers": 0, "pin_memory": False})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = tmp_path / "checkpoints/synthetic/best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model_state": AdaptiveResNet50().state_dict(),
                "split_hashes": split_hashes({"split_dir": split_dir})}, checkpoint)
    monkeypatch.setattr(sys, "argv", ["export_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--validation-samples", "2"])
    export_main()
    reader1, indices1 = training_reader({"split_dir": split_dir, "dataset_root": images, "image_size": 32}, 3, 42)
    reader2, indices2 = training_reader({"split_dir": split_dir, "dataset_root": images, "image_size": 32}, 3, 42)
    assert indices1 == indices2 and len(set(indices1)) == 3
    assert reader1.get_next()["image"].shape == (1, 3, 32, 32)
    reader1.rewind()
    assert np.array_equal(reader1.get_next()["image"], reader2.get_next()["image"])
    monkeypatch.setattr(sys, "argv", ["quantize_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--calibration-samples", "3"])
    quantize_main()
    deployment = tmp_path / "results/synthetic/deployment"
    int8 = deployment / "adaptive_int8.onnx"
    report = json.loads((deployment / "ptq_report.json").read_text())
    assert int8.is_file()
    assert report["calibration_split"] == "train" and report["calibration_sample_count"] == 3
    assert report["selected_training_indices"] == indices1
    assert report["quantization_format"] == "QDQ"
    assert report["int8_footprint"] == model_footprint(int8)
    assert report["int8_size_bytes"] == report["int8_footprint"]["deployment_size_bytes"]
    assert report["fp32_size_bytes"] == report["fp32_footprint"]["deployment_size_bytes"]
    assert report["source_graph"]["quantized_target_operators"]["Conv"] > 0
    assert report["int8_graph"]["int8_initializers"] > 0
    assert report["int8_graph"]["qdq_wrapped_target_operators"] == report["source_graph"]["quantized_target_operators"]
    image, _ = reader1.dataset[0]
    outputs = run_all_exits(load_session(int8), image.unsqueeze(0))
    assert [tuple(value.shape) for value in outputs] == [(1, 5)] * 3
    monkeypatch.setattr(sys, "argv", ["benchmark_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--samples", "2",
                                       "--warmup", "1", "--repetitions", "2",
                                       "--intra-op-threads", "1"])
    benchmark_main()
    latency = json.loads((deployment / "onnx_latency.json").read_text())
    assert latency["execution_provider"] == "CPUExecutionProvider"
    assert latency["providers_by_model"] == {"fp32": ["CPUExecutionProvider"],
                                               "int8": ["CPUExecutionProvider"]}
    assert latency["batch_size"] == 1 and latency["sample_indices_in_validation_manifest"] == [0, 1]
    assert latency["warmup_count_per_model"] == 1 and latency["repetitions_per_model"] == 2
    assert "not conditional early-exit latency" in latency["runtime_scope"]
    for result in latency["results"].values():
        assert result["measured_calls"] == 2
        assert result["mean_ms"] >= 0 and result["p50_ms"] >= 0 and result["p95_ms"] >= 0
    calibration = {"schema_version": 1, "fit_split": "validation", "threshold_selection_split": "validation",
                   "checkpoint_sha256": file_sha256(checkpoint), "split_hashes": split_hashes({"split_dir": split_dir}),
                   "policies": {"calibrated": {"policy": RoutingPolicy(-1, -1).to_dict()}}}
    run_dir = tmp_path / "results/synthetic"
    (run_dir / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    flops = {"method": "synthetic_path_costs", "checkpoint_sha256": file_sha256(checkpoint),
             "image_size": 32, "exit1": 1.0, "exit2": 2.0, "exit3": 3.0}
    (run_dir / "flops.json").write_text(json.dumps(flops), encoding="utf-8")
    points = {"selection_split": "validation", "checkpoint_sha256": file_sha256(checkpoint),
              "calibration_sha256": file_sha256(run_dir / "calibration.json"),
              "flops_sha256": file_sha256(run_dir / "flops.json"),
              "split_hashes": split_hashes({"split_dir": split_dir}), "image_size": 32,
              "requested_points": 1, "points": [{"id": "adaptive_op1", "label": "synthetic",
                                                  **RoutingPolicy(-1, -1).to_dict()}]}
    (run_dir / "operating_points.json").write_text(json.dumps(points), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluate_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--compare-int8", "--split", "test"])
    with pytest.raises(ValueError, match="validation comparison"):
        evaluate_main()
    monkeypatch.setattr(sys, "argv", ["evaluate_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--compare-int8", "--split", "validation"])
    evaluate_main()
    fp32_metrics = json.loads((deployment / "fp32_validation_metrics.json").read_text())
    int8_metrics = json.loads((deployment / "int8_validation_metrics.json").read_text())
    comparison = json.loads((deployment / "quantization_comparison.json").read_text())
    assert fp32_metrics["operating_points"][0]["policy"] == int8_metrics["operating_points"][0]["policy"]
    assert comparison["compression_ratio"] == pytest.approx(report["compression_ratio_fp32_over_int8"])
    assert fp32_metrics["deployment_size_bytes"] == report["fp32_size_bytes"]
    assert int8_metrics["deployment_size_bytes"] == report["int8_size_bytes"]
    assert comparison["fp32_footprint"] == report["fp32_footprint"]
    assert comparison["int8_footprint"] == report["int8_footprint"]
    assert len(comparison["operating_points"]) == 1
    assert comparison["operating_points"][0]["routing_switch_rate"] == 0
    assert comparison["operating_points"][0]["exit_fraction_delta"] == [0, 0, 0]
    assert all((deployment / name).is_file() for name in (
        "model_size_vs_auroc_validation.csv", "variant_vs_size_validation.csv",
        "routing_distribution_validation.csv"))
    monkeypatch.setattr(sys, "argv", ["evaluate_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--compare-int8", "--split", "test"])
    with pytest.raises(ValueError, match="--validation-reviewed"):
        evaluate_main()
    monkeypatch.setattr(sys, "argv", ["quantize_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--calibration-samples", "5",
                                       "--output-dir", str(tmp_path / "invalid")])
    with pytest.raises(ValueError, match="training split has"):
        quantize_main()
    export_report_path = deployment / "fp32_export_report.json"
    failed_export = json.loads(export_report_path.read_text())
    failed_export["validation_parity"]["passed"] = False
    export_report_path.write_text(json.dumps(failed_export), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["quantize_onnx.py", "--config", str(config_path),
                                       "--checkpoint", str(checkpoint), "--calibration-samples", "2",
                                       "--output-dir", str(tmp_path / "failed_parity")])
    with pytest.raises(ValueError, match="pass validation parity"):
        quantize_main()
    capsys.readouterr()


def test_comparison_deltas_and_routing_transitions():
    fp32 = {"exit_metrics": {"exit3_logits": {"macro_auroc": 0.8}},
            "operating_points": [{"id": "op1", "metrics": {"macro_auroc": 0.75, "macro_f1": 0.6,
                                                      "binary_bce": 0.4, "binary_ece": 0.1},
                                  "exit_fractions": [0.5, 0.5, 0.0], "mean_executed_flops": 1.5}]}
    int8 = {"exit_metrics": {"exit3_logits": {"macro_auroc": 0.78}},
            "operating_points": [{"id": "op1", "metrics": {"macro_auroc": 0.70, "macro_f1": 0.5,
                                                      "binary_bce": 0.45, "binary_ece": 0.12},
                                  "exit_fractions": [0.0, 0.5, 0.5], "mean_executed_flops": 2.5}]}
    result = compare_variants(fp32, int8, {"op1": torch.tensor([1, 2])},
                              {"op1": torch.tensor([2, 3])}, 400, 100)
    assert result["compression_ratio"] == 4
    assert result["size_reduction_percent"] == 75
    assert result["exit3_macro_auroc_delta"] == pytest.approx(-0.02)
    point = result["operating_points"][0]
    assert point["macro_auroc_delta"] == pytest.approx(-0.05)
    assert point["routing_switch_rate"] == 1
    assert point["exit_fraction_delta"] == pytest.approx([-0.5, 0, 0.5])
    assert point["routing_transition_counts_fp32_rows_int8_columns"] == [[0, 1, 0], [0, 0, 1], [0, 0, 0]]


def test_paired_latency_uses_identical_inputs_and_alternates_model_order():
    calls = []

    class Session:
        def __init__(self, name):
            self.name = name

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, names, feed):
            calls.append((self.name, int(feed["image"][0, 0, 0, 0])))
            return [np.zeros((1, 5), dtype=np.float32)] * 3

    sessions = {name: Session(name) for name in ("fp32", "int8")}
    images = [np.full((1, 3, 32, 32), value, dtype=np.float32) for value in (1, 2)]
    result = benchmark_pair(sessions, images, warmup=1, repetitions=2)
    assert calls == [("fp32", 1), ("int8", 1), ("fp32", 1), ("int8", 1),
                     ("int8", 2), ("fp32", 2)]
    assert result["fp32"]["measured_calls"] == result["int8"]["measured_calls"] == 2
    class WrongProvider(Session):
        def get_providers(self):
            return ["CUDAExecutionProvider"]

    with pytest.raises(ValueError, match="CPUExecutionProvider"):
        benchmark_pair({"fp32": Session("fp32"), "int8": WrongProvider("int8")}, images, 1, 2)
