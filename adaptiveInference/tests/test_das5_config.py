"""DAS-5 path mapping is checked without cluster access or training."""

import sys

import pytest
import yaml

from scripts.prepare_das5_config import main, scratch_config
from src.utils.config import PROJECT_ROOT


def test_das5_config_maps_all_artifacts_to_scratch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prepare_das5_config.py", "--source", "configs/tiny.yaml",
                                       "--scratch-root", str(tmp_path)])
    main()
    output = tmp_path / "configs/tiny_das5.yaml"
    assert output.is_file()
    config = yaml.safe_load(output.read_text())
    for key in ("dataset_root", "metadata_csv", "pretrained_checkpoint", "split_dir",
                "checkpoint_root", "results_root"):
        assert str(config[key]).startswith(str(tmp_path))
    assert config["image_size"] == 64
    main()  # Identical generated config is safe to reuse.
    capsys.readouterr()


def test_das5_config_rejects_absolute_source_paths(tmp_path):
    source = yaml.safe_load((PROJECT_ROOT / "configs/tiny.yaml").read_text())
    source["dataset_root"] = str(tmp_path)
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="repository-relative"):
        scratch_config(path, tmp_path / "scratch")
