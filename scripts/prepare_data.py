"""Create deterministic patient-separated split manifests from local images."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.chestxray import LABELS, available_records, load_splits, save_splits, split_patients, training_class_stats
from src.utils.config import load_config, resolve_project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    for name in ("dataset_root", "metadata_csv", "split_dir"):
        value = getattr(args, name)
        if value is not None:
            config[name] = resolve_project_path(value)
    records = available_records(config["metadata_csv"], config["dataset_root"], config["max_images"])
    splits = split_patients(records, config["seed"])
    save_splits(splits, config["split_dir"], args.overwrite)
    loaded = load_splits(config["split_dir"], config["dataset_root"])
    print(json.dumps({
        "images": {name: len(rows) for name, rows in loaded.items()},
        "patients": {name: len({row['patient_id'] for row in rows}) for name, rows in loaded.items()},
        "train_class_stats": training_class_stats(loaded["train"]),
        "labels": LABELS,
    }, indent=2))


if __name__ == "__main__":
    main()
