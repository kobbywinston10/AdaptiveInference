"""NIH ChestX-ray14 metadata, patient splits, and five-label image loading."""

import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


LABELS = ("Atelectasis", "Effusion", "Mass", "Nodule", "Pneumothorax")
SPLITS = ("train", "validation", "test")
FIELDS = ("image", "patient_id", "finding_labels")


def parse_labels(raw: str) -> tuple[int, ...]:
    found = set(raw.split("|"))
    return tuple(int(label in found) for label in LABELS)


def available_records(metadata_csv: Path, dataset_root: Path, max_images: int | None = None) -> list[dict]:
    """Index only local images; paths in records stay relative to dataset root."""
    images = {}
    for path in dataset_root.rglob("*.png"):
        if path.name in images:
            raise ValueError(f"Duplicate image filename: {path.name}")
        images[path.name] = path.relative_to(dataset_root).as_posix()
    records = []
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image = images.get(row["Image Index"])
            if image is not None:
                records.append({"image": image, "patient_id": row["Patient ID"], "finding_labels": row["Finding Labels"]})
                if max_images is not None and len(records) >= max_images:
                    break
    if not records:
        raise ValueError(f"No metadata rows with local images under {dataset_root}")
    return records


def split_patients(records: list[dict], seed: int) -> dict[str, list[dict]]:
    patients = sorted({row["patient_id"] for row in records})
    if len(patients) < 3:
        raise ValueError("At least three patients are required")
    random.Random(seed).shuffle(patients)
    n_train = int(0.70 * len(patients))
    n_val = int(0.15 * len(patients))
    if min(n_train, n_val, len(patients) - n_train - n_val) < 1:
        raise ValueError("Too few patients for nonempty 70/15/15 splits")
    memberships = {
        "train": set(patients[:n_train]),
        "validation": set(patients[n_train:n_train + n_val]),
        "test": set(patients[n_train + n_val:]),
    }
    result = {split: [row for row in records if row["patient_id"] in ids] for split, ids in memberships.items()}
    assert_no_patient_leakage(result)
    return result


def assert_no_patient_leakage(splits: dict[str, list[dict]]) -> None:
    ids = {split: {row["patient_id"] for row in splits[split]} for split in SPLITS}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if ids[left] & ids[right]:
                raise ValueError(f"Patient leakage between {left} and {right}")


def save_splits(splits: dict[str, list[dict]], split_dir: Path, overwrite: bool = False) -> None:
    assert_no_patient_leakage(splits)
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = [split_dir / f"{name}.csv" for name in SPLITS]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("Split files already exist; use --overwrite to replace them")
    for name, path in zip(SPLITS, paths):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(splits[name])


def load_splits(split_dir: Path, dataset_root: Path) -> dict[str, list[dict]]:
    splits = {}
    for name in SPLITS:
        with (split_dir / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty {name} split")
        for row in rows:
            image = (dataset_root / row["image"]).resolve()
            if not image.is_relative_to(dataset_root.resolve()) or not image.is_file():
                raise FileNotFoundError(f"Missing or unsafe image path: {row['image']}")
        splits[name] = rows
    assert_no_patient_leakage(splits)
    all_images = [row["image"] for rows in splits.values() for row in rows]
    if len(all_images) != len(set(all_images)):
        raise ValueError("Image appears in multiple splits")
    return splits


def training_class_stats(rows: list[dict]) -> dict[str, dict[str, float]]:
    positives = Counter()
    for row in rows:
        positives.update(label for label, value in zip(LABELS, parse_labels(row["finding_labels"])) if value)
    return {
        label: {
            "positive": positives[label],
            "negative": len(rows) - positives[label],
            "prevalence": positives[label] / len(rows),
            "pos_weight": (len(rows) - positives[label]) / positives[label] if positives[label] else float("inf"),
        }
        for label in LABELS
    }


def positive_weights(train_rows: list[dict]) -> torch.Tensor:
    stats = training_class_stats(train_rows)
    absent = [label for label in LABELS if not stats[label]["positive"]]
    if absent:
        raise ValueError(f"No positive training examples for {absent}; cannot compute pos_weight")
    return torch.tensor([stats[label]["pos_weight"] for label in LABELS], dtype=torch.float32)


class ChestXrayDataset(Dataset):
    """Deterministic grayscale -> RGB -> square resize -> [-1, 1] tensor."""

    def __init__(self, rows: list[dict], dataset_root: Path, image_size: int):
        self.rows = rows
        self.dataset_root = dataset_root
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.dataset_root / row["image"]) as image:
            image = image.convert("L").convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy()
        x = torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0
        y = torch.tensor(parse_labels(row["finding_labels"]), dtype=torch.float32)
        return x, y


def make_loaders(splits: dict[str, list[dict]], dataset_root: Path, image_size: int, batch_size: int, seed: int, num_workers: int = 0, pin_memory: bool = False) -> dict[str, DataLoader]:
    assert_no_patient_leakage(splits)
    return {
        name: DataLoader(
            ChestXrayDataset(splits[name], dataset_root, image_size),
            batch_size=batch_size,
            shuffle=name == "train",
            generator=torch.Generator().manual_seed(seed),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for name in SPLITS
    }
