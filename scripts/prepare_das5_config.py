"""Write a DAS-5 scratch-path copy of an existing relative-path config."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.utils.config import load_config, resolve_project_path
from src.utils.run import atomic_write_text


SCRATCH_KEYS = ("dataset_root", "metadata_csv", "pretrained_checkpoint", "split_dir", "checkpoint_root", "results_root")


def scratch_config(source: Path, scratch_root: Path) -> dict:
    load_config(source)  # Validate the source before copying it.
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    for key in SCRATCH_KEYS:
        relative = Path(config[key])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{key} must be a safe repository-relative path in the source config")
        config[key] = str(scratch_root / relative)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="configs/tiny.yaml")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    username = os.environ.get("USER")
    if args.scratch_root is None and not username:
        raise ValueError("Set USER or pass --scratch-root")
    scratch_root = (args.scratch_root or Path("/var/scratch") / username / "adaptiveInference").expanduser().resolve()
    source = resolve_project_path(args.source)
    output = (args.output or scratch_root / "configs" / f"{source.stem}_das5.yaml").expanduser().resolve()
    config = scratch_config(source, scratch_root)
    contents = yaml.safe_dump(config, sort_keys=False)
    if output.exists() and output.read_text(encoding="utf-8") != contents and not args.overwrite:
        raise FileExistsError(f"DAS-5 config differs: {output}; inspect it or pass --overwrite")
    if not output.exists() or args.overwrite:
        atomic_write_text(output, contents)
    print(json.dumps({"config": str(output), "scratch_root": str(scratch_root),
                      "dataset_root": config["dataset_root"], "checkpoint_root": config["checkpoint_root"],
                      "results_root": config["results_root"]}, indent=2))


if __name__ == "__main__":
    main()
