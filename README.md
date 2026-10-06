# Adaptive chest X-ray inference

The project uses five independent NIH ChestX-ray14 labels and a
RadImageNet-pretrained ResNet50. `configs/tiny.yaml` is for smoke tests,
`configs/dev.yaml` uses the local dev subset, and `configs/full.yaml` is for
the later full DAS-6 dataset. Paths in config files are relative to this
repository unless absolute paths are provided.

## DAS-6 setup

Run commands from the `adaptiveInference` directory inside an allocated GPU
session. The cluster's GPU allocation command and CUDA module depend on its
configuration. Install a CUDA-enabled PyTorch build compatible with the
allocated GPU, then the remaining dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA PyTorch/GPU allocation unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
python scripts/check_environment.py --config configs/dev.yaml --device cuda --check-checkpoint
```

Select a matching PyTorch CUDA wheel using the [official installer](https://pytorch.org/get-started/locally/) if the environment's wheel is CPU-only.
`weights/ResNet50.pt`, the dev images, metadata, and split manifests must be
present on DAS-6. To create manifests there when absent:

```bash
python scripts/prepare_data.py --config configs/dev.yaml
```

The script refuses to replace existing manifests unless `--overwrite` is
given. For a different mount, edit the paths in the YAML or pass
`--dataset-root`, `--metadata-csv`, and `--split-dir` to `prepare_data.py`;
the training script accepts these path overrides too. The `full.yaml`
paths expect `ChestXray14/full/Data_Entry_2017_v2020.csv` and local PNGs
beneath `ChestXray14/full/`. Prepare its splits separately before a full run.

## Phase 3 baseline

Start the dev baseline on an allocated DAS-6 GPU:

```bash
python scripts/train_baseline.py --config configs/dev.yaml --device cuda
```

For a scheduled DAS-6 job, submit from this repository directory after
loading its Slurm tooling. The included job script requests one A4000 GPU:

```bash
module load prun
sbatch --time=01:00:00 scripts/das6_baseline.sbatch
```

Resume from the last completed epoch, with the same config and split files:

```bash
python scripts/train_baseline.py --config configs/dev.yaml --device cuda --resume
```

The equivalent scheduled resume is:

```bash
module load prun
sbatch --time=01:00:00 --export=ALL,RESUME=1 scripts/das6_baseline.sbatch
```

The one-hour walltime is a job request, not an expected training duration.
[DAS-6's published policy](https://www.cs.vu.nl/das/jobs.shtml) limits daytime
jobs to 15 minutes; schedule longer training at night or on weekends, or
follow the site's approval process. The [GPU guide](https://www.cs.vu.nl/das/gpu.shtml)
documents the A4000 constraint and GPU resource request.

`--epochs N` sets the **total** number of epochs and may be increased on
resume. `--batch-size`, `--num-workers`, `--run-name`, and path overrides are
available for new runs; resume rejects changes to training settings or split
manifests. `--smoke` performs one real-image forward pass without training.

Default dev artifacts are:

| Path | Purpose |
| --- | --- |
| `checkpoints/baseline_dev/best.pt` | Best validation macro AUROC model; Phase 4 input |
| `checkpoints/baseline_dev/last.pt` | Latest model, optimizer, scaler, RNG, and history for resume |
| `results/baseline_dev/config.yaml` | Resolved run configuration |
| `results/baseline_dev/environment.json` | Software, hardware, seed, commit, and run signature |
| `results/baseline_dev/history.csv` | Per-epoch train loss and validation metrics |

The trainer uses train-only `pos_weight` with `BCEWithLogitsLoss`, selects
the best model by validation macro AUROC, and never consults the test split.
Undefined per-label AUROCs are excluded from the macro mean. Accuracy is
exact-match subset accuracy at a fixed 0.5 sigmoid threshold. Mixed
precision is active only on CUDA. A fresh run refuses to replace existing
checkpoints unless `--overwrite` is explicitly supplied.

## Phase 4 auxiliary heads

Once the baseline best checkpoint exists, start two separate runs from the
same saved baseline:

```bash
python scripts/train_adaptive.py --config configs/dev.yaml --device cuda --baseline-checkpoint checkpoints/baseline_dev/best.pt --kd-weight 0 --run-name adaptive_dev_no_kd
python scripts/train_adaptive.py --config configs/dev.yaml --device cuda --baseline-checkpoint checkpoints/baseline_dev/best.pt --kd-weight 1 --run-name adaptive_dev_with_kd
```

The scheduled DAS-6 equivalents are:

```bash
sbatch --time=01:00:00 --export=ALL,KD_WEIGHT=0,RUN_NAME=adaptive_dev_no_kd scripts/das6_adaptive.sbatch
sbatch --time=01:00:00 --export=ALL,KD_WEIGHT=1,RUN_NAME=adaptive_dev_with_kd scripts/das6_adaptive.sbatch
```

Phase 4 freezes the baseline backbone and final teacher, trains only the two
early heads, and records validation BCE for each exit. Its outputs are
`checkpoints/<run_name>/best.pt` and `results/<run_name>/history.json`, plus
resolved config and environment metadata. These commands are prepared but
have not been run on real data. Tests use synthetic fixtures and are not
experimental results: `python -m pytest tests -q`.
