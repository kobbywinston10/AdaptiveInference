# DAS-5 Setup Guide

This section documents the DAS-5 setup used for the **Adaptive Inference for Chest X-ray Classification** project.

The verified setup uses:

- Python 3.10
- PyTorch with CUDA support
- NVIDIA TITAN RTX
- NIH ChestX-ray14 `dev-mini`
- RadImageNet ResNet50 weights
- SLURM `srun`

Large files should live under `/var/scratch`; the repository itself can stay in `/home`.

---

## 1. Connect and update the repository

```bash
ssh <DAS5_USERNAME>@fs2.das5.science.uva.nl
cd ~/adaptiveInference
git status
git pull
```

If the repository is not cloned yet:

```bash
git clone <YOUR_REPOSITORY_URL>
cd adaptiveInference
```

---

## 2. Check storage quotas

Check your personal quota:

```bash
quota -s
```

Check the filesystem for the current directory:

```bash
df -h .
```

On DAS-5, `/home` has a relatively small personal quota, while `/var/scratch` provides much more working space.

Use:

```text
~/adaptiveInference
```

for code and small files, and:

```text
/var/scratch/$USER/adaptiveInference
```

for datasets, weights, checkpoints, results, environments, and temporary files.

Create the scratch directories:

```bash
mkdir -p /var/scratch/$USER/adaptiveInference/ChestXray14/dev-mini
mkdir -p /var/scratch/$USER/adaptiveInference/weights
mkdir -p /var/scratch/$USER/adaptiveInference/checkpoints
mkdir -p /var/scratch/$USER/adaptiveInference/results
```

If `/home` is near its quota, inspect usage:

```bash
du -sh ~/* 2>/dev/null | sort -h
du -sh ~/.??* 2>/dev/null | sort -h
```

A large pip cache can be safely cleared with:

```bash
python3 -m pip cache purge
```

Then check again:

```bash
quota -s
```

---

## 3. Upload the dev dataset and RadImageNet weights

The dataset and pretrained weights are intentionally not stored in Git.

From the local machine, upload the NIH archive:

```bash
scp ChestXray14/dev-mini/images_001.tar.gz \
<DAS5_USERNAME>@fs2.das5.science.uva.nl:/var/scratch/<DAS5_USERNAME>/adaptiveInference/ChestXray14/dev-mini/
```

Upload the metadata:

```bash
scp ChestXray14/dev-mini/Data_Entry_2017_v2020.csv \
<DAS5_USERNAME>@fs2.das5.science.uva.nl:/var/scratch/<DAS5_USERNAME>/adaptiveInference/ChestXray14/dev-mini/
```

Upload the RadImageNet checkpoint:

```bash
scp weights/ResNet50.pt \
<DAS5_USERNAME>@fs2.das5.science.uva.nl:/var/scratch/<DAS5_USERNAME>/adaptiveInference/weights/
```

### Extract the NIH archive

On DAS-5:

```bash
cd /var/scratch/$USER/adaptiveInference/ChestXray14/dev-mini
tar -tzf images_001.tar.gz | head
```

The archive used for this project extracts to:

```text
images_001/images/<IMAGE>.png
```

Extract it:

```bash
mkdir -p images_001
tar -xzf images_001.tar.gz -C images_001
```

Verify:

```bash
find images_001 -type f -name "*.png" | wc -l
```

The current development subset contains:

```text
4999 PNG files
```

After verifying the extraction, the archive can be removed to save space:

```bash
rm images_001.tar.gz
```

---

## 4. Configure DAS-5 paths

`configs/dev.yaml` should point to the scratch copy of the dataset.

Example:

```yaml
dataset_root: /var/scratch/<DAS5_USERNAME>/adaptiveInference/ChestXray14/dev-mini
metadata_csv: /var/scratch/<DAS5_USERNAME>/adaptiveInference/ChestXray14/dev-mini/Data_Entry_2017_v2020.csv
```

The RadImageNet checkpoint should resolve to:

```text
/var/scratch/<DAS5_USERNAME>/adaptiveInference/weights/ResNet50.pt
```

Checkpoints and result files should also be written to scratch:

```text
/var/scratch/<DAS5_USERNAME>/adaptiveInference/checkpoints/
/var/scratch/<DAS5_USERNAME>/adaptiveInference/results/
```

Do not store large training outputs under `/home`.

---

## 5. Install a modern Python environment

The default DAS-5 Python is Python 3.6.8, which is too old for the current PyTorch requirements.

No suitable Python/Conda module was available, so Miniforge was installed under `/var/scratch`.

### Install Miniforge

```bash
cd /var/scratch/$USER

wget -O Miniforge3.sh \
https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash Miniforge3.sh -b -p /var/scratch/$USER/miniforge3
rm Miniforge3.sh
```

Load Conda:

```bash
source /var/scratch/$USER/miniforge3/etc/profile.d/conda.sh
```

Create a Python 3.10 environment:

```bash
mkdir -p /var/scratch/$USER/conda-envs

conda create -p /var/scratch/$USER/conda-envs/adaptive-inference \
python=3.10 pip -y
```

Activate it:

```bash
conda activate /var/scratch/$USER/conda-envs/adaptive-inference
```

Verify:

```bash
python --version
which python
pip --version
```

The tested setup used Python 3.10.

---

## 6. Install project dependencies

```bash
cd ~/adaptiveInference

source /var/scratch/$USER/miniforge3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/conda-envs/adaptive-inference

pip install -r requirements.txt
```

The verified PyTorch environment reported:

```text
PyTorch: 2.14.1+cu130
CUDA build: 13.0
```

---

## 7. Keep caches and temporary files off `/home`

Python and pytest may write generated files into the repository or home directory.

Create scratch locations:

```bash
mkdir -p /var/scratch/$USER/python-cache
mkdir -p /var/scratch/$USER/tmp
```

Export:

```bash
export PYTHONPYCACHEPREFIX=/var/scratch/$USER/python-cache
export TMPDIR=/var/scratch/$USER/tmp
export PYTHONPATH="$HOME/adaptiveInference${PYTHONPATH:+:$PYTHONPATH}"
```

These first two exports can optionally be added to `~/.bashrc`:

```bash
echo 'export PYTHONPYCACHEPREFIX=/var/scratch/$USER/python-cache' >> ~/.bashrc
echo 'export TMPDIR=/var/scratch/$USER/tmp' >> ~/.bashrc
```

---

## 8. Check GPU availability

Inspect DAS-5 node features and GPU resources:

```bash
sinfo -o "%40N %25f %25G %10T"
```

The tested setup used an available `TitanRTX` node.

Verify that SLURM can allocate one:

```bash
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:05:00 hostname
```

For this setup, `srun` was used directly instead of `prun`.

---

## 9. Load CUDA inside the GPU job

CUDA tools are not automatically on the compute-node `PATH`.

Inside the job, load:

```bash
source /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
```

Test the GPU:

```bash
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:05:00 \
bash -lc '
source /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
hostname
nvidia-smi
'
```

---

## 10. Verify PyTorch CUDA access

```bash
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:05:00 \
bash -lc '
source /etc/profile.d/lmod.sh
module load cuda12.3/toolkit

source /var/scratch/$USER/miniforge3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/conda-envs/adaptive-inference

cd $HOME/adaptiveInference

python -c "
import os
import torch

print(\"PyTorch:\", torch.__version__)
print(\"CUDA build:\", torch.version.cuda)
print(\"CUDA available:\", torch.cuda.is_available())
print(\"CUDA_VISIBLE_DEVICES:\", os.environ.get(\"CUDA_VISIBLE_DEVICES\"))
if torch.cuda.is_available():
    print(\"GPU:\", torch.cuda.get_device_name(0))
"
'
```

The verified output included:

```text
CUDA available: True
CUDA_VISIBLE_DEVICES: 0
GPU: NVIDIA TITAN RTX
```

---

## 11. Run the test suite

```bash
cd ~/adaptiveInference

source /var/scratch/$USER/miniforge3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/conda-envs/adaptive-inference

export PYTHONPYCACHEPREFIX=/var/scratch/$USER/python-cache
export TMPDIR=/var/scratch/$USER/tmp
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -p no:cacheprovider -q
```

All implemented Phase 1-4 tests passed in the verified setup.

---

## 12. Run the project environment checker

```bash
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:05:00 \
bash -lc '
source /etc/profile.d/lmod.sh
module load cuda12.3/toolkit

source /var/scratch/$USER/miniforge3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/conda-envs/adaptive-inference

export PYTHONPYCACHEPREFIX=/var/scratch/$USER/python-cache
export TMPDIR=/var/scratch/$USER/tmp
export PYTHONPATH=$HOME/adaptiveInference

cd $HOME/adaptiveInference
python scripts/check_environment.py --config configs/dev.yaml
'
```

The verified environment check reported:

```text
Python 3.10
CUDA available: true
selected device: cuda
mixed precision: true
metadata found
RadImageNet checkpoint found
4999 local PNG files
```

---

## 13. Run the first end-to-end GPU smoke test

Before a larger development run, use `configs/tiny.yaml`.

```bash
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:15:00 \
bash -lc '
source /etc/profile.d/lmod.sh
module load cuda12.3/toolkit

source /var/scratch/$USER/miniforge3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/conda-envs/adaptive-inference

export PYTHONPYCACHEPREFIX=/var/scratch/$USER/python-cache
export TMPDIR=/var/scratch/$USER/tmp
export PYTHONPATH=$HOME/adaptiveInference

cd $HOME/adaptiveInference
python scripts/train_baseline.py --config configs/tiny.yaml
'
```

This verifies:

```text
ChestX-ray images
    ↓
metadata
    ↓
patient-level split
    ↓
DataLoader
    ↓
RadImageNet ResNet50
    ↓
5-label classifier
    ↓
GPU forward/backward
    ↓
BCE loss
    ↓
validation
    ↓
checkpoint saving
```

Tiny-mode metrics are only for pipeline validation. They must not be treated as meaningful experimental results.

The verified tiny run completed successfully.

---

## 14. Returning to the project later

At the beginning of a new DAS-5 session:

```bash
cd ~/adaptiveInference

source /var/scratch/$USER/miniforge3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/conda-envs/adaptive-inference

export PYTHONPYCACHEPREFIX=/var/scratch/$USER/python-cache
export TMPDIR=/var/scratch/$USER/tmp
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

For a GPU command, use the following template:

```bash
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:15:00 \
bash -lc '
source /etc/profile.d/lmod.sh
module load cuda12.3/toolkit

source /var/scratch/$USER/miniforge3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/conda-envs/adaptive-inference

export PYTHONPYCACHEPREFIX=/var/scratch/$USER/python-cache
export TMPDIR=/var/scratch/$USER/tmp
export PYTHONPATH=$HOME/adaptiveInference

cd $HOME/adaptiveInference

# Replace with the command to run.
python scripts/train_baseline.py --config configs/tiny.yaml
'
```

---

## Troubleshooting

### `No matching distribution found for torch>=2.3`

Check:

```bash
python --version
```

If it reports Python 3.6, use the Miniforge Python 3.10 environment described above.

### `Disk quota exceeded`

Check:

```bash
quota -s
```

Inspect cache usage:

```bash
du -h --max-depth=1 ~/.cache | sort -h
```

Clear pip cache:

```bash
python -m pip cache purge
```

Keep large/generated files under `/var/scratch`.

### `ModuleNotFoundError: No module named 'scripts'`

From the repository root:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Then run tests with:

```bash
python -m pytest -p no:cacheprovider -q
```

### `nvidia-smi: command not found`

Load CUDA inside the job:

```bash
source /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
```

### `prun ... Invalid Trackable RESource (TRES) specification`

Use `srun` directly:

```bash
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:05:00 hostname
```

### `Missing or unsafe image path`

For the current development archive, expected relative paths look like:

```text
images_001/images/00000002_000.png
```

The corresponding real path should therefore be:

```text
<dataset_root>/images_001/images/00000002_000.png
```

Check with:

```bash
find /var/scratch/$USER/adaptiveInference/ChestXray14/dev-mini \
-type f -name '00000002_000.png' -print
```

---

## Current verified state

## Phase 1-8 pipeline smoke test

The earlier tiny baseline check covers only part of the pipeline. After the
scratch Conda environment, dev images, metadata, and weights are in place,
run the complete one-epoch validation pipeline from `~/adaptiveInference`:

```bash
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:05:00 bash scripts/das5_e2e.sh --check
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:30:00 bash scripts/das5_e2e.sh --run
```

`--check` loads CUDA and Conda, verifies the scratch paths and checkpoint,
and previews the run without training. `--run` adds one baseline epoch, one
adaptive-head epoch with KD, validation calibration, budget and degradation
checks, FLOPs, latency, threshold sweep, and Pareto plots. It never evaluates
the test split. Set `SCENARIO=both` to run the no-KD variant too. For another
run, set `RUN_PREFIX` to a fresh value; existing output directories are not
replaced.

The wrapper generates
`/var/scratch/$USER/adaptiveInference/configs/tiny_das5.yaml` and writes
checkpoints and results beneath the same scratch root. These smoke-test
artifacts do not establish final model or latency results.

## Current verified state

The following have been verified on DAS-5:

- Python 3.10 environment under `/var/scratch`
- project dependencies installed
- NVIDIA TITAN RTX allocation using SLURM
- CUDA available to PyTorch
- Phase 1-4 tests pass
- `dev.yaml` environment check passes
- 4,999 NIH development PNGs detected
- RadImageNet checkpoint detected
- CUDA mixed-precision training enabled
- tiny baseline training completes successfully

Before larger training runs, ensure checkpoint and result output paths point to `/var/scratch`, not `/home`.
