#!/usr/bin/env bash
#SBATCH --job-name=baseline-dev
#SBATCH --nodes=1
#SBATCH --constraint=A4000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

python scripts/check_environment.py --config configs/dev.yaml --device cuda --check-checkpoint
if [[ "${RESUME:-0}" == "1" ]]; then
    python scripts/train_baseline.py --config configs/dev.yaml --device cuda --resume
else
    python scripts/train_baseline.py --config configs/dev.yaml --device cuda
fi
