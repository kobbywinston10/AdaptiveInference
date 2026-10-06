#!/usr/bin/env bash
#SBATCH --job-name=adaptive-dev
#SBATCH --nodes=1
#SBATCH --constraint=A4000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

python scripts/check_environment.py --config configs/dev.yaml --device cuda --check-checkpoint
python scripts/train_adaptive.py \
    --config configs/dev.yaml \
    --device cuda \
    --baseline-checkpoint checkpoints/baseline_dev/best.pt \
    --kd-weight "${KD_WEIGHT:?Set KD_WEIGHT to 0 or 1}" \
    --run-name "${RUN_NAME:?Set RUN_NAME}"
