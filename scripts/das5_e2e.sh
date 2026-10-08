#!/usr/bin/env bash
# Run inside a DAS-5 TitanRTX srun allocation; one-epoch local pipeline check.
set -euo pipefail

mode="${1:---check}"
if [[ "$mode" != "--check" && "$mode" != "--run" ]]; then
    echo "Usage: bash scripts/das5_e2e.sh [--check|--run]" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this script inside a DAS-5 srun GPU allocation." >&2
    exit 2
fi

cd "$(dirname "$0")/.."
source /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
source "/var/scratch/$USER/miniforge3/etc/profile.d/conda.sh"
conda activate "/var/scratch/$USER/conda-envs/adaptive-inference"

mkdir -p "/var/scratch/$USER/python-cache" "/var/scratch/$USER/tmp"
export PYTHONPYCACHEPREFIX="/var/scratch/$USER/python-cache"
export TMPDIR="/var/scratch/$USER/tmp"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

scratch_config="/var/scratch/$USER/adaptiveInference/configs/tiny_das5.yaml"
python scripts/prepare_das5_config.py --source configs/tiny.yaml
python scripts/check_environment.py --config "$scratch_config" --device cuda --check-checkpoint

if [[ "$mode" == "--run" ]]; then
    python scripts/run_local_e2e.py --config "$scratch_config" --device cuda --run \
        --run-prefix "${RUN_PREFIX:-das5_e2e}" --scenario "${SCENARIO:-with-kd}"
else
    python scripts/run_local_e2e.py --config "$scratch_config" --device cuda \
        --run-prefix "${RUN_PREFIX:-das5_e2e}" --scenario "${SCENARIO:-with-kd}"
fi
