# Adaptive chest X-ray inference

The project uses five independent NIH ChestX-ray14 labels and a
RadImageNet-pretrained ResNet50. `configs/tiny.yaml` is for smoke tests,
`configs/dev.yaml` uses the local dev subset, and `configs/full.yaml` is for
the later full DAS-5 dataset. Paths in config files are relative to this
repository unless absolute paths are provided.
The target cluster is **DAS-5**; the Windows commands below are optional
local checks.

## Full local pipeline smoke test (Phases 1-8)

From the repository directory in **PowerShell** on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/check_environment.py --config configs/tiny.yaml --device cpu --check-checkpoint
python scripts/run_local_e2e.py --config configs/tiny.yaml --device cpu
python scripts/run_local_e2e.py --config configs/tiny.yaml --device cpu --run
```

The penultimate command is a read-only preflight. The final command runs one
epoch of baseline and one epoch of auxiliary-head training on the configured
tiny real-image subset, then validation-only static evaluation, calibration,
budget evaluation, degradation, FLOP counting, batch-one latency, threshold
sweep, and Pareto plots. It prepares patient-level splits if all three split
manifests are absent. It refuses existing output directories; use a new
`--run-prefix` for another run. The test split is not evaluated.

Expected checkpoints are `checkpoints/local_e2e_baseline/best.pt` and
`checkpoints/local_e2e_with_kd/best.pt`. The matching `results/` directories
contain histories and evaluation artifacts. The adaptive results directory
contains `calibration.json`, `flops.{json,csv}`, `latency.{json,csv}`,
`pareto_points.csv`, `pareto_summary.json`, and three plots under `plots/`.
The baseline results directory contains `static_validation.json` and
`local_e2e_summary.json`. Use `--scenario both` to exercise both the no-KD
and KD auxiliary-head variants; `--scenario no-kd` runs only the former.
`--latency-samples`, `--warmup`, `--measurements`, and `--quantile-steps`
control smoke-test duration. On a CUDA-equipped machine, select `--device
cuda`. These one-epoch outputs check that the pipeline works; they are not
final training or benchmark results. The runner never evaluates the test
split.

## DAS-5 setup and full local pipeline smoke test

Follow [the DAS-5 setup guide](docs/DAS5_SETUP%20(1).md) to install the
Python 3.10 Conda environment under `/var/scratch/$USER`, upload the dev
images/metadata and RadImageNet weights to scratch, and verify a TitanRTX
allocation. The wrapper expects Miniforge at
`/var/scratch/$USER/miniforge3`, the environment at
`/var/scratch/$USER/conda-envs/adaptive-inference`, dev images and metadata
under `/var/scratch/$USER/adaptiveInference/ChestXray14/dev-mini`, and
`ResNet50.pt` under `/var/scratch/$USER/adaptiveInference/weights`. Install
project dependencies into that environment before the job:

```bash
source /var/scratch/$USER/miniforge3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/conda-envs/adaptive-inference
python -m pip install -r requirements.txt
```

From `~/adaptiveInference` on the login node, launch the
read-only preflight inside a short GPU allocation:

```bash
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:05:00 bash scripts/das5_e2e.sh --check
```

When the preflight passes, run the one-epoch Phase 1-8 pipeline check:

```bash
srun -N 1 -n 1 -C TitanRTX --gres=gpu:1 --time=00:30:00 bash scripts/das5_e2e.sh --run
```

The wrapper loads the documented CUDA module and scratch Conda environment.
It creates `/var/scratch/$USER/adaptiveInference/configs/tiny_das5.yaml`
from `configs/tiny.yaml`; dataset, weight, split, checkpoint, and result paths
all point to scratch. The runner prepares splits if absent and refuses to
replace previous runs. Outputs live under
`/var/scratch/$USER/adaptiveInference/checkpoints/das5_e2e_*` and
`/var/scratch/$USER/adaptiveInference/results/das5_e2e_*`.
Set `SCENARIO=both` before `srun` to check both KD variants, or set
`RUN_PREFIX` to choose fresh output names. These are short validation-only
pipeline checks, not final experimental runs. The requested walltime is an
allocation limit, not an expected training duration.

For the later dev/full experiments, generate a scratch-path config from the
corresponding template with `scripts/prepare_das5_config.py --source
configs/dev.yaml` (or `configs/full.yaml`). Verify the files exist at the
generated paths before training. The full dataset run is Phase 9.

## Phase 3 baseline

Start the dev baseline inside an allocated DAS-5 GPU session after generating
the scratch-path dev config:

```bash
python scripts/train_baseline.py --config /var/scratch/$USER/adaptiveInference/configs/dev_das5.yaml --device cuda
```

Resume from the last completed epoch, with the same config and split files:

```bash
python scripts/train_baseline.py --config /var/scratch/$USER/adaptiveInference/configs/dev_das5.yaml --device cuda --resume
```

`--epochs N` sets the **total** number of epochs and may be increased on
resume. `--batch-size`, `--num-workers`, `--run-name`, and path overrides are
available for new runs; resume rejects changes to training settings or split
manifests. `--smoke` performs one real-image forward pass without training.

Default dev artifacts under the generated DAS-5 scratch root are:

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
python scripts/train_adaptive.py --config /var/scratch/$USER/adaptiveInference/configs/dev_das5.yaml --device cuda --baseline-checkpoint /var/scratch/$USER/adaptiveInference/checkpoints/baseline_dev/best.pt --kd-weight 0 --run-name adaptive_dev_no_kd
python scripts/train_adaptive.py --config /var/scratch/$USER/adaptiveInference/configs/dev_das5.yaml --device cuda --baseline-checkpoint /var/scratch/$USER/adaptiveInference/checkpoints/baseline_dev/best.pt --kd-weight 1 --run-name adaptive_dev_with_kd
```

Phase 4 freezes the baseline backbone and final teacher, trains only the two
early heads, and records validation BCE for each exit. Its outputs are
`checkpoints/<run_name>/best.pt` and `results/<run_name>/history.json`, plus
resolved config and environment metadata. These commands are prepared but
have not been run on real data. Tests use synthetic fixtures and are not
experimental results: `python -m pytest tests -q`.

## Phase 5: validation calibration and routing

After a trained Phase 4 checkpoint exists, fit one positive scalar
temperature per exit and select Exit 1/2 thresholds from **validation only**:

```bash
python scripts/calibrate.py --config configs/dev.yaml --checkpoint checkpoints/adaptive_dev_with_kd/best.pt --device cuda
```

The output is `results/adaptive_dev_with_kd/calibration.json`. It includes
both calibrated and uncalibrated policies, per-exit BCE and binary ECE before
and after temperature fitting, and a threshold sweep. The primary uncertainty
is the **maximum binary entropy** over five sigmoid probabilities; use
`--aggregation mean` for the ablation. Threshold selection chooses the
shallowest validation policy whose multi-label BCE is no more than 0.02 above
the final exit's BCE by default (`--max-bce-increase` changes this). No test
labels enter calibration or threshold selection. `load_policy` checks the
saved model's SHA-256 before routing. The ECE definition uses equal-width
bins over all image-label Bernoulli predictions.

`src.routing.policy.route_one` applies either saved policy through real
conditional inference, skipping later layers after an early decision.

## Phase 6: resource budgets

`LOW`, `MEDIUM`, and `HIGH` cap actual depth at Exit 1, Exit 2, and Final
Exit. Evaluate the exact uncertainty-preferred and budget-constrained exits
for a saved policy with:

```bash
python scripts/evaluate_adaptive.py --config configs/dev.yaml --checkpoint checkpoints/adaptive_dev_with_kd/best.pt --mode calibrated --split validation --device cuda
```

The default report, `results/adaptive_dev_with_kd/budget_validation_calibrated.json`,
contains per-image preferred/actual exits, forced-exit flags, forced-exit
rates, and predictive metrics for all three budgets. `--mode uncalibrated`
uses the other validation-selected policy. `--split test` is for final
evaluation only.

This report computes all three exits once to recover the **exact** preferred
exit, so its runtime is not budgeted latency. For actual budgeted inference,
`src.routing.budget.route_one_budgeted` stops at the permitted exit. If a
budget forces a stop, deeper uncertainty has not been computed and its exact
preferred exit is returned as unknown (`None`); the forced flag remains
known. No real-data calibration or budget results have been generated yet.

## Phase 7: evaluation-only image degradation

After Phase 4 training and Phase 5 calibration, run blur and noise on the
same held-out split with all three resource budgets:

```bash
python scripts/run_degradation_experiment.py --config configs/dev.yaml --checkpoint checkpoints/adaptive_dev_with_kd/best.pt --mode calibrated --split validation --degradation both --device cuda
```

The `both` setting gives **nine quality × budget conditions per degradation
family** (18 rows, with clean repeated for comparison). Use `--degradation
blur` or `--degradation noise` for one family. The default output is:

- `results/adaptive_dev_with_kd/degradation_validation_calibrated.json`
- `results/adaptive_dev_with_kd/degradation_validation_calibrated.csv`
- `results/adaptive_dev_with_kd/degradation_validation_calibrated_confident_errors.csv`

Configured mild/severe blur standard deviations are 1/2 pixels; mild/severe
noise standard deviations are 0.05/0.15 on the normalized [-1, 1] image.
Blur uses a Gaussian kernel truncated at three standard deviations with
replicate padding. Noise uses one shared draw across the three identical
grayscale channels, clips to [-1, 1], and is deterministic per image and
paired across severities. The experiment
reports Exit 1/2 uncertainty, preferred and actual depth, forced-exit rate,
prediction metrics, and highly confident wrong labels. A confident error
means at least one wrong binary label was decided with confidence >= 0.9.
The clean training transforms and fitted routing thresholds stay unchanged.

Average FLOPs is `null` until measured per-exit costs are supplied via
`--flops-file` (JSON keys: `method`, `exit1`, `exit2`, `exit3`). No FLOP values
are inferred from depth. The report computes all exits for exact preferred
depth, so it is an offline diagnostic rather than a latency measurement.
`--split test` is reserved for final evaluation. No real degradation results
have been generated yet.

## Phase 8: FLOPs, latency, and Pareto plots

Run these commands after a trained Phase 4 checkpoint and its validation
calibration artifact exist. They do not train or retune the model.

```bash
python scripts/benchmark_flops.py --config configs/dev.yaml --checkpoint checkpoints/adaptive_dev_with_kd/best.pt
python scripts/benchmark_latency.py --config configs/dev.yaml --checkpoint checkpoints/adaptive_dev_with_kd/best.pt --device cuda --samples 32 --warmup 10 --measurements 30
python scripts/generate_pareto.py --config configs/dev.yaml --checkpoint checkpoints/adaptive_dev_with_kd/best.pt --device cuda
```

The default outputs are `results/adaptive_dev_with_kd/flops.{json,csv}`,
`latency.{json,csv}`, `pareto_points.csv`, `pareto_summary.json`, and three
PNG files under `plots/`. Use the same config, checkpoint, and calibration
artifact for all three commands. Existing outputs require `--overwrite`.

FLOPs use one module-hook method at batch size 1 and the configured image
size: two operations per multiply-add, plus bias, batch normalization, ReLU,
pooling, residual additions, and the selected exit heads. Routing entropy,
tensor indexing, and memory operations are excluded. Fixed Exit 2 and Full
compute only their selected head; adaptive Exit 2 and Final also compute
earlier heads. Average adaptive FLOPs weights these actual paths by the
validation exit distribution.

Latency uses the first 32 validation images by default, preloaded onto the
requested device, and excludes image loading. It measures static paths and
every calibrated and uncalibrated threshold-sweep policy through conditional
inference. Results include mean, p50, and p95 after warmup, with CUDA
synchronization around each timed call. Adaptive timing includes sigmoid,
entropy, and decisions. A separate cached-logit diagnostic measures both
threshold decisions and final sigmoid, without model layers. FLOPs and
latency are separate measurements.

Pareto points use validation labels and the saved validation-fitted sweep;
the accuracy axis is exact-match subset accuracy for this five-label task.
The other axes are macro AUROC and measured mean latency. Dominance is
computed from actual coordinates, with equal points retained. These commands
need a trained checkpoint and are not evidence of real performance until run
against it. Use `--device cpu` for small smoke tests.
