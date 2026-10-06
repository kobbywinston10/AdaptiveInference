# Project Specification

## 1. Objective

Build an adaptive chest X-ray classifier for the Big Data at the Edge course using a **RadImageNet-pretrained ResNet50** fine-tuned on **NIH ChestX-ray14**.

The system should not always execute the full network. It should expose three shared-weight inference paths:

1. Exit 1 after `layer2`
2. Exit 2 after `layer3`
3. Final prediction after `layer4`

The central question is:

> Can calibrated early-exit inference allocate computation according to input difficulty while respecting changing resource budgets, and achieve a better prediction-quality-versus-compute trade-off than fixed-depth inference?

The required course artifact is a performance-vs-FLOPs Pareto frontier.

---

## 2. Repository Layout

Current repository:

```text
adaptiveInference/
├── ChestXray14/
│   ├── dev-mini/
│   │   ├── images_001/
│   │   ├── Data_Entry_2017_v2020.csv
│   │   └── images_001.tar.gz
│   └── full/
├── configs/
│   ├── tiny.yaml
│   └── full.yaml
├── data/
│   ├── metadata/
│   ├── splits/
│   └── README.md
├── docs/
│   └── PROJECT_SPEC.md
├── notebooks/
│   └── RadImageNetExample/
│       └── pytorch_example.ipynb
├── results/
├── scripts/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── routing/
│   ├── training/
│   └── utils/
├── weights/
│   ├── README.md
│   └── ResNet50.pt
├── .gitignore
├── README.md
└── requirements.txt
```

Add:

```text
configs/dev.yaml
```

Configuration roles:

- `tiny.yaml`: smoke tests
- `dev.yaml`: laptop development using `ChestXray14/dev-mini`
- `full.yaml`: full DAS-5 experiment

Preserve this structure unless a change has a clear engineering benefit.

---

## 3. Dataset and Labels

Dataset: **NIH ChestX-ray14**

Use only these five targets:

- Atelectasis
- Effusion
- Mass
- Nodule
- Pneumothorax

This remains a **multi-label** problem.

Example:

```text
Finding Labels = Atelectasis|Effusion
```

becomes:

```text
Atelectasis     1
Effusion        1
Mass            0
Nodule          0
Pneumothorax    0
```

Model output:

```text
5 independent logits
```

Training:

```text
BCEWithLogitsLoss
```

Inference:

```text
sigmoid(logits)
```

For dev subsets, metadata must be filtered to image files that actually exist locally.

---

## 4. Patient-Level Splitting

Use patient-level splits so that images from the same patient never cross partitions.

Default:

```text
70% train
15% validation
15% test
```

The split must be deterministic and saved under:

```text
data/splits/
```

Required assertions:

```text
train ∩ validation = empty
train ∩ test       = empty
validation ∩ test  = empty
```

The test set must not be used for:

- training;
- class weights;
- calibration;
- threshold selection;
- hyperparameter tuning.

---

## 5. Class Imbalance

Compute per-label positive weights from the training split only:

```text
pos_weight = negatives / positives
```

Pass them to `BCEWithLogitsLoss`.

Record:

- positive count;
- negative count;
- prevalence;
- `pos_weight`.

---

## 6. Image Preprocessing

Chest X-rays are grayscale while ResNet50 expects three channels.

Use a consistent documented conversion:

```text
grayscale
-> 3 channels
-> resize
-> tensor
-> normalization
```

Training augmentation should remain conservative.

Validation and test preprocessing must be deterministic.

Synthetic degradations belong in evaluation, not the normal training transform.

---

## 7. Backbone and Static Baseline

Backbone:

```text
RadImageNet-pretrained ResNet50
```

Checkpoint:

```text
weights/ResNet50.pt
```

Use:

```text
notebooks/RadImageNetExample/pytorch_example.ipynb
```

as a reference for checkpoint loading.

Inspect the actual checkpoint structure. Handle common forms such as `state_dict`, nested dictionaries, and `module.` prefixes without silently hiding major mismatches.

Replace the final classifier with:

```python
Linear(2048, 5)
```

First establish a full-depth static baseline.

Primary validation model-selection metric:

```text
macro AUROC
```

Save the best baseline checkpoint and keep it unchanged for later comparison.

---

## 8. Adaptive Architecture

Architecture:

```text
Input
  ↓
Stem
  ↓
layer1
  ↓
layer2 ─────→ Exit 1
  ↓
layer3 ─────→ Exit 2
  ↓
layer4 ─────→ Final Exit
```

Suggested early-exit heads:

```python
AdaptiveAvgPool2d(1)
Flatten()
Linear(channels, 5)
```

Provide two forward modes:

```python
forward_all_exits(x)
```

for training/evaluation, and:

```python
forward_adaptive(x, ...)
```

for real conditional inference.

If Exit 1 is selected, `layer3` and `layer4` must not execute.

If Exit 2 is selected, `layer4` must not execute.

Add tests proving this.

---

## 9. Self-Distillation

Use **self-distillation** as the compression technique.

Teacher:

```text
Final Exit
```

Students:

```text
Exit 1
Exit 2
```

Because the task is multi-label, use temperature-scaled sigmoid outputs rather than standard multiclass softmax KD.

Conceptually:

```text
p_teacher = sigmoid(z_teacher / T)
p_student = sigmoid(z_student / T)
```

Use a suitable binary/Bernoulli soft-target loss.

Each early-exit loss combines:

```text
hard-label BCE
+
weighted distillation loss
```

Make configurable:

- temperature;
- hard-label weight;
- KD weight;
- per-exit auxiliary weights.

Compare early-exit performance with and without KD.

---

## 10. Uncertainty and Calibration

For sigmoid probability `p`:

```text
H(p) = -p log(p) - (1-p) log(1-p)
```

Primary uncertainty:

```text
U(x) = max_j H(p_j)
```

across the five labels.

Optional ablation:

```text
mean binary entropy
```

Use numerical clamping.

Calibration must be fit on the validation set only.

Initial approach:

```text
one scalar temperature per exit
```

Compare calibrated and uncalibrated routing.

---

## 11. Early-Exit Routing

Routing logic:

```text
Exit 1:
if U1 <= threshold_1:
    stop
else:
    continue

Exit 2:
if U2 <= threshold_2:
    stop
else:
    continue

Otherwise:
    use Final Exit
```

Thresholds must be selected on validation data.

Generate several policies by sweeping thresholds from aggressive to conservative early exiting.

These policies will form adaptive operating points on the Pareto plot.

---

## 12. Resource-Aware Controller

Support:

```text
LOW
MEDIUM
HIGH
```

Initial structural interpretation:

```text
LOW    -> maximum Exit 1
MEDIUM -> maximum Exit 2
HIGH   -> Final Exit allowed
```

The uncertainty policy determines the preferred exit first. The resource budget then constrains the maximum permitted depth.

Record:

```text
preferred exit
actual exit
```

and calculate:

```text
budget-forced exit rate
```

After benchmarking, associate the budget states with measured FLOP/latency constraints rather than invented values.

---

## 13. Image Degradation Experiment

Core evaluation-only degradations:

- Gaussian blur
- Gaussian noise

Severity:

- clean
- mild
- severe

Optional later:

- contrast reduction;
- downsampling.

Apply degradations dynamically during evaluation.

Run the factorial experiment:

```text
image quality × resource budget
```

giving nine core conditions:

```text
clean  × low
clean  × medium
clean  × high
mild   × low
mild   × medium
mild   × high
severe × low
severe × medium
severe × high
```

Do not claim resource budget causes image degradation.

Test whether degradation changes:

- uncertainty;
- preferred exit depth;
- actual exit depth;
- average FLOPs;
- predictive performance.

Also inspect incorrect but highly confident degraded predictions.

---

## 14. Required Comparisons

Core comparison set:

1. Full ResNet50
2. Fixed Exit 1
3. Fixed Exit 2
4. Adaptive uncalibrated
5. Adaptive calibrated

Optional extension:

6. Static INT8 ResNet50

Do not add pruning to the core project.

---

## 15. Metrics

### Prediction

- macro AUROC
- per-label AUROC
- macro F1
- documented multi-label accuracy metric

### Calibration

- suitable multi-label ECE/calibration error

### Adaptive Behaviour

- Exit 1 percentage
- Exit 2 percentage
- Final Exit percentage
- average exit depth
- average FLOPs/image
- budget-forced exit rate
- premature-exit rate

Document the correctness definition used for premature-exit rate.

### Efficiency

- Exit 1 FLOPs
- Exit 2 FLOPs
- Final Exit FLOPs
- adaptive average FLOPs
- mean latency
- p50 latency
- p95 latency
- controller overhead
- model size
- peak memory if practical

---

## 16. FLOPs and Pareto Frontier

Use one FLOP-counting method consistently.

Include early-exit head cost.

For adaptive policy `k`:

```text
average_FLOPs_k =
    r1 * F1
  + r2 * F2
  + r3 * F3
```

with:

```text
r1 + r2 + r3 = 1
```

Static Pareto points:

- Fixed Exit 1
- Fixed Exit 2
- Full ResNet50

Adaptive points:

- threshold-sweep routing policies

Generate at least:

1. course accuracy metric vs FLOPs
2. macro AUROC vs FLOPs
3. predictive performance vs measured latency

Compute Pareto dominance programmatically.

Do not manually invent or move points.

---

## 17. Latency Benchmarking

Default batch size:

```text
1
```

Use:

- warm-up runs;
- repeated measurements;
- CUDA synchronization where needed.

Report:

- mean;
- p50;
- p95.

Measure:

- Exit 1;
- Exit 2;
- Final Exit;
- complete adaptive inference.

Adaptive latency must include:

- exit head;
- sigmoid;
- entropy;
- routing decision.

Measure controller overhead separately where practical.

Do not equate FLOPs with latency.

---

## 18. Results and Reproducibility

Suggested result layout:

```text
results/
└── <run_name>/
    ├── config.yaml
    ├── environment.json
    ├── baseline_metrics.csv
    ├── exit_metrics.csv
    ├── adaptive_metrics.csv
    ├── degradation_results.csv
    ├── flops.csv
    ├── latency.csv
    ├── pareto_points.csv
    └── plots/
```

Record where practical:

- seed;
- config;
- Git commit;
- Python version;
- PyTorch version;
- CUDA version;
- hardware;
- timestamp.

Evaluation must be runnable from saved checkpoints without retraining.

---

## 19. Expected Scripts

By project completion, likely scripts include:

```text
scripts/check_environment.py
scripts/prepare_data.py
scripts/train_baseline.py
scripts/train_adaptive.py
scripts/calibrate.py
scripts/evaluate_static.py
scripts/evaluate_adaptive.py
scripts/run_degradation_experiment.py
scripts/benchmark_flops.py
scripts/benchmark_latency.py
scripts/generate_pareto.py
scripts/run_experiment.py
```

Only create a script when its phase requires it.

---

## 20. Implementation Phases

### Phase 1 — Environment and Configuration

- verify `.gitignore`;
- verify dependencies;
- add `configs/dev.yaml`;
- implement config loading;
- add environment checker;
- verify dataset/checkpoint paths.

### Phase 2 — Data Pipeline

- load metadata;
- filter to available images;
- parse five labels;
- patient-level split;
- leakage checks;
- class weights;
- Dataset/DataLoaders;
- tiny/dev/full support.

### Phase 3 — Static Baseline

- inspect RadImageNet checkpoint;
- load ResNet50;
- replace classifier;
- run real dev forward pass;
- train/evaluate baseline;
- save best checkpoint.

### Phase 4 — Early Exits and Self-Distillation

- add two exits;
- verify conditional skipping;
- train auxiliary heads;
- implement multi-label KD;
- compare with/without KD.

### Phase 5 — Calibration and Routing

- temperature scaling;
- binary entropy;
- threshold selection;
- calibrated vs uncalibrated routing.

### Phase 6 — Resource Controller

- LOW/MEDIUM/HIGH;
- preferred vs actual exit;
- budget-forced exit rate.

### Phase 7 — Degradation Experiment

- blur/noise;
- clean/mild/severe;
- quality × budget evaluation.

### Phase 8 — Benchmarking and Pareto

- FLOPs;
- latency;
- threshold sweep;
- Pareto computation;
- plots.

### Phase 9 — Full DAS-5 Experiment

Only after local pipeline validation:

- full dataset;
- final training;
- final calibration/evaluation;
- degradation study;
- benchmarks;
- final Pareto results.

### Phase 10 — Audit

Check:

- patient leakage;
- test leakage;
- multi-label correctness;
- checkpoint loading;
- genuine early-exit skipping;
- KD implementation;
- calibration;
- routing;
- budget logic;
- FLOP methodology;
- latency methodology;
- reproducibility;
- documentation.

---

## 21. Testing Priorities

Add tests for:

- label parsing;
- deterministic patient split;
- no patient leakage;
- missing-image filtering;
- Dataset output shape;
- checkpoint loading;
- output shapes;
- early-exit skipping;
- KD loss;
- entropy;
- calibration save/load;
- routing;
- resource limits;
- degradation transforms;
- average FLOPs;
- Pareto dominance;
- checkpoint save/load.

Synthetic fixtures are allowed for automated tests but must never be reported as real experimental results.

---

## 22. Immediate Milestones

### Milestone 1

Using:

```text
ChestXray14/dev-mini/
```

produce valid patient-separated train/validation/test DataLoaders for the five-label task with no patient leakage.

### Milestone 2

Using:

```text
weights/ResNet50.pt
```

load the RadImageNet-pretrained ResNet50, replace its classifier with five outputs, and successfully run a forward pass on a real development batch.

Only after these work should substantial model training begin.
