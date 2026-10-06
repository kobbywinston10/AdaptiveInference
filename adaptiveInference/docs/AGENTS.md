# AGENTS.md

## Purpose

This repository implements the Big Data at the Edge project:

**Reliability- and Resource-Aware Adaptive Inference for Chest X-ray Classification**

Treat `docs/PROJECT_SPEC.md` as the source of truth for the experiment design. Before editing code, inspect the relevant existing files and avoid duplicating working functionality.

## Core Rules

1. Do not fabricate metrics, benchmark numbers, dataset contents, or experimental results.
2. Do not launch long training jobs unless explicitly asked.
3. Do not tune anything on the final test set.
4. Do not hardcode machine-specific absolute paths.
5. Preserve the project as a **5-label multi-label** classification problem.
6. Use patient-level dataset splitting.
7. Run relevant tests or smoke tests after changes.
8. Prefer small, modular changes over large rewrites.
9. Reuse existing code and repository structure where possible.
10. Clearly distinguish synthetic test fixtures from real experimental data.

## Dataset

Dataset: **NIH ChestX-ray14**

Target labels:

- Atelectasis
- Effusion
- Mass
- Nodule
- Pneumothorax

The task is multi-label. One image may contain multiple positive findings.

Use:

- raw logits during training;
- `BCEWithLogitsLoss`;
- sigmoid probabilities during inference.

Current local development data is under:

```text
ChestXray14/dev-mini/
```

The full dataset will later be under:

```text
ChestXray14/full/
```

For development subsets, filter metadata to image files that physically exist.

## Data Leakage Rules

Splits must be patient-level.

No patient may occur in more than one of:

- train
- validation
- test

Use the training split only for:

- fitting model parameters;
- class prevalence;
- `pos_weight`.

Use the validation split only for:

- model selection;
- calibration;
- routing threshold selection;
- hyperparameter tuning.

Use the test split only for final evaluation.

## Model

Backbone:

```text
RadImageNet-pretrained ResNet50
```

Checkpoint:

```text
weights/ResNet50.pt
```

Reference notebook:

```text
notebooks/RadImageNetExample/pytorch_example.ipynb
```

Classifier output:

```text
5 logits
```

Adaptive paths:

```text
layer2 -> Exit 1
layer3 -> Exit 2
layer4 -> Final Exit
```

Early exits must genuinely skip later layers at inference.

## Self-Distillation

The final exit teaches the early exits.

This is a multi-label problem, so do not blindly use multiclass softmax KD.

Use temperature-scaled sigmoid outputs with an appropriate binary/Bernoulli soft-target loss.

## Routing

Primary uncertainty:

```text
maximum binary entropy across the 5 sigmoid outputs
```

Optional ablation:

```text
mean binary entropy
```

Fit calibration and exit thresholds using validation data only.

Resource budgets:

```text
LOW    -> maximum Exit 1
MEDIUM -> maximum Exit 2
HIGH   -> Final Exit allowed
```

Record both the uncertainty-preferred exit and the actual budget-constrained exit.

## Evaluation

Core metrics:

- macro AUROC
- per-label AUROC
- macro F1
- documented multi-label accuracy metric
- calibration error
- exit distribution
- average exit depth
- average FLOPs/image
- latency
- budget-forced exit rate
- premature-exit rate

The main course figure is the performance-vs-FLOPs Pareto frontier.

Static comparison points:

- Fixed Exit 1
- Fixed Exit 2
- Full ResNet50

Adaptive points come from a routing-threshold sweep.

## Image Degradation

Core evaluation-only degradations:

- Gaussian blur
- Gaussian noise

Severity:

- clean
- mild
- severe

Do not assume degradation increases uncertainty; measure it.

Do not claim low compute budget causes degraded images. They are separate variables.

## Git Policy

Do not commit large artifacts such as:

```text
ChestXray14/dev-mini/images_001/
ChestXray14/dev-mini/images_001.tar.gz
ChestXray14/full/
weights/*.pt
weights/*.pth
checkpoints/
*.onnx
```

Commit code, configs, documentation, small split files, small result CSVs, and final plots.

## Working Procedure

For every task:

1. Read `AGENTS.md`.
2. Read only the relevant section(s) of `docs/PROJECT_SPEC.md`.
3. Inspect the existing implementation.
4. State a short implementation plan.
5. Make the smallest coherent change.
6. Run relevant tests/smoke tests.
7. Report:
   - files changed;
   - tests run/results;
   - what remains untested on real data/hardware;
   - blockers;
   - recommended next step.
