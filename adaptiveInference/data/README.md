# Data

This project uses NIH ChestX-ray14. Raw images are not stored in Git. Local
images and metadata live under `ChestXray14/dev-mini/` or
`ChestXray14/full/`; split manifests live under `data/splits/`.

Target labels, in order:

- Atelectasis
- Effusion
- Mass
- Nodule
- Pneumothorax

Run `python scripts/prepare_data.py --config configs/dev.yaml` to filter
metadata to local PNG files and save deterministic, patient-separated CSV
manifests under `data/splits/dev/`. The script refuses to replace existing
manifests unless `--overwrite` is passed.

The loader returns five independent float labels. Images are converted to
grayscale, copied to three channels, resized, and scaled to [-1, 1]. Train,
validation, and test use deterministic preprocessing; no synthetic
degradation is applied. `pos_weight` uses the train manifest only.
