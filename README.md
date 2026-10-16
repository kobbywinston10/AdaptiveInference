# Adaptive chest X-ray inference

This project studies whether an adaptive classifier can reduce computation for
chest X-rays while preserving prediction quality. It uses a RadImageNet-pretrained
ResNet50 and five labels from NIH ChestX-ray14: Atelectasis, Effusion, Mass,
Nodule, and Pneumothorax. The main comparison is prediction quality versus
computation for early exits and full-network inference.

## Model and dataset downloads

**RadImageNet pretrained weights.** The ResNet50 backbone is initialized from
the [official RadImageNet PyTorch pretrained models](https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing).
The [original RadImageNet repository](https://github.com/BMEII-AI/RadImageNet)
contains the model definitions, examples, and publication information. Place
the ResNet50 weights at `weights/ResNet50.pt`.

**NIH ChestX-ray14.** Download `Data_Entry_2017_v2020.csv` and the 12 image
archives (`images_001.tar.gz` through `images_012.tar.gz`) from the
[official NIH Clinical Center download page](https://nihcc.app.box.com/v/ChestXray-NIHCC).
NIH also provides [batch_download_zips.py](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/220660789610)
for downloading the archives. Place the metadata and extracted images under
`ChestXray14/full/`, as expected by `configs/full.yaml`.

After saving the NIH script locally, a basic download and extraction workflow
on Linux is:

```bash
python batch_download_zips.py
mkdir -p ChestXray14/full/images
for archive in images_*.tar.gz; do
    tar -xzf "$archive" -C ChestXray14/full/images/
done
```

The RadImageNet download provides the **original pretrained weights**, not the
adaptive model trained by this project. A final adaptive checkpoint is not
included in this repository; reproducing the final model requires training it
or obtaining that checkpoint separately.

## Further information

- [Project specification](docs/PROJECT_SPEC.md) describes the experiment.
- [DAS-5 setup guide](docs/DAS5_SETUP%20%281%29.md) contains environment and run instructions.
- [Data notes](data/README.md) describe the dataset layout and target labels.
