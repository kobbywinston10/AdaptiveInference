# Data

This project uses NIH ChestX-ray14.

The raw image files are not stored in this repository.

Expected local structure:

data/
├── raw/
│   ├── images/
│   └── Data_Entry_2017.csv
├── metadata/
└── splits/

Target labels:
- Atelectasis
- Effusion
- Mass
- Nodule
- Pneumothorax

The dataset should be split at patient level.
