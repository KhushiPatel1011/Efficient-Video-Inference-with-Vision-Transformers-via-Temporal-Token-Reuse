# Dataset Documentation

This directory will contain all dataset-related information for the project  
**Efficient Video Inference with Vision Transformers via Temporal Token Reuse**.

To ensure reproducibility and compliance with dataset licenses, **raw data will never be modified directly** and **no large datasets will be committed to GitHub**.

## Directory Structure

data/
├── raw/ # Original datasets (read-only, not tracked by git)
├── processed/ # Preprocessed frames / clips / cached features
└── README.md # This file

## Intended Dataset Characteristics

This project focuses on **video inference**, so datasets must:
- Contain **temporal continuity** (multiple consecutive frames)
- Exhibit **temporal redundancy** (static or slowly changing regions)
- Be suitable for benchmarking inference efficiency (latency, FPS)

## Candidate Public Datasets (Planned)

> Final selection will be updated once experiments begin.

Examples:
- **Kinetics-400 / Kinetics-700** (video classification)
- **UCF101 / HMDB51** (action recognition)
- **Synthetic or trimmed clips** for controlled temporal redundancy experiments

## Data Usage Policy

- Raw datasets must be placed under `data/raw/`
- Preprocessing scripts will generate:
  - resized frames
  - normalized tensors
  - cached token / KV representations (when applicable)
- Generated outputs must be stored under `data/processed/`

No preprocessing should overwrite raw files.

## Reproducibility

If a dataset cannot be redistributed:
- Download instructions
- Licensing notes
- Preprocessing commands

will be documented here so results can be reproduced from a clean clone.

## Notes
This project prioritizes **inference efficiency and correctness**, not large-scale training.  
Therefore, datasets will be used primarily for **evaluation**, not retraining ViTs.



