# Efficient Video Inference with Vision Transformers via Temporal Token Reuse

## Motivation
Vision Transformers (ViTs) are strong for video understanding, but they are expensive at inference time because they recompute self-attention for **every token in every frame**. In many real videos, large regions (e.g., background) remain stable across frames, so recomputing representations repeatedly wastes compute and increases latency.

This project explores **temporal token reuse**: identifying tokens that correspond to stable regions across consecutive frames and **reusing previously computed representations** (including attention key–value pairs) instead of recomputing them.

## Problem Statement
**Goal:** Design and evaluate methods that exploit temporal redundancy in video to accelerate ViT inference while preserving accuracy.

**Core idea:** For consecutive frames, detect which spatial tokens are temporally stable (background / unchanged regions) and:
- reuse their cached representations (e.g., token embeddings and/or KV pairs),
- recompute only the tokens likely to change (foreground, motion, new objects),
- optionally combine temporal reuse with existing token reduction methods (e.g., ToMe-style merging).

## Task Definition & Scoping
- **Learning setting:** Primarily inference-time optimization (training-free), with optional lightweight learning-based policies for reuse decisions.
- **Input:** Video clip or a stream of frames (RGB).
- **Output:** Model predictions per frame or per clip (classification and/or other video understanding outputs depending on benchmark).
- **Constraints:** Preserve accuracy as much as possible while improving efficiency:
  - latency / throughput
  - FLOPs
  - memory usage (cache cost)
  - compatibility with real-time / edge-style settings

## Dataset(s)
**TBD (will be updated).** We will use public video benchmarks suitable for evaluating temporal redundancy and inference efficiency.  
Details, licenses, and access instructions will be documented in: `data/README.md` (raw data will never be modified in place).

## Evaluation Plan
We will measure trade-offs between:
- **Accuracy / performance:** top-1 / top-5 accuracy (or task-appropriate metrics), prediction stability across frames
- **Efficiency:** throughput (frames/sec), latency per frame, estimated FLOPs
- **Memory:** cache size (KV storage), peak memory usage

We will include comparisons against:
- **Baseline ViT inference** (no reuse, no reduction)
- **Token reduction baselines** (e.g., token merging where applicable)
- **Temporal reuse variants** (different matching strategies, reuse thresholds, cache update rules)

## Expected Deliverables
- A reproducible codebase following the required structure
- Baseline and temporal reuse implementations
- Experimental results (tables/plots) and error analysis
- IEEE-style technical report (in `docs/report/`)
- Poster presentation

## Repository Structure (Required)
This repository follows the required capstone layout:
- `configs/` experiment configs (`.yaml`)
- `data/` raw/processed with `data/README.md` for dataset instructions
- `src/` core code (data, models, training, evaluation, utils)
- `experiments/` logs/results outputs
- `notebooks/` analysis and prototyping
- `scripts/` entrypoints to run training/evaluation/inference
- `docs/` report and documentation
- `tests/` unit/integration tests (as feasible)

## Setup
### 1) Create environment and install dependencies
> We keep dependencies in `requirements.txt` for reproducibility.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
