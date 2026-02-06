# Efficient Video Inference with Vision Transformers via Temporal Token Reuse

## Overview
Vision Transformers (ViTs) are widely used for image and video understanding tasks but suffer from high inference cost due to the quadratic complexity of self-attention with respect to the number of tokens. While recent methods such as Token Merging (ToMe) reduce spatial redundancy within a single frame, most real-world applications process video streams where consecutive frames exhibit strong temporal redundancy.
This project explores **temporal token reuse** for Vision Transformers, with the goal of accelerating video inference by reusing previously computed token representations and key–value (KV) pairs across frames, particularly for temporally stable background regions.

## Motivation
In many video scenarios (e.g., surveillance, robotics, edge cameras), large portions of the scene remain unchanged across consecutive frames. Standard ViT inference recomputes attention, keys, and values for all tokens at every frame, resulting in redundant computation and unnecessary latency.
By exploiting **temporal consistency**, this project aims to reduce inference cost while preserving model accuracy, making Vision Transformers more suitable for real-time and edge deployments.

## Core Idea
The project builds on existing token reduction techniques and extends them to the temporal domain:

- **Spatial token reduction:** Token Merging (ToMe) merges semantically similar tokens within a single frame.
- **Temporal token reuse:** Tokens corresponding to stable background regions are matched across frames, and their previously computed representations and KV pairs are reused instead of recomputed.

This approach is inspired by KV caching in language models but adapted to the self-attention–only structure of Vision Transformers.

## Approach
The proposed system investigates:
- Temporal background token identification and matching across frames
- Reuse of cached key–value representations for matched tokens
- Integration with existing token merging and pruning techniques
- Training-free and lightweight inference-time strategies

The method is evaluated by analyzing trade-offs between:
- Inference latency and throughput
- Computational cost (FLOPs)
- Memory usage
- Prediction stability and accuracy

## Project Status
- Reproduced and validated Token Merging (ToMe) baselines
- Benchmarked inference speed improvements and token reduction behavior
- Visualized token merging across transformer layers
- Verified prediction stability under aggressive token reduction
- Project environment successfully configured in PyCharm

## Datasets
Public video datasets will be used for evaluation (to be finalized).  
Dataset details, licenses, and access instructions will be documented in `data/README.md`.

Raw data will not be modified directly.

## Expected Deliverables
- Reproducible research codebase
- Baseline and temporally optimized ViT inference implementations
- Quantitative evaluation and analysis
- IEEE-style technical report
- Poster presentation

## References
- Dosovitskiy et al., *An Image is Worth 16×16 Words*, ICLR 2021  
- Bolya et al., *Token Merging: Your ViT but Faster*, 2022  
- Bolya et al., *VidToMe: Video Token Merging*, 2023  
- Li et al., *Vid-TLDR: Training-Free Token Merging for Lightweight Video Transformers*, 2023
