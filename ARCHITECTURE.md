# VisionGuard Architecture

## Pipeline

Input -> Wide ResNet-50 -> Memory Bank -> kNN -> Heatmap -> CLIP -> Report

## Feature Extraction

- Layer 1: 56x56x256 (fine texture)
- Layer 2: 28x28x512 (structural context)
- Combined: 768-dim patch vectors

## Key Findings

- Multi-scale features: +13.8% AUROC on bottle
- Z-score normalization: +25.5% AUROC on cable
- Pixel AUROC consistently high (0.915-0.970)
