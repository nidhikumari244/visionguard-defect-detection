# VisionGuard Experiment Notes

These notes summarize the useful experimental history found in the project notebooks.

## Notebook Sources

- `Untitled1 (1).ipynb`: early research notebook with bottle baseline, combined-layer improvement, cable normalization, pixel AUROC, and threshold experiments.
- `Untitled2.ipynb`: later reusable per-category evaluation notebook for carpet, wood, screw, and saved memory-bank artifacts.
- `FINAL_VISIONGUARD_NOTEBOOK.ipynb`: final inference, Gradio deployment, Streamlit export, and repository packaging notebook.

## Experiment 1: Single-Layer Baseline

The first version used only one Wide ResNet-50 feature layer for patch embeddings.

Pipeline:

1. Extract patch features from one feature map.
2. Build a memory bank from normal bottle images.
3. Fit nearest-neighbor search on the memory bank.
4. Use maximum patch distance as the image anomaly score.

Reported bottle result:

| Setup | Image AUROC |
|-------|-------------|
| Single feature layer | 0.8563 |

## Experiment 2: Combined Multi-Scale Features

The second version combined two feature layers:

- Layer 1: fine spatial texture
- Layer 2: higher-level structure

Layer 2 was upsampled to match Layer 1, then both were concatenated into 768-dimensional patch features.

Reported bottle result:

| Setup | Image AUROC |
|-------|-------------|
| Single feature layer | 0.8563 |
| Combined feature layers | 0.9746 |

Major finding:

Combined multi-scale features significantly improved bottle anomaly ranking.

## Experiment 3: Feature Normalization

Cable performance was weak before normalization because feature channels from different layers had different magnitudes. The normalized version computed mean and standard deviation from the full normal training memory bank, then applied z-score normalization.

Reported cable result:

| Setup | Image AUROC |
|-------|-------------|
| Combined layers without normalization | 0.620 |
| Combined layers with normalization | 0.778 |

Major finding:

Per-channel feature normalization is critical for categories with complex structure, especially cable.

## Experiment 4: Threshold Tuning

The original threshold used P95 of normal training scores. Cable had high precision but weak recall, so lower percentiles were tested.

Reported cable threshold study:

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| P95 | 0.8846 | 0.2500 | 0.3898 |
| P90 | 0.8627 | 0.4783 | 0.6154 |
| P85 | 0.8852 | 0.5870 | 0.7059 |

Major finding:

Threshold choice has a large effect on recall and F1. P85 was better for cable than P95.

## Experiment 5: Pixel-Level Localization

The notebooks include pixel-level AUROC calculation using MVTec ground-truth masks. This supports the claim that VisionGuard does not only classify images, but also localizes defective regions.

The current repo reports:

| Category | Pixel AUROC |
|----------|-------------|
| Bottle | 0.954 |
| Cable | 0.915 |
| Carpet | 0.958 |
| Wood | 0.923 |
| Screw | 0.970 |

Major finding:

Pixel-level localization is consistently strong across categories, even where image-level classification is weaker.

## Report-Ready Ablation Summary

Use this table in the final project report:

| Ablation | Before | After | Improvement |
|----------|--------|-------|-------------|
| Single layer to combined layers on bottle | 0.8563 | 0.9746 | +13.8% |
| Unnormalized to normalized features on cable | 0.620 | 0.778 | +25.5% |
| Cable threshold P95 to P85 | 0.3898 F1 | 0.7059 F1 | +81.1% |

## Technical Lessons

1. Multi-scale patch embeddings are better than single-layer features.
2. Feature normalization should be treated as part of the model, not as an optional preprocessing step.
3. Thresholds should be selected per category instead of using one fixed percentile everywhere.
4. Pixel AUROC and image AUROC can disagree, especially for tiny defects such as screw defects.
5. Top-k image scoring should be evaluated as the next improvement over maximum patch distance.
