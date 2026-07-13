# VisionGuard Major Project Roadmap

This roadmap turns VisionGuard from an internship demo into a final-year major project with a stronger system design, better evaluation, and a more professional application layer.

## Current System

VisionGuard currently supports five MVTec AD categories:

- Bottle
- Cable
- Carpet
- Wood
- Screw

The current inference pipeline is:

1. Resize and center-crop the uploaded image to 224 x 224.
2. Extract patch features from Wide ResNet-50 layers 1 and 2.
3. Upsample and concatenate features into 768-dimensional patch vectors.
4. Normalize patch vectors using category-specific mean and standard deviation.
5. Compare patches against a normal-only memory bank using nearest-neighbor distance.
6. Use patch distances for anomaly score and heatmap generation.
7. Crop the highest-anomaly region and use CLIP for zero-shot defect description.

## Highest-Impact Improvements

### 1. Recalibrate Image-Level Scoring

The current app uses maximum patch distance as the image score. This works for large defects, but can be unstable for tiny defects and noisy patches.

Recommended scoring methods to evaluate:

- Maximum patch distance
- Top-1% average patch distance
- Top-5% average patch distance
- Mean of patches above the 95th percentile

Expected impact:

- Better image-level AUROC and F1 for the screw category
- More stable normal/anomaly decisions
- Stronger explanation for threshold selection in the final report

### 2. Threshold Optimization

The current thresholds are fixed percentile values. For the final project, evaluate thresholds on validation/test data and report the best threshold by F1 score.

Recommended table:

| Category | Score Method | Best Threshold | Precision | Recall | F1 | Image AUROC |
|----------|--------------|----------------|-----------|--------|----|-------------|

This makes the project more defensible than manually choosing P85, P90, or P95.

### 3. Complete All 15 MVTec AD Categories

Expanding from 5 to all 15 categories is the most visible major-project upgrade.

Additional categories:

- Capsule
- Grid
- Hazelnut
- Leather
- Metal nut
- Pill
- Tile
- Toothbrush
- Transistor
- Zipper

Expected impact:

- Stronger benchmark section
- Better project scale
- More complete industrial inspection story

### 4. Add Baseline Comparisons

Add at least one baseline so the report shows comparative analysis, not only one method.

Recommended baselines:

- Autoencoder reconstruction error
- PaDiM-style Gaussian feature distance
- Single-layer memory bank
- Unnormalized feature bank

Minimum useful comparison:

| Method | Bottle AUROC | Cable AUROC | Carpet AUROC | Wood AUROC | Screw AUROC |
|--------|--------------|-------------|---------------|------------|-------------|
| Single layer memory bank | | | | | |
| Combined layer memory bank | | | | | |
| Combined + normalization | | | | | |
| Combined + top-k scoring | | | | | |

### 5. Improve CLIP Defect Descriptions

The current CLIP prompts are short labels. Use more descriptive prompts with category context.

Example:

```text
a photo of a bottle with a crack on glass
a photo of a bottle with liquid contamination
a photo of a normal clean bottle
```

Report top-3 CLIP candidates instead of only top-1.

## Web App Improvements Already Added

The Streamlit app has been upgraded with:

- Dashboard page
- Single-image inspection page
- Batch inspection page
- Benchmark page
- Architecture and roadmap page
- Inspection history
- Downloadable inspection report
- Downloadable batch CSV
- Optional top-1% average scoring mode for experimentation

## Experiment Script

The provided `FINAL_VISIONGUARD_NOTEBOOK.ipynb` contains the final loading, inference, Gradio deployment, and export flow. The second notebook, `Untitled2.ipynb`, contains the missing memory-bank creation and MVTec evaluation logic.

That repeated Colab code has been converted into:

```bash
python scripts/evaluate_mvtec.py --data-root /content/mvtec_ad --output-dir /content/drive/MyDrive/defect_detection_project/artifacts
```

The script supports:

- MVTec dataset loading from `train/good`, `test`, and `ground_truth`
- Memory bank creation from normal training images
- Image AUROC calculation
- Pixel AUROC calculation
- Precision, recall, and F1 calculation
- Threshold comparison across P80, P85, P90, and P95
- Max, top-1%, top-5%, and P95-mean image scoring

## Recommended Final Report Structure

1. Abstract
2. Introduction
3. Problem statement
4. Literature survey
5. Dataset description
6. Proposed methodology
7. System architecture
8. Algorithm and mathematical formulation
9. Implementation details
10. Results and evaluation
11. Ablation study
12. Web application screenshots
13. Limitations
14. Future scope
15. Conclusion
16. References

## Final-Year Project Positioning

Use this project title:

**VisionGuard: Unsupervised Industrial Defect Detection and Zero-Shot Defect Description using Deep Patch Embeddings and Vision-Language Models**

This sounds more research-oriented and better reflects the actual technical contribution.
