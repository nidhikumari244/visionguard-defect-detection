# 🔍 VisionGuard — AI Industrial Defect Detection

> Unsupervised anomaly detection + zero-shot defect description using vision-language models

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B/32-purple)](https://openai.com/research/clip)
[![MVTec AD](https://img.shields.io/badge/Dataset-MVTec%20AD-green)](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## 🎯 What is VisionGuard?

VisionGuard is an AI-powered industrial quality inspection system that:
- **Trains only on normal/good product images** — zero defect labels needed
- **Detects any defect type** — including ones never seen before
- **Localizes defects** with pixel-level anomaly heatmaps
- **Describes defects** in plain English using CLIP vision-language model

## 🏗️ Architecture

Input Image → Wide ResNet-50 → Memory Bank → kNN Scoring → Anomaly Heatmap → CLIP Description → Inspection Report

### Pipeline Steps
1. **Feature Extraction** — Wide ResNet-50 extracts 768-dim patch features
2. **Memory Bank** — 10,000 representative normal patch vectors per category
3. **Anomaly Scoring** — Nearest-neighbor Euclidean distance from normal features
4. **Heatmap Generation** — Per-patch distance map overlaid as color heatmap
5. **CLIP Description** — Zero-shot defect description using OpenAI CLIP ViT-B/32

## 📊 Results — MVTec AD Benchmark

| Category | Image AUROC | Pixel AUROC | Precision | Recall | F1 Score |
|----------|:-----------:|:-----------:|:---------:|:------:|:--------:|
| 🍾 Bottle | 0.929 | 0.954 | 0.962 | 0.794 | 0.870 |
| 🔌 Cable | 0.778 | 0.915 | 0.885 | 0.587 | 0.706 |
| 🟫 Carpet | 0.939 | 0.958 | 0.951 | 0.876 | 0.912 |
| 🪵 Wood | 0.983 | 0.923 | 0.937 | 0.983 | 0.959 |
| 🔩 Screw | 0.544 | 0.970 | 0.783 | 0.151 | 0.254 |

## 🔬 Ablation Study

| Experiment | Before | After | Improvement |
|------------|--------|-------|-------------|
| Single → Combined layers (bottle) | 0.856 | 0.975 | +13.8% |
| Unnormalized → Normalized (cable) | 0.620 | 0.778 | +25.5% |
| Threshold P95 → P85 (cable F1) | 0.390 | 0.706 | +81.0% |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| Deep Learning | PyTorch |
| Feature Extraction | Wide ResNet-50 (timm) |
| Vision-Language | CLIP ViT-B/32 (OpenAI) |
| Similarity Search | scikit-learn NearestNeighbors |
| Computer Vision | OpenCV |
| Web Interface | Gradio |
| Dataset | MVTec AD |
| Environment | Google Colab T4 GPU |

## 🚀 Quick Start

1. Clone the repo
   git clone https://github.com/nidhikumari244/visionguard-defect-detection.git

2. Install dependencies
   pip install -r requirements.txt

3. Open VisionGuard_Final.ipynb in Google Colab and run all cells

## 🔮 Future Work

- Top-k average pooling to improve screw image AUROC
- Category-adaptive normalization strategy
- Extend to all 15 MVTec AD categories
- Autoencoder baseline comparison
- Real-time API deployment for inline factory inspection

## 📄 References

- MVTec AD: Bergmann et al., 2019
- PatchCore: Roth et al., 2022 — https://arxiv.org/abs/2106.08265
- CLIP: Radford et al., 2021 — https://arxiv.org/abs/2103.00020

## 🎓 Project Context

B.Tech CSE (4th Year) Internship Project
Topic: Machine Learning & Deep Learning for Image Processing

---
Built with PyTorch · CLIP · MVTec AD
