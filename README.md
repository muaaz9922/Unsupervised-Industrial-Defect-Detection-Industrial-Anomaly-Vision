# 🏭 Unsupervised Industrial Defect Detection

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

A robust anomaly detection pipeline designed for industrial quality control environments where defective data is scarce or nonexistent. By training exclusively on "good" (nominal) manufacturing samples, this system learns the standard distribution of the product and identifies structural or textural defects by measuring deviations in deep feature space.

---

## ✨ Key Features

* **Zero-Defect Training (Unsupervised):** Completely eliminates the need to collect and annotate thousands of defective samples. The model trains only on normal, healthy images.
* **Deep Feature Extraction:** Utilizes powerful pre-trained CNN backbones (`[e.g., ResNet-18 / WideResNet-50]`) to extract hierarchical visual features without requiring task-specific fine-tuning.
* **Pixel-Precise Localization:** Generates high-resolution spatial anomaly heatmaps that pinpoint the exact location, size, and severity of the manufacturing defect.
* **Algorithm Implementations:** Built using state-of-the-art unsupervised anomaly detection paradigms (`[e.g., PatchCore, PaDiM, or FastFlow]`).
* **Automated Thresholding:** Dynamically calculates the optimal anomaly score cutoff using validation distribution statistics (F1-Max or AUROC optimization).

---

## 🏗️ Pipeline Architecture

```text
                      +-------------------------+
                      | Defect-Free Training Set|
                      +------------+------------+
                                   |
                                   v
                      +-------------------------+
                      | Pre-trained Backbone    |
                      | (Feature Extraction)    |
                      +------------+------------+
                                   |
                                   v
                      +-------------------------+
                      | Nominal Memory Bank /   |
                      | Multivariate Gaussians  |
                      +-------------------------+
                                   |
[During Inference]                 | Distance / Likelihood Metric
                                   v
+------------------+  +-------------------------+
| New Test Image   |->| Anomaly Scoring Engine  |
| (Pass or Fail?)  |  | (Global Score + Heatmap)|
+------------------+  +------------+------------+
                                   |
                                   v
                      +-------------------------+
                      | Visual Defect Overlay   |
                      | & Quality Control Alert |
                      +-------------------------+
