# 🛍️ Experiment 4: Advanced Customer Segmentation (GMM & PCA)

This project implements a production-grade Machine Learning pipeline to identify **hidden behavioral segments** in multi-dimensional customer data. Updated in 2026 to include modular architecture and advanced soft-clustering.

## 🚀 Key Highlights (As per Resume)
* **Advanced Soft-Clustering:** Utilized **Gaussian Mixture Models (GMM)** to capture overlapping customer behaviors through probabilistic membership.
* **Dimensionality Reduction:** Implemented **PCA** to reduce 3D feature sets (Age, Income, Spending) into a 2D space, isolating primary behavioral drivers.
* **Mathematical Validation:** Optimized via **Silhouette Scores** and **Davies-Bouldin Index** to ensure maximum cluster separation and density.

---

## 📊 Technical Performance & Metrics
In Unsupervised Learning, "Accuracy" is replaced by **Spatial Density** and **Separation Quality**.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Optimal Clusters (k)** | **2** | Determined via automated Silhouette analysis loop. |
| **Silhouette Score** | **0.4172** | Indicates strong separation (Range -1 to 1). |
| **Davies-Bouldin Index** | **0.9708** | Confirms low intra-cluster similarity (Lower is better). |
| **Mean Certainty Score** | **96.9%** | Average probability of GMM assignment confidence. |

---

## 📈 Visual Validation & "Accuracy"
Since standard Confusion Matrices require ground-truth labels, we validate performance through **Segment Profiling**:

### 1. PCA Behavioral Space (Math View)
Ensures clear mathematical boundaries and no significant noise overlap between the identified Gaussian distributions.


### 2. Strategic Segment Profiles
Translating the mathematical clusters into actionable business segments:

* **Segment 0 (The Mature Savers):**
    * *Profile:* Avg Age ~47, High Income, Low Spending.
    * *Strategy:* Target with premium loyalty programs and long-term value reliability.
* **Segment 1 (The Young High-Rollers):**
    * *Profile:* Avg Age ~28, High Income, High Spending.
    * *Strategy:* Target with digital flash sales, fashion trends, and high-engagement social media campaigns.



---

## 🏗️ Modular Architecture
* `/src`: Decoupled logic for data loading, scaling (PCA), and the GMM engine.
* `main.py`: The central pipeline orchestrator.