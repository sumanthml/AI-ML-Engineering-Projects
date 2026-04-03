# 🛍️ Advanced Customer Segmentation: GMM & PCA Pipeline

This project implements a production-grade Machine Learning pipeline to identify **hidden behavioral segments** within multi-dimensional customer data. By leveraging **Gaussian Mixture Models (GMM)** for soft-clustering and **PCA** for noise reduction, the system transforms raw data into actionable business strategies.

## 🚀 Key Highlights (As per Resume)
* **Advanced Soft-Clustering:** Utilized **GMM** to capture overlapping customer behaviors through probabilistic membership.
* **Dimensionality Reduction:** Implemented **PCA** to reduce 3D feature sets (Age, Income, Spending) into a 2D behavioral space, significantly improving cluster density.
* **Mathematical Validation:** Optimized the model using **Silhouette Scores** and the **Davies-Bouldin Index** to ensure maximum cluster separation.
* **Modular Architecture:** Engineered with a decoupled `src/` directory structure for high maintainability and scalability.

---

## 🏗️ Project Architecture
The project follows a modular "Clean Code" structure:
* `data_loader.py`: Handles ingestion, header cleaning, and null-value management.
* `processor.py`: Implements **StandardScaler** and **PCA** transformations.
* `model_engine.py`: Core logic for GMM optimization and internal metrics calculation.
* `main.py`: The central orchestrator for the entire pipeline.

---

## 📊 Technical Performance & Metrics

Unlike supervised models, clustering performance is measured by **Spatial Density** and **Separation**.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Optimal Clusters (k)** | **2** | Determined via Silhouette analysis loop (k=2 to 10). |
| **Silhouette Score** | **0.4172** | Indicates strong separation between segments (Range -1 to 1). |
| **Davies-Bouldin Index** | **0.9708** | Confirms low intra-cluster similarity (Lower is better). |
| **Mean Certainty Score** | **96.9%** | Average probability of GMM assignment confidence. |

---

## 📈 Visualizing "Accuracy" in Clustering
In Unsupervised Learning, we validate "Accuracy" through **Segment Profiling** and **Feature Distribution**.

### 1. PCA Behavioral Space (Math View)
Visualizes the clusters in the reduced feature space to ensure clear mathematical boundaries and no significant noise overlap.


### 2. Business Strategy View (Income vs Spending)
Maps the mathematical clusters back to real-world variables (Annual Income vs Spending Score) to visualize the "Strategic Segments."

### 3. Feature Distribution (Boxplots)
Validates the **Hidden Behaviors** by showing the variance of Spending Scores across segments, proving the model found distinct spending archetypes.


---

## 💡 Strategic Business Segments
The pipeline successfully identified two primary archetypes for targeted marketing:

### **Segment 0: The Mature Savers**
* **Profile:** Avg Age ~47, High Income, Low Spending.
* **Strategy:** These are established adults who are conservative with money. **Target:** Focus on long-term value, insurance, and premium reliability.

### **Segment 1: The Young High-Rollers**
* **Profile:** Avg Age ~28, High Income, High Spending.
* **Strategy:** These are young professionals with high disposable income. **Target:** Focus on trend-based digital marketing, fashion, and flash sales.

---

## 🛠️ Setup & Execution
1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt