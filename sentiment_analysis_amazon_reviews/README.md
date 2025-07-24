# 📌 Customer Segmentation using Gaussian Mixture Model (GMM)

## 🔍 Objective
Apply **Gaussian Mixture Models** for customer segmentation using **annual income** and **spending score**, aiming for meaningful real-world clusters.

## 🗂 Dataset
- **Mall_Customers.csv** (200 rows; raw)
- Features: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`

## 🛠 Workflow
1. Load and inspect raw data
2. Clean/transform columns (rename, drop nulls, encode gender)
3. Scale features using `StandardScaler`
4. Reduce dimensionality with `PCA`
5. Fit GMM for `k=2..10`
6. Evaluate using `Silhouette Score`
7. Visualize best clusters

## 📈 Best Result
- **Best k**: `X` (replace in README)
- **Silhouette Score**: `0.XXXX`

## 🚀 Usage
```bash
pip install -r requirements.txt
python gmm_segmentation.py

