import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load raw dataset
df = pd.read_csv("Mall_Customers.csv")

# Quick look at the first rows
print("Raw data preview:")
print(df.head())

# Feature engineering and cleanup:
#   - Rename columns for consistency
#   - Drop rows with nulls (if any)
df = df.rename(columns={
    'Annual Income (k$)': 'AnnualIncome',
    'Spending Score (1-100)': 'SpendingScore'
})
df.dropna(inplace=True)

# Optional: Convert Gender to numeric if needed
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Select features
features = ['AnnualIncome', 'SpendingScore']
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for noise reduction (optional but helpful)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Evaluate GMM cluster quality
best_score, best_k = -1, None
best_labels = None

print("🔍 GMM Silhouette Scores")
for k in range(2, 11):
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f" k = {k}, score = {score:.4f}")
    if score > best_score:
        best_score, best_k, best_labels = score, k, labels

print(f"\n✅ Best GMM: k={best_k}, Silhouette Score={best_score:.4f}")

# Store cluster labels and visualise
df['GMM_Cluster'] = best_labels
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x='AnnualIncome', y='SpendingScore',
    hue='GMM_Cluster', palette='Set2', s=100
)
plt.title(f"GMM Customer Segments (k={best_k})")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

