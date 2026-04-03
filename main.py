import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_and_clean_data
from src.processor import DataTransformer
from src.model_engine import optimize_clusters

def run_segmentation():
    # 1. DATA INGESTION & CLEANING
    # This step triggers your data_loader module
    df = load_and_clean_data("data/Mall_Customers.csv")
    if df is None:
        print("❌ Pipeline stopped: Dataset not found or corrupted.")
        return

    # 2. FEATURE ENGINEERING & PCA TRANSFORMATION
    # We use Age, AnnualIncome, and SpendingScore for a multi-dimensional view
    transformer = DataTransformer(n_components=2)
    X_pca = transformer.prepare_data(df)
    
    # 3. UNSUPERVISED MODEL OPTIMIZATION
    # Finds the best 'k' using Silhouette Score and Davies-Bouldin Index
    model, k, s_score, db_index = optimize_clusters(X_pca)
    
    # 4. CLUSTER ASSIGNMENT & SOFT-CLUSTERING PROBABILITIES
    # GMM provides a probability distribution for each customer
    df['Cluster'] = model.predict(X_pca)
    probs = model.predict_proba(X_pca)
    df['Certainty_Score'] = probs.max(axis=1)

    # 5. TECHNICAL PERFORMANCE METRICS (The 'Accuracy' of Clustering)
    print("\n" + "="*30)
    print("📊 ADVANCED CLUSTERING METRICS")
    print("="*30)
    print(f"Optimal Segments (k)         : {k}")
    print(f"Silhouette Score (Separation) : {s_score:.4f}  (Range -1 to 1 | Higher is better)")
    print(f"Davies-Bouldin Index         : {db_index:.4f}  (Lower value = Better separation)")
    print("="*30)

    # 6. BUSINESS INTELLIGENCE: STRATEGIC SEGMENT PROFILES
    # This translates raw data into the 'Targeted Business Strategies' on your resume
    print("\n🔍 STRATEGIC SEGMENT BEHAVIORAL PROFILES (Averages):")
    profile = df.groupby('Cluster')[['Age', 'AnnualIncome', 'SpendingScore', 'Certainty_Score']].mean()
    print(profile)

    # 7. MULTI-VIEW VISUALIZATION
    plt.figure(figsize=(16, 6))
    
    # View A: PCA Behavioral Space (Math View)
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis', s=100, alpha=0.7)
    plt.title(f"PCA Reduced Segments (k={k})")
    plt.xlabel("PC1 (Variance)")
    plt.ylabel("PC2 (Variance)")

    # View B: Business Strategy View (Income vs Spending)
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='viridis', s=100)
    plt.title("Strategic View: Income vs Spending")
    plt.grid(True, linestyle='--', alpha=0.6)

    # View C: Segment Quality (Boxplots for Spending Behavior)
    plt.subplot(1, 3, 3)
    sns.boxplot(x='Cluster', y='SpendingScore', data=df, palette='viridis')
    plt.title("Cluster Spending Density")

    plt.tight_layout()
    print("\n📈 Rendering Visualizations...")
    plt.show()

if __name__ == "__main__":
    run_segmentation()