from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataTransformer:
    def __init__(self, n_components=2):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

    def prepare_data(self, df):
        features = ['Age', 'AnnualIncome', 'SpendingScore']
        # Standardize (Resume point: Standardized high-dimensional datasets)
        X_scaled = self.scaler.fit_transform(df[features])
        # PCA (Resume point: PCA for advanced soft-clustering)
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca