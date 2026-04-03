from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

def optimize_clusters(data, max_k=10):
    best_score = -1
    results = []

    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(data)
        
        s_score = silhouette_score(data, labels)
        db_index = davies_bouldin_score(data, labels)
        
        results.append({'k': k, 's_score': s_score, 'db_index': db_index, 'model': gmm})
        
    # Sort by Silhouette Score to find the best
    best_result = max(results, key=lambda x: x['s_score'])
    return best_result['model'], best_result['k'], best_result['s_score'], best_result['db_index']