import numpy as np
from sklearn.utils.validation import check_array

def skmeans_lloyd_update(
        X
        , centroids
        , similarities
):
    n_obs = X.shape[0]

    new_labels = np.empty(n_obs, dtype = int)
    new_centroids = np.zeros_like(centroids)
    for n in range(n_obs):
        new_labels[n] = np.argmax(similarities[n, :])
        new_centroids[new_labels[n], :] += X[n, :]
    
    my_norms = np.linalg.norm(new_centroids, axis = 1).reshape(-1, 1)
    if np.any(my_norms == 0):
        raise RuntimeError('One or more clusters are empty')

    new_centroids = new_centroids / my_norms
    new_similarities = np.dot(X, new_centroids.T)

    frobenius_norm = np.sqrt(np.sum((new_centroids - centroids) ** 2))

    return new_similarities, new_labels, new_centroids, frobenius_norm

class SphericalKMeans:
    def __init__(self, n_clusters=3, n_init=10, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.inertia_ = None

    def fit(self, X, y=None):
        X = check_array(X)
        X = self._normalize(X)
        X_unique = np.unique(X, axis = 0)
        
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        best_simiarities = None

        for _ in range(self.n_init):
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            # Initialize centroids
            centroids = self._initialize_centroids(X_unique)
            similarities = np.dot(X, centroids.T)  # Initial similarities
            
            iter_count = 0
            while iter_count < self.max_iter:
                try:
                    # Update centroids and calculate inertia using optimized Lloyd update
                    similarities, labels, centroids, frob_norm = skmeans_lloyd_update(X, centroids, similarities)
                    
                    # Check for convergence
                    if frob_norm <= self.tol:
                        break
                    
                    iter_count += 1
                except Exception as e:
                    # if centroids are empty, reset and continue
                    centroids = self._initialize_centroids(X)
                    similarities = np.dot(X, centroids.T)
                    iter_count = 0  # Reset iteration count
                    continue

            inertia = np.sum(1 - np.max(similarities, axis = 1))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_simiarities = similarities

        # Set the best results
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.similarity_matrix = best_simiarities

        return self

    def _initialize_centroids(self, Xunique):
        indices = np.random.choice(Xunique.shape[0], self.n_clusters, replace=False)
        return Xunique[indices]

    def _normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / norms

    def _assign_clusters(self, X, hard=True):
        # Calculate the cosine similarity between each point and each centroid
        similarities = np.dot(X, self.centroids.T)
        if hard:
            # Assign each point to the nearest centroid (highest similarity)
            return np.argmax(similarities, axis=1)
        else:
            return similarities
    
    def transform(self, X):
        X = check_array(X)
        X = self._normalize(X)
        return self._assign_clusters(X, hard=False)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.similarity_matrix

    def predict(self, X):
        X = check_array(X)
        X = self._normalize(X)
        return self._assign_clusters(X, hard=True)

    def fit_predict(self, X, y = None):
        self.fit(X)
        return self.labels_

