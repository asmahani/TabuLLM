import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import check_random_state

def _skmeans_lloyd_update(
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

class SphericalKMeans(BaseEstimator, ClusterMixin):
    """
    Spherical K-Means clustering algorithm.
    
    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        
    Attributes
    ----------
    cluster_centers_ : array, shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : array, shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.
    similarity_matrix : array, shape (n_samples, n_clusters)
        Similarity matrix between samples and cluster centers.
    """
    def __init__(self, n_clusters=3, n_init=10, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.inertia_ = None

    def fit(self, X, y=None):
        """
        Fit spherical k-means clustering to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """        
        X = check_array(X)
        X = self._normalize(X)
        X_unique = np.unique(X, axis = 0)
        
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        best_simiarities = None
        random_state = check_random_state(self.random_state)

        for _ in range(self.n_init):
            # Initialize centroids
            centroids = self._initialize_centroids(X_unique, random_state)
            similarities = np.dot(X, centroids.T)  # Initial similarities
            
            iter_count = 0
            while iter_count < self.max_iter:
                try:
                    # Update centroids and calculate inertia using optimized Lloyd update
                    similarities, labels, centroids, frob_norm = _skmeans_lloyd_update(X, centroids, similarities)
                    
                    # Check for convergence
                    if frob_norm <= self.tol:
                        break
                    
                    iter_count += 1
                except Exception as e:
                    # if centroids are empty, reset and continue
                    centroids = self._initialize_centroids(X_unique, random_state)
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
        self.n_iter_ = iter_count

        return self

    def _initialize_centroids(self, X, random_state):
        indices = random_state.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / norms

    def _assign_clusters(self, X, hard=True):
        # Calculate the cosine similarity between each point and each centroid
        similarities = np.dot(X, self.cluster_centers_.T)
        if hard:
            # Assign each point to the nearest centroid (highest similarity)
            return np.argmax(similarities, axis=1)
        return similarities
    
    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        In the new space, each dimension is the cosine similarity to the cluster centers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        similarities : ndarray of shape (n_samples, n_clusters)
            Transformed array.
        """
        check_is_fitted(self, ['cluster_centers_'])
        X = check_array(X)
        X = self._normalize(X)
        return self._assign_clusters(X, hard=False)
    
    def fit_transform(self, X, y=None):
        """
        Compute clustering and transform X to cluster-distance space.

        Equivalent to calling fit(X) followed by transform(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster and transform.

        Returns
        -------
        similarities : ndarray of shape (n_samples, n_clusters)
            Transformed array.
        """
        self.fit(X, y)
        return self.similarity_matrix

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, ['cluster_centers_'])
        X = check_array(X)
        X = self._normalize(X)
        return self._assign_clusters(X, hard=True)

    def fit_predict(self, X, y=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to fit and predict.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_

