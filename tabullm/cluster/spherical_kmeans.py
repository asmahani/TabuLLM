"""
Spherical K-Means clustering for normalized embeddings

Note: For normalized embeddings (typical for text), SphericalKMeans is
mathematically equivalent to sklearn's KMeans. For feature extraction,
use GMMFeatureExtractor (provides cluster-specific scaling and superior
performance in most cases).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning
import warnings


class SphericalKMeans(BaseEstimator, ClusterMixin):
    """
    Spherical K-Means clustering using cosine distance.

    Clusters data on unit hypersphere using cosine similarity.
    For L2-normalized embeddings (typical for text), mathematically equivalent
    to sklearn's KMeans. For feature extraction, consider GMMFeatureExtractor
    (provides cluster-specific scaling).

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form.

    n_init : int, default=10
        Number of random initializations. Best result kept.

    max_iter : int, default=300
        Maximum iterations per initialization.

    tol : float, default=1e-4
        Convergence tolerance (Frobenius norm of centroid changes).

    random_state : int or None, default=None
        Random seed for reproducibility.

    return_hard_labels : bool, default=False
        If True, transform() returns hard labels (n_samples, 1).
        If False, transform() returns similarities (n_samples, n_clusters).
        Note: Non-standard parameter (sklearn uses separate predict/transform).
        Consider using predict() for labels, transform() for features.

    verbose : int, default=0
        Verbosity level (0=silent, >0=progress messages).

    Attributes
    ----------
    cluster_centers_ : array, shape (n_clusters, n_features)
        Normalized cluster centroids.

    labels_ : array, shape (n_samples,)
        Cluster labels from fit().

    inertia_ : float
        Sum of distances to nearest cluster.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> from tabullm.cluster import SphericalKMeans
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> skm = SphericalKMeans(n_clusters=5, random_state=42)
    >>> labels = skm.fit_predict(X)

    Notes
    -----
    For normalized text embeddings, SphericalKMeans ≈ sklearn.cluster.KMeans
    (cosine distance ≈ Euclidean for normalized vectors).
    """

    def __init__(self, n_clusters=3, n_init=10, max_iter=300, tol=1e-4,
                 random_state=None, return_hard_labels=False, verbose=0
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.return_hard_labels = return_hard_labels
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Fit spherical k-means clustering to data and optionally calculate consistency.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object or dict
            If `return_consistency` is False, returns the fitted estimator.
            If `return_consistency` is True, returns a dictionary with average consistency for each `n`.
        """
        X = check_array(X)
        X = self._normalize(X)
        X_unique = np.unique(X, axis=0)
        random_state = check_random_state(self.random_state)

        # Track best inertia and clustering solution per initialization
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None

        for init_num in range(self.n_init):
            if self.verbose:
                print(f"Initialization {init_num + 1}/{self.n_init}")

            # Initialize centroids randomly
            centroids = self._initialize_centroids(X_unique, random_state)
            iter_count = 0

            for i in range(self.max_iter):
                iter_count += 1
                try:
                    centroids, labels, frob_norm = self._lloyd_iteration(X, centroids)
                except RuntimeError as e:
                    if self.verbose:
                        print(f"Re-initializing centroids due to: {e}")
                    centroids = self._initialize_centroids(X_unique, random_state)
                    iter_count = 0
                    continue

                if frob_norm <= self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iter_count}")
                    break
            else:
                warnings.warn(
                    f"Initialization {init_num + 1} did not converge within {self.max_iter} iterations.",
                    ConvergenceWarning
                )

            # Compute inertia
            similarities = np.dot(X, centroids.T)
            inertia = np.sum(1 - np.max(similarities, axis=1))

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        # Set the best results
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = iter_count

        return self

    def _initialize_centroids(self, X, random_state):
        indices = random_state.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[indices]
        return centroids

    def _normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero norm encountered in input data; cannot normalize.")
        return X / norms

    def _lloyd_iteration(self, X, centroids):
        # Compute similarities
        similarities = np.dot(X, centroids.T)
        # Assign labels
        labels = np.argmax(similarities, axis=1)
        # Recompute centroids
        new_centroids = np.zeros_like(centroids)
        for idx in range(self.n_clusters):
            mask = labels == idx
            if np.any(mask):
                cluster_points = X[mask]
                centroid = np.sum(cluster_points, axis=0)
                norm = np.linalg.norm(centroid)
                if norm == 0:
                    # This should not happen with non-zero input vectors
                    raise RuntimeError(f"Zero norm encountered for centroid {idx}.")
                new_centroids[idx] = centroid / norm
            else:
                # Empty cluster encountered
                raise RuntimeError(f"Cluster {idx} is empty.")
        # Compute Frobenius norm
        frob_norm = np.linalg.norm(new_centroids - centroids)
        return new_centroids, labels, frob_norm

    def _assign_clusters(self, X, hard=True):
        # Calculate the cosine similarity between each point and each centroid
        similarities = np.dot(X, self.cluster_centers_.T)
        if hard:
            # Assign each point to the nearest centroid (highest similarity)
            return np.argmax(similarities, axis=1)
        return similarities

    def transform(self, X):
        """
        Transform X to a cluster-distance space or return hard labels based on `return_hard_labels`.

        In the new space, each dimension is the cosine similarity to the cluster centers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        similarities : ndarray of shape (n_samples, n_clusters) or labels : ndarray of shape (n_samples,)
            Transformed array of cosine similarities, or hard labels (cluster assignments).
        """
        check_is_fitted(self, 'cluster_centers_')
        X = check_array(X)
        X = self._normalize(X)

        if self.return_hard_labels:
            labels = self._assign_clusters(X, hard=True)
            return labels.reshape(-1, 1)
        else:
            similarities = self._assign_clusters(X, hard=False)
            return similarities

    def fit_transform(self, X, y=None):
        """
        Compute clustering and transform X to cluster-distance space or hard labels based on `return_hard_labels`.

        Equivalent to calling fit(X) followed by transform(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster and transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        similarities : ndarray of shape (n_samples, n_clusters) or labels : ndarray of shape (n_samples,)
            Transformed array of cosine similarities, or hard labels (cluster assignments).
        """
        self.fit(X, y)
        return self.transform(X)

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
        check_is_fitted(self, 'cluster_centers_')
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
        self.fit(X, y)
        return self.labels_


# Utilities no longer needed (consistency features removed)
