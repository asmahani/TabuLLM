"""
Clustering and feature extraction for text embeddings

This module provides clustering algorithms optimized for text embeddings,
including spherical k-means (cosine distance) and GMM-based feature extraction.

Classes
-------
SphericalKMeans : Clustering with cosine distance for normalized embeddings
GMMFeatureExtractor : Transform embeddings to Mahalanobis distances for ML pipelines

Examples
--------
>>> from tabullm.cluster import SphericalKMeans, GMMFeatureExtractor
>>> from sklearn.pipeline import Pipeline
>>>
>>> # Spherical k-means clustering
>>> kmeans = SphericalKMeans(n_clusters=5)
>>> labels = kmeans.fit_predict(embeddings)
>>>
>>> # GMM feature extraction in pipeline
>>> pipeline = Pipeline([
...     ('gmm', GMMFeatureExtractor(n_components=10)),
...     ('clf', RandomForestClassifier())
... ])
"""

# Cluster submodule exports
from .spherical_kmeans import SphericalKMeans
from .gmm_features import GMMFeatureExtractor

__all__ = ['SphericalKMeans', 'GMMFeatureExtractor']
