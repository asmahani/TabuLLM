# Copyright (c) 2024 Alireza S. Mahani and Mansour T.A. Sharabiani
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Clustering and feature extraction for text embeddings

This module provides GMM-based feature extraction optimized for text embeddings.

Classes
-------
GMMFeatureExtractor : Transform embeddings to per-cluster log-joint features for ML pipelines

Examples
--------
>>> from tabullm.cluster import GMMFeatureExtractor
>>> from sklearn.pipeline import Pipeline
>>>
>>> # GMM feature extraction in pipeline
>>> pipeline = Pipeline([
...     ('gmm', GMMFeatureExtractor(n_components=10)),
...     ('clf', RandomForestClassifier())
... ])
"""

# Cluster submodule exports
from .gmm_features import GMMFeatureExtractor

__all__ = ['GMMFeatureExtractor']
