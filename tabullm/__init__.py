# Copyright (c) 2024 Alireza S. Mahani and Mansour T.A. Sharabiani
# Licensed under the MIT License. See LICENSE file in the project root.

"""
TabuLLM: Feature Extraction and Interpretation of Text Columns in Tabular Data Using LLMs

A Python package for seamless integration of text embeddings into tabular ML pipelines,
with tools for clustering and LLM-based interpretation.
"""

__version__ = "1.2.1"

# Core components
from .embed import TextColumnTransformer
from .cluster import GMMFeatureExtractor
from .explain import ClusterExplainer

# Data utilities
from .data import load_fraud

__all__ = [
    "TextColumnTransformer",
    "GMMFeatureExtractor",
    "ClusterExplainer",
    "load_fraud",
    "__version__",
]