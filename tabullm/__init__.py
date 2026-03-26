"""
TabuLLM: Feature Extraction and Interpretation of Text Columns in Tabular Data Using LLMs

A Python package for seamless integration of text embeddings into tabular ML pipelines,
with tools for clustering and LLM-based interpretation.
"""

__version__ = "1.1.0"

# Core components
from .embed import TextColumnTransformer
from .cluster import SphericalKMeans, GMMFeatureExtractor
from .explain import ClusterExplainer

# Data utilities
from .data import load_fraud

__all__ = [
    "TextColumnTransformer",
    "SphericalKMeans",
    "GMMFeatureExtractor",
    "ClusterExplainer",
    "load_fraud",
    "__version__",
]
