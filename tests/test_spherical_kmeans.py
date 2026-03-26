"""
Tests for SphericalKMeans (simplified version without ensemble features)
"""

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from tabullm.cluster import SphericalKMeans


@pytest.fixture
def sample_data():
    """Generate normalized sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(100, 20)
    # Normalize (spherical k-means expects normalized data)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = np.random.randint(0, 2, 100)
    return X, y


class TestSphericalKMeansBasic:
    """Basic functionality tests"""

    def test_initialization(self):
        """Test initialization with default parameters"""
        skm = SphericalKMeans(n_clusters=5)
        assert skm.n_clusters == 5
        assert skm.n_init == 10
        assert skm.max_iter == 300

    def test_fit(self, sample_data):
        """Test fit method"""
        X, _ = sample_data
        skm = SphericalKMeans(n_clusters=3, random_state=42)
        result = skm.fit(X)

        # Returns self
        assert result is skm

        # Sets fitted attributes
        assert hasattr(skm, 'cluster_centers_')
        assert hasattr(skm, 'labels_')
        assert hasattr(skm, 'inertia_')

        # Correct shapes
        assert skm.cluster_centers_.shape == (3, 20)
        assert skm.labels_.shape == (100,)

    def test_predict(self, sample_data):
        """Test predict method"""
        X, _ = sample_data
        skm = SphericalKMeans(n_clusters=3, random_state=42)
        skm.fit(X)

        labels = skm.predict(X)

        assert labels.shape == (100,)
        assert np.all(labels >= 0)
        assert np.all(labels < 3)

    def test_fit_predict(self, sample_data):
        """Test fit_predict convenience method"""
        X, _ = sample_data
        skm = SphericalKMeans(n_clusters=3, random_state=42)

        labels = skm.fit_predict(X)

        assert labels.shape == (100,)
        # Should be same as fit().predict()
        skm2 = SphericalKMeans(n_clusters=3, random_state=42)
        np.testing.assert_array_equal(labels, skm2.fit(X).predict(X))

    def test_transform(self, sample_data):
        """Test transform returns similarities"""
        X, _ = sample_data
        skm = SphericalKMeans(n_clusters=3, random_state=42, return_hard_labels=False)
        skm.fit(X)

        X_transformed = skm.transform(X)

        # Shape: (n_samples, n_clusters)
        assert X_transformed.shape == (100, 3)

        # Cosine similarities (between -1 and 1, but typically 0-1 for good clustering)
        assert np.all(X_transformed >= -1)
        assert np.all(X_transformed <= 1)

    def test_reproducibility(self, sample_data):
        """Test random_state ensures reproducibility"""
        X, _ = sample_data

        skm1 = SphericalKMeans(n_clusters=3, random_state=42)
        skm2 = SphericalKMeans(n_clusters=3, random_state=42)

        labels1 = skm1.fit_predict(X)
        labels2 = skm2.fit_predict(X)

        np.testing.assert_array_equal(labels1, labels2)

    def test_pipeline_integration(self, sample_data):
        """Test SphericalKMeans in sklearn Pipeline"""
        X, y = sample_data

        pipeline = Pipeline([
            ('cluster', SphericalKMeans(n_clusters=3, random_state=42)),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])

        # Fit and predict
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert predictions.shape == (100,)

    def test_normalization_required(self):
        """Test that data gets normalized internally"""
        # Unnormalized data
        X = np.random.randn(50, 10)

        skm = SphericalKMeans(n_clusters=3, random_state=42)
        skm.fit(X)  # Should normalize internally

        # Centroids should be normalized
        norms = np.linalg.norm(skm.cluster_centers_, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3))

    def test_convergence(self, sample_data):
        """Test that algorithm converges"""
        X, _ = sample_data

        skm = SphericalKMeans(n_clusters=3, max_iter=100, random_state=42, verbose=0)
        skm.fit(X)

        # Should have converged (n_iter < max_iter for this data)
        assert skm.n_iter_ < 100

    def test_return_hard_labels_true(self, sample_data):
        """Test return_hard_labels=True in transform"""
        X, _ = sample_data
        skm = SphericalKMeans(n_clusters=3, random_state=42, return_hard_labels=True)
        skm.fit(X)

        X_transformed = skm.transform(X)

        # Should return hard labels (n_samples, 1)
        assert X_transformed.shape == (100, 1)
        assert np.all(X_transformed >= 0)
        assert np.all(X_transformed < 3)

    def test_return_hard_labels_false(self, sample_data):
        """Test return_hard_labels=False in transform (default)"""
        X, _ = sample_data
        skm = SphericalKMeans(n_clusters=3, random_state=42, return_hard_labels=False)
        skm.fit(X)

        X_transformed = skm.transform(X)

        # Should return similarities (n_samples, n_clusters)
        assert X_transformed.shape == (100, 3)
