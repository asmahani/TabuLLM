"""
Tests for GMMFeatureExtractor
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from tabullm.cluster import GMMFeatureExtractor


class TestGMMFeatureExtractor:
    """Test suite for GMMFeatureExtractor"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        np.random.seed(42)
        X = np.random.randn(200, 20)  # 200 samples, 20 features
        y = np.random.randint(0, 2, 200)  # Binary target
        return X, y

    def test_initialization(self):
        """Test GMMFeatureExtractor initialization"""
        gmm = GMMFeatureExtractor(n_components=5, random_state=42)

        assert gmm.n_components == 5
        assert gmm.covariance_type == 'full'
        assert gmm.n_init == 3
        assert gmm.random_state == 42

    def test_fit(self, sample_data):
        """Test fit method"""
        X, _ = sample_data
        gmm = GMMFeatureExtractor(n_components=5, random_state=42)

        result = gmm.fit(X)

        # Should return self
        assert result is gmm

        # Should set fitted attributes
        assert hasattr(gmm, 'gmm_')
        assert hasattr(gmm, 'means_')
        assert hasattr(gmm, 'covariances_')
        assert hasattr(gmm, 'n_components_')

        # Check shapes
        assert gmm.means_.shape == (5, 20)
        assert gmm.n_components_ == 5

    def test_transform(self, sample_data):
        """Test transform method"""
        X, _ = sample_data
        gmm = GMMFeatureExtractor(n_components=5, random_state=42)
        gmm.fit(X)

        X_transformed = gmm.transform(X)

        # Check output shape
        assert X_transformed.shape == (200, 5)  # (n_samples, n_components)

        # Values differ across clusters (not all equal)
        assert not np.all(X_transformed == X_transformed[:, [0]])

        # argmax of log-joints must match hard cluster assignments
        assert np.array_equal(X_transformed.argmax(axis=1), gmm.predict(X))

    def test_fit_transform(self, sample_data):
        """Test fit_transform method"""
        X, _ = sample_data
        gmm = GMMFeatureExtractor(n_components=5, random_state=42)

        X_transformed = gmm.fit_transform(X)

        # Should have same result as fit().transform()
        gmm2 = GMMFeatureExtractor(n_components=5, random_state=42)
        X_transformed2 = gmm2.fit(X).transform(X)

        np.testing.assert_array_almost_equal(X_transformed, X_transformed2)

    def test_predict(self, sample_data):
        """Test predict method"""
        X, _ = sample_data
        gmm = GMMFeatureExtractor(n_components=5, random_state=42)
        gmm.fit(X)

        labels = gmm.predict(X)

        # Check output shape
        assert labels.shape == (200,)

        # Check labels are in valid range
        assert np.all(labels >= 0)
        assert np.all(labels < 5)

        # Check all clusters are used (with enough samples)
        unique_labels = np.unique(labels)
        assert len(unique_labels) > 1  # At least 2 clusters

    def test_pipeline_integration(self, sample_data):
        """Test GMMFeatureExtractor in sklearn Pipeline"""
        X, y = sample_data

        pipeline = Pipeline([
            ('gmm', GMMFeatureExtractor(n_components=5, random_state=42)),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])

        # Should fit without error
        pipeline.fit(X, y)

        # Should predict
        predictions = pipeline.predict(X)
        assert predictions.shape == (200,)

        # Should score
        score = pipeline.score(X, y)
        assert 0 <= score <= 1

    def test_cross_validation(self, sample_data):
        """Test GMMFeatureExtractor in cross-validation"""
        X, y = sample_data

        pipeline = Pipeline([
            ('gmm', GMMFeatureExtractor(n_components=3, random_state=42)),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])

        # Cross-validation should work (GMM refits on each fold)
        scores = cross_val_score(pipeline, X, y, cv=3)

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_different_covariance_types(self, sample_data):
        """Test different covariance types"""
        X, _ = sample_data

        for cov_type in ['full', 'tied', 'diag', 'spherical']:
            gmm = GMMFeatureExtractor(n_components=3, covariance_type=cov_type,
                                     random_state=42)
            gmm.fit(X)
            X_transformed = gmm.transform(X[:10])

            assert X_transformed.shape == (10, 3)

    def test_reproducibility(self, sample_data):
        """Test that random_state ensures reproducibility"""
        X, _ = sample_data

        gmm1 = GMMFeatureExtractor(n_components=5, random_state=42)
        gmm2 = GMMFeatureExtractor(n_components=5, random_state=42)

        X1 = gmm1.fit_transform(X)
        X2 = gmm2.fit_transform(X)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_include_log_density(self, sample_data):
        """include_log_density=True appends log p(x) as K+1-th column."""
        X, _ = sample_data
        gmm = GMMFeatureExtractor(n_components=5, random_state=42,
                                  include_log_density=True)
        gmm.fit(X)
        out = gmm.transform(X)
        assert out.shape == (200, 6)
        # Last column should equal logsumexp of the first K columns
        from scipy.special import logsumexp
        expected = logsumexp(out[:, :5], axis=1)
        np.testing.assert_allclose(out[:, -1], expected, rtol=1e-5)

    def test_include_log_density_false(self, sample_data):
        """include_log_density=False (default) returns K columns."""
        X, _ = sample_data
        gmm = GMMFeatureExtractor(n_components=5, random_state=42)
        gmm.fit(X)
        assert gmm.transform(X).shape == (200, 5)

    def test_get_feature_names_out(self, sample_data):
        """Feature names are lj_0…lj_{K-1}, plus log_density when enabled."""
        X, _ = sample_data
        gmm = GMMFeatureExtractor(n_components=3, random_state=42)
        gmm.fit(X)
        assert list(gmm.get_feature_names_out()) == ['lj_0', 'lj_1', 'lj_2']

        gmm_ld = GMMFeatureExtractor(n_components=3, random_state=42,
                                     include_log_density=True)
        gmm_ld.fit(X)
        assert list(gmm_ld.get_feature_names_out()) == [
            'lj_0', 'lj_1', 'lj_2', 'log_density'
        ]

    def test_unfitted_error(self, sample_data):
        """Test that transform fails on unfitted estimator"""
        X, _ = sample_data
        gmm = GMMFeatureExtractor(n_components=5)

        with pytest.raises(Exception):  # Should raise NotFittedError
            gmm.transform(X)

    def test_distances_vary_by_component(self, sample_data):
        """Test that different components give different distances"""
        X, _ = sample_data
        gmm = GMMFeatureExtractor(n_components=5, random_state=42)
        gmm.fit(X)

        X_transformed = gmm.transform(X[:10])

        # Distances to different components should vary
        # (not all the same)
        assert np.std(X_transformed, axis=1).mean() > 0


class TestAssignmentConfidenceStats:

    @pytest.fixture
    def fitted_gmm(self):
        np.random.seed(42)
        X = np.random.randn(200, 10)
        gmm = GMMFeatureExtractor(n_components=5, random_state=42)
        gmm.fit(X)
        return gmm, X

    def test_returns_dataframe(self, fitted_gmm):
        gmm, X = fitted_gmm
        result = gmm.assignment_confidence_stats(X)
        assert isinstance(result, pd.DataFrame)

    def test_shape(self, fitted_gmm):
        gmm, X = fitted_gmm
        result = gmm.assignment_confidence_stats(X)
        assert result.shape == (len(X), 4)

    def test_columns(self, fitted_gmm):
        gmm, X = fitted_gmm
        result = gmm.assignment_confidence_stats(X)
        assert list(result.columns) == ['max_posterior', 'entropy', 'log_joint_margin', 'log_density']

    def test_max_posterior_range(self, fitted_gmm):
        gmm, X = fitted_gmm
        col = gmm.assignment_confidence_stats(X)['max_posterior']
        assert col.between(0.0, 1.0).all()
        assert (col >= 1 / gmm.n_components).all()

    def test_entropy_range(self, fitted_gmm):
        gmm, X = fitted_gmm
        col = gmm.assignment_confidence_stats(X)['entropy']
        assert (col >= 0).all()
        assert (col <= np.log(gmm.n_components) + 1e-6).all()

    def test_log_joint_margin_non_negative(self, fitted_gmm):
        gmm, X = fitted_gmm
        col = gmm.assignment_confidence_stats(X)['log_joint_margin']
        assert (col >= 0).all()

    def test_unfitted_raises(self, fitted_gmm):
        _, X = fitted_gmm
        gmm = GMMFeatureExtractor(n_components=5)
        with pytest.raises(Exception):
            gmm.assignment_confidence_stats(X)

    def test_log_density_non_positive(self, fitted_gmm):
        gmm, X = fitted_gmm
        col = gmm.assignment_confidence_stats(X)['log_density']
        assert (col <= 0).all()

    def test_log_density_matches_score_samples(self, fitted_gmm):
        gmm, X = fitted_gmm
        col = gmm.assignment_confidence_stats(X)['log_density']
        np.testing.assert_allclose(col.values, gmm.gmm_.score_samples(X))

    def test_cluster_rollup_shape(self, fitted_gmm):
        gmm, X = fitted_gmm
        labels = gmm.predict(X)
        rollup = gmm.assignment_confidence_stats(X).groupby(labels).mean()
        assert rollup.shape == (gmm.n_components, 4)
