"""
Gaussian Mixture Model feature extraction for embeddings.

Provides GMMFeatureExtractor, which adds transform() to sklearn's GaussianMixture
for feature extraction. Returns per-cluster log-joint features log p(x, c_k),
the quantity the GMM maximises for hard assignment.

Key contribution: sklearn's GaussianMixture lacks transform() - only has predict()
and predict_proba(). This fills that gap for ML pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted


class GMMFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract features using Gaussian Mixture Model log-joint probabilities.

    ``transform()`` returns an (n_samples, K) array of per-cluster log-joints

        ℓ_k(x) = log p(x, c_k) = log π_k + log p(x | c_k)

    where log p(x | c_k) is the Gaussian log-likelihood under component k.
    This is the quantity the GMM maximises for hard assignment
    (k* = argmax_k ℓ_k), so features are in exact correspondence with the
    model's own criterion.  Posterior probabilities are a softmax of these
    features.

    Adds transform() method that sklearn's GaussianMixture lacks, enabling
    use in feature extraction pipelines.

    Parameters
    ----------
    n_components : int, default=10
        Number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        Type of covariance parameters.

    n_init : int, default=3
        Number of initializations to perform.

    random_state : int, default=None
        Random seed for reproducibility.

    include_log_density : bool, default=False
        If True, append log p(x) = log Σ_k exp(ℓ_k(x)) as a (K+1)-th column.
        This is the log marginal likelihood — how well the overall mixture
        explains x.  It is a deterministic function of the K log-joints
        (log-sum-exp), so it adds no information for expressive nonlinear
        models but provides an explicit outlier score for linear ones.

    **gmm_kwargs
        Additional parameters for GaussianMixture.

    Attributes
    ----------
    gmm_ : GaussianMixture
        Fitted Gaussian Mixture Model.

    means_ : array, shape (n_components, n_features)
        Component means.

    covariances_ : array
        Component covariances.

    labels_ : array, shape (n_samples,)
        Cluster labels for each sample from the training set.

    Examples
    --------
    >>> from tabullm import TextColumnTransformer, GMMFeatureExtractor
    >>> from sklearn.pipeline import Pipeline
    >>>
    >>> pipeline = Pipeline([
    ...     ('embed', TextColumnTransformer(model=model, text_columns=['text'])),
    ...     ('gmm', GMMFeatureExtractor(n_components=10)),
    ...     ('clf', RandomForestClassifier())
    ... ])
    """

    def __init__(self, n_components=10, covariance_type='full',
                 n_init=3, random_state=None, include_log_density=False,
                 **gmm_kwargs):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state
        self.include_log_density = include_log_density
        self.gmm_kwargs = gmm_kwargs

    def fit(self, X, y=None):
        """Fit GMM to data."""
        X = check_array(X)

        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            random_state=self.random_state,
            **self.gmm_kwargs
        )
        self.gmm_.fit(X)

        self.means_ = self.gmm_.means_
        self.covariances_ = self.gmm_.covariances_
        self.n_components_ = self.n_components
        self.labels_ = self.gmm_.predict(X)

        return self

    def transform(self, X):
        """
        Transform X to per-cluster log-joint features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, K) or (n_samples, K+1)
            Per-cluster log p(x, c_k) for k = 0 … K-1, plus log p(x) as
            the final column when ``include_log_density=True``.
        """
        check_is_fitted(self, 'gmm_')
        X = check_array(X)
        log_joints = self.gmm_._estimate_weighted_log_prob(X)   # (n_samples, K)
        if self.include_log_density:
            log_density = self.gmm_.score_samples(X).reshape(-1, 1)
            return np.hstack([log_joints, log_density])
        return log_joints

    def get_feature_names_out(self, input_features=None):
        """
        Feature names: ``lj_0 … lj_{K-1}``, plus ``log_density`` when
        ``include_log_density=True``.
        """
        check_is_fitted(self, 'gmm_')
        names = [f'lj_{k}' for k in range(self.n_components_)]
        if self.include_log_density:
            names.append('log_density')
        return np.array(names)

    def predict(self, X):
        """Predict cluster labels."""
        check_is_fitted(self, 'gmm_')
        return self.gmm_.predict(check_array(X))

    def assignment_confidence_stats(self, X):
        """
        Compute per-observation assignment confidence statistics.

        Returns four metrics that characterise how confidently and
        unambiguously the fitted GMM assigns each observation to a cluster,
        plus the marginal log-density as an outlier score.
        All four are scalars per observation and can be aggregated to
        cluster-level summaries via ``groupby(cluster_labels).mean()``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score. Must have the same number of features as the
            training data.

        Returns
        -------
        pd.DataFrame of shape (n_samples, 4)
            Columns:

            ``max_posterior``
                Maximum posterior probability :math:`\\max_k p(k \\mid x)`.
                Range :math:`(1/K, 1]`. Higher means the model is more
                certain about the cluster assignment.

            ``entropy``
                Shannon entropy :math:`-\\sum_k p(k \\mid x) \\log p(k \\mid x)`
                of the posterior distribution.
                Range :math:`[0, \\log K]`. Lower means the probability mass
                is concentrated on fewer clusters.

            ``log_joint_margin``
                Difference between the top-1 and top-2 per-cluster log joints
                :math:`\\ell_{k^*}(x) - \\max_{j \\neq k^*} \\ell_j(x)`.
                Range :math:`[0, \\infty)`. Larger means the assigned cluster
                is more decisively preferred over its nearest rival.

            ``log_density``
                Log marginal likelihood :math:`\\log p(x) = \\log \\sum_k e^{\\ell_k(x)}`.
                Range :math:`(-\\infty, 0]`. Captures how well the overall GMM
                explains this observation; large negative values flag outliers.
                Orthogonal to the three assignment metrics above: two observations
                can share the same ``max_posterior`` yet have very different
                ``log_density``.

        Examples
        --------
        >>> stats = gmm.assignment_confidence_stats(X_embeddings)
        >>> stats.describe()

        >>> # Per-cluster rollup
        >>> stats.groupby(cluster_labels).mean()

        >>> # Feed into ClusterExplainer
        >>> explainer.explain(X, cluster_labels, y=y, observation_stats=stats)
        """
        check_is_fitted(self, 'gmm_')
        X = check_array(X)

        # (n_samples, K) posterior probabilities
        posteriors = self.gmm_.predict_proba(X)

        # 1. Assignment confidence: max posterior per observation
        assignment_confidence = posteriors.max(axis=1)

        # 2. Entropy: -sum_k p(k|x) log p(k|x)
        log_posteriors = np.log(np.clip(posteriors, 1e-10, 1.0))
        entropy = -(posteriors * log_posteriors).sum(axis=1)

        # 3. Log-joint margin: top-1 minus top-2 log joint
        log_joints = self.gmm_._estimate_weighted_log_prob(X)  # (n_samples, K)
        sorted_lj = np.sort(log_joints, axis=1)[:, ::-1]
        log_joint_margin = sorted_lj[:, 0] - sorted_lj[:, 1]

        # 4. Log density: log p(x) = log-sum-exp of log joints
        log_density = self.gmm_.score_samples(X)

        return pd.DataFrame({
            'max_posterior': assignment_confidence,
            'entropy': entropy,
            'log_joint_margin': log_joint_margin,
            'log_density': log_density,
        })
