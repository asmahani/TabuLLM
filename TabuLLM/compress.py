import copy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import KFold, RepeatedKFold

class CompressClassifier(BaseEstimator, ClassifierMixin):
    """
    Compressing a set of features - such as text embeddings - into a single feature,
    using K-Nearest-Neightbors classification, wrapped in cross-fit.

    Parameters
    ----------
    nx : int, optional
        Number of text embedding features to include in compression. Defaults to all features in X.
    ncv : int or cross-validation generator, default=5
        Number of cross-validation folds or a cross-validation generator. (If ncv < 2,
        no cross-fit is performed. This should only be used for experimentation.)
    logit : bool, default=True
        Whether to apply the logit transformation to predicted probabilities.
    laplace : bool, default=True
        Whether to apply Laplace smoothing to predicted probabilities.
    **kwargs : dict
        Additional parameters for KNeighborsClassifier.

    Attributes
    ----------
    trained_models : object
        Collection of fitted KNeighborsClassifier models for all folds.
    insample_prediction_proba : ndarray of shape (n_samples, 1)
        In-sample prediction probabilities.
    kfolds : object
        Cross-validation generator.
    """
    def __init__(self, nx=None, ncv=5, logit=True, laplace=True, **kwargs):
        super().__init__()
        self.knn = KNeighborsClassifier(**kwargs)
        self.nx = nx
        self.ncv = ncv
        self.logit = logit
        self.laplace = laplace

    def fit(self, X, y):
        """
        Fit the K-Nearest Neighbors classifier using cross-validation or a single fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        y = np.array(y)
        
        if not self.nx:
            self.nx = X.shape[1]
        
        if self.nx > X.shape[1]:
            raise ValueError('X has fewer columns than nx')
        
        X, y = check_X_y(X, y)
        if type_of_target(y) != 'binary':
            raise ValueError('Target type must be binary')
        
        # Select subset of columns and renormalize
        X = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x * x)), 1, X[:, :self.nx])

        if isinstance(self.ncv, int) and self.ncv < 2:
            # No cross-fitting
            self.trained_models = [copy.deepcopy(self.knn).fit(X, y)]
            insample_prediction_proba = self.trained_models[0].predict_proba(X)[:, 1]
            self.insample_prediction_proba = np.reshape(insample_prediction_proba, (insample_prediction_proba.size, 1))
            return self
        
        # Create folds
        if isinstance(self.ncv, (KFold, RepeatedKFold)):
            kf = self.ncv
        else:
            kf = KFold(n_splits=self.ncv, shuffle=True)
        
        kf.get_n_splits(X)
        self.kfolds = kf
        
        # Train model within each fold
        trained_models = []
        insample_prediction_proba = np.empty(len(y), dtype=float)
        for train_index, test_index in kf.split(X):
            tmp_knn = copy.deepcopy(self.knn).fit(X[train_index, :], y[train_index])
            tmp_pred = tmp_knn.predict_proba(X[test_index, :])[:, 1]
            if self.laplace:
                tmp_pred = (tmp_pred * self.knn.n_neighbors + 1) / (self.knn.n_neighbors + 2)
            if self.logit:
                tmp_pred = np.log(tmp_pred / (1.0 - tmp_pred))
            insample_prediction_proba[test_index] = tmp_pred
            trained_models.append(tmp_knn)

        self.trained_models = trained_models
        self.insample_prediction_proba = np.reshape(insample_prediction_proba, (insample_prediction_proba.size, 1))
        return self

    def fit_transform(self, X, y):
        """
        Fit the model and return the in-sample predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        insample_prediction_proba : ndarray of shape (n_samples, 1)
            In-sample prediction probabilities.
        """
        self.fit(X, y)
        return self.insample_prediction_proba
        
    def transform(self, X):
        """
        Transform the input data using the fitted K-Nearest Neighbors classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        transformed_data : ndarray of shape (n_samples, 1)
            Transformed data.
        """
        X = check_array(X)
        
        # Select subset of columns and renormalize
        X = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x * x)), 1, X[:, :self.nx])
        
        if isinstance(self.ncv, int) and self.ncv < 2:
            tmp_pred = self.trained_models[0].predict_proba(X)[:, 1]
            if self.laplace:
                tmp_pred = (tmp_pred * self.knn.n_neighbors + 1) / (self.knn.n_neighbors + 2)
            if self.logit:
                tmp_pred = np.log(tmp_pred / (1.0 - tmp_pred))
            tmp_pred = np.reshape(tmp_pred, (tmp_pred.size, 1))
            return tmp_pred
        
        all_preds = np.empty((len(X), self.kfolds.get_n_splits()), dtype=float)
        for n, model in enumerate(self.trained_models):
            tmp_pred = model.predict_proba(X)[:, 1]
            if self.laplace:
                tmp_pred = (tmp_pred * self.knn.n_neighbors + 1) / (self.knn.n_neighbors + 2)
            if self.logit:
                tmp_pred = np.log(tmp_pred / (1.0 - tmp_pred))
            all_preds[:, n] = tmp_pred
        ret = np.mean(all_preds, axis=1)
        return np.reshape(ret, (ret.size, 1))

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        ret = self.predict_proba(X)
        return np.where(ret < 0.5, 0, 1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, 1)
            Predicted class probabilities.
        """
        ret = self.transform(X)
        if self.logit:
            ret = 1.0 / (1.0 + np.exp(-ret))
        return ret

class CompressRegressor(BaseEstimator, RegressorMixin):
    """
    Compressing a set of features - such as text embeddings - into a single feature,
    using K-Nearest-Neightbors regression, wrapped in cross-fit.

    Parameters
    ----------
    nx : int, optional
        Number of text embedding features to include in compression. Defaults to all features in X.
    ncv : int or cross-validation generator, default=5
        Number of cross-validation folds or a cross-validation generator. (If ncv < 2,
        no cross-fit is performed. This should only be used for experimentation.)
    **kwargs : dict
        Additional parameters for KNeighborsRegressor.

    Attributes
    ----------
    trained_model : object
        The fitted KNeighborsRegressor model.
    insample_predictions : ndarray of shape (n_samples, 1)
        In-sample predictions.
    trained_models : list
        List of trained models for each fold.
    kfolds : object
        Cross-validation generator.
    """
    def __init__(self, nx=None, ncv=5, **kwargs):
        super().__init__()
        self.knn = KNeighborsRegressor(**kwargs)
        self.nx = nx
        self.ncv = ncv

    def fit(self, X, y):
        """
        Fit the K-Nearest Neighbors regressor using cross-validation or a single fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        y = np.array(y)
        
        if not self.nx:
            self.nx = X.shape[1]
        
        if self.nx > X.shape[1]:
            raise ValueError('X has fewer columns than nx')
        
        X, y = check_X_y(X, y)
        
        # Select subset of columns and renormalize
        X = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x * x)), 1, X[:, :self.nx])

        if isinstance(self.ncv, int) and self.ncv < 2:
            # No cross-fitting
            self.trained_models = [copy.deepcopy(self.knn).fit(X, y)]
            insample_predictions = self.trained_models[0].predict(X)
            self.insample_predictions = np.reshape(insample_predictions, (insample_predictions.size, 1))
            return self
        
        # Create folds
        if isinstance(self.ncv, (KFold, RepeatedKFold)):
            kf = self.ncv
        else:
            kf = KFold(n_splits=self.ncv, shuffle=True)
        
        kf.get_n_splits(X)
        self.kfolds = kf
        
        # Train model within each fold
        trained_models = []
        insample_predictions = np.empty(len(y), dtype=float)
        for train_index, test_index in kf.split(X):
            tmp_knn = copy.deepcopy(self.knn).fit(X[train_index, :], y[train_index])
            tmp_pred = tmp_knn.predict(X[test_index, :])
            insample_predictions[test_index] = tmp_pred
            trained_models.append(tmp_knn)

        self.trained_models = trained_models
        self.insample_predictions = np.reshape(insample_predictions, (insample_predictions.size, 1))
        return self

    def fit_transform(self, X, y):
        """
        Fit the model and return the in-sample predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        insample_predictions : ndarray of shape (n_samples, 1)
            In-sample predictions.
        """
        self.fit(X, y)
        return self.insample_predictions
        
    def transform(self, X):
        """
        Transform the input data using the fitted K-Nearest Neighbors regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        transformed_data : ndarray of shape (n_samples, 1)
            Transformed data.
        """
        X = check_array(X)
        
        # Select subset of columns and renormalize
        X = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x * x)), 1, X[:, :self.nx])
        
        if isinstance(self.ncv, int) and self.ncv < 2:
            ret = self.trained_models[0].predict(X)
            return np.reshape(ret, (ret.size, 1))
        
        all_preds = np.empty((len(X), self.kfolds.get_n_splits()), dtype=float)
        for n, model in enumerate(self.trained_models):
            tmp_pred = model.predict(X)
            all_preds[:, n] = tmp_pred
        ret = np.mean(all_preds, axis=1)
        return np.reshape(ret, (ret.size, 1))

    def predict(self, X):
        """
        Predict the target values for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples, 1)
            Predicted target values.
        """
        return self.transform(X)