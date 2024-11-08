import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TextColumnTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for converting one or more text columns to a numeric matrix using various embedding models.

    Parameters
    ----------
    model : ?? (find class name in langchain documentation)
        The langchain text embedding model to use for transforming the text columns.
    prefix : str, default='X_'
        The prefix for the returned embedding columns.
    colsep : str, default=' || '
        The column separator for concatenating multiple text columns, if applicable.
    colnames : dict or None, default=None
        Descriptive names for the text columns in the dataframe to be embedded. If not provided, the column names will be used.

    """
    
    def __init__(
        self,
        model,
        colsep=' || ',
        prefix='X_',
        colnames = None
    ):
        # Assign constructor parameters to instance attributes without modification
        self.model = model
        self.colsep = colsep
        self.prefix = prefix
        self.colnames = colnames
    
    def _validate_and_prepare_args(self):
        pass
    
    def prep_X(self, X):
        """
        Preprocess the input DataFrame X by concatenating all column values into a single string for each row.
        
        This method converts all columns in the DataFrame to strings, fills any missing values with empty strings, 
        and then concatenates the column values using the specified column separator (`self.colsep`). The resulting 
        list of concatenated strings is returned. If `self.colnames` is provided, those names are used in place of 
        the DataFrame's column names where specified.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame containing the text columns to be transformed. All columns must be of string (object) type.

        Returns
        -------
        list of str
            A list of concatenated strings, where each string corresponds to a row in the DataFrame with the column values
            separated by `self.colsep`.
        
        Raises
        ------
        TypeError
            If any column in the input DataFrame is not of string (object) type.

        Example
        -------
        >>> df = pd.DataFrame({'col1': ['hello', 'world'], 'col2': ['foo', 'bar']})
        >>> transformer = TextColumnTransformer(colsep=' || ', colnames={'col1': 'Column 1'})
        >>> transformer.prep_X(df)
        ['Column 1: hello || col2: foo', 'Column 1: world || col2: bar']
        """
        if not (X.dtypes == 'object').all():
            raise TypeError('All columns of X must be of string (object) type')

        # Use colnames if available, otherwise fall back to X column names
        if self.colnames is None:
            self.colnames = {}
        Xstr = X.fillna('').astype(str).apply(
            lambda row: self.colsep.join([f'{self.colnames.get(col, col)}: {row[col]}' for col in X.columns]), axis=1
        ).tolist()
        return Xstr

    def fit(self, X, y=None):
        """
        Fit the transformer on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit.
        y : array-like of shape (n_samples,), optional
            The target values (ignored).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Perform argument validation and preparation
        self._validate_and_prepare_args()
        
        return self

    def fit_transform(self, X, y=None):
        """
        Fit the transformer on the data and then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit and transform.
        y : array-like of shape (n_samples,), optional
            The target values (ignored).

        Returns
        -------
        embeddings : numpy.ndarray of shape (n_samples, embedding_dim)
            The transformed embeddings.
        """
        self.fit(X, y)
        return self.transform(X)
    
    def transform(self, X):
        """
        Transform the data into embeddings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        embeddings : numpy.ndarray of shape (n_samples, embedding_dim)
            The transformed embeddings.
        """
        Xstr = self.prep_X(X)

        ret_list = self.model.embed_documents(Xstr)
        
        ret_df = pd.DataFrame(ret_list)
        ret_df.columns = [f'{self.prefix}{i}' for i in range(ret_df.shape[1])]
        return ret_df
