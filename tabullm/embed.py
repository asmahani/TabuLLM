# Copyright (c) 2024 Alireza S. Mahani and Mansour T.A. Sharabiani
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Module for create numeric embeddings from text columns.

The TextColumnTransformer class in this module can leverage any of the embedding models 
available through LangChain to convert one or more text columns into dense numerical vectors. 
Adhering to the scikit-learn transformer interface, TextColumnTransformer can be integrated 
into larger machine learning pipelines as a feature extraction step.
"""


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize as sklearn_normalize
from langchain_core.embeddings import Embeddings


class TextColumnTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for converting one or more text columns to a numeric matrix 
    using Langchain embedding models.

    Parameters
    ----------
    model : Embeddings
        The Langchain text embedding model to use for transforming the text columns.
    colnames : dict or None, default=None
        Optional display names for columns in embedding payload.
        Format: {actual_column_name: display_name}
        If None, uses actual column names.
        Example: {'desc': 'Product Description', 'title': 'Title'}
    colsep : str, default=' || '
        The column separator for concatenating multiple text columns, if applicable.
    prefix : str, default='X_'
        The prefix for the returned embedding columns.
    normalize : bool, default=False
        If True, L2-normalize each embedding vector after generation.
        Projects all vectors onto the unit hypersphere, making cosine similarity
        equivalent to dot product. Useful when the downstream model (e.g. GMM)
        is sensitive to vector magnitude or when switching between embedding
        models that differ in their output norms.

    Examples
    --------
    >>> from langchain.embeddings import OpenAIEmbeddings
    >>> embedding_model = OpenAIEmbeddings()
    >>> transformer = TextColumnTransformer(
    ...     model=embedding_model,
    ...     text_columns=['col1', 'col2'],
    ...     colsep=' || ',
    ...     prefix='emb_',
    ...     colnames={'col1': 'Column 1', 'col2': 'Column 2'}
    ... )
    >>> df = pd.DataFrame({'col1': ['hello', 'world'], 'col2': ['foo', 'bar']})
    >>> embeddings = transformer.fit_transform(df)
    """

    def __init__(
        self,
        model,
        colnames=None,
        colsep=' || ',
        prefix='X_',
        normalize=False
    ):
        self.model = model
        self.colnames = colnames
        self.colsep = colsep
        self.prefix = prefix
        self.normalize = normalize

    def _validate_and_prepare_args(self):
        # Validate model
        if not isinstance(self.model, Embeddings):
            raise TypeError('model must be an instance of langchain_core.embeddings.Embeddings')

        # Validate colsep
        if not isinstance(self.colsep, str):
            raise TypeError('colsep must be a string')

        # Validate prefix
        if not isinstance(self.prefix, str):
            raise TypeError('prefix must be a string')

        # Validate normalize
        if not isinstance(self.normalize, bool):
            raise TypeError('normalize must be a boolean')

        # Validate colnames
        if self.colnames is not None:
            if not isinstance(self.colnames, dict):
                raise TypeError('colnames must be a dictionary or None')
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.colnames.items()):
                raise TypeError('colnames keys and values must be strings')

    def prep_X(self, X):
        """
        Preprocess the input DataFrame X by concatenating all column values 
        into a single string for each row.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame')

        X_str = X.fillna('').astype(str)

        colnames = self.colnames or {}
        column_names = [colnames.get(col, col) for col in X.columns]

        concatenated = X_str.apply(
            lambda row: self.colsep.join(
                [f'{col_name}: {row[col]}' for col_name, col in zip(column_names, X.columns)]
            ),
            axis=1
        ).tolist()

        return concatenated

    def estimate_tokens(self, X, chars_per_token=4, cost_per_1M_tokens=None):
        """
        Estimate token count and cost for embedding operation without calling API.

        Useful for cost estimation before expensive embedding operations.

        Parameters
        ----------
        X : pandas.DataFrame
            Data to estimate tokens for.
        chars_per_token : float, default=4
            Character-to-token ratio heuristic. Typical values:
            - English text: ~4 characters per token
            - Code: ~3 characters per token
            Adjust based on your text type.
        cost_per_1M_tokens : float or None, default=None
            Embedding model pricing in USD per 1 million tokens (as of Feb 2026).
            Examples:
            - AWS Titan Text V2: 0.02
            - AWS Cohere Embed v4: 0.10
            - OpenAI text-embedding-3-small: 0.02
            - OpenAI text-embedding-3-large: 0.13
            - HuggingFace (local): 0.0 (free)
            If None, cost is not estimated.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'n_samples': Number of samples
            - 'total_chars': Total character count
            - 'estimated_tokens': Estimated token count (total_chars / chars_per_token)
            - 'estimated_cost': Estimated cost in USD (None if cost_per_1M_tokens not provided)

        Notes
        -----
        Uses character count heuristic (not exact tokenization).
        Accuracy typically 75-80% for English text.
        Sufficient for cost estimation purposes.

        Examples
        --------
        >>> # Estimate for free model
        >>> info = transformer.estimate_tokens(df)
        >>> print(f"Will embed {info['estimated_tokens']:,} tokens")
        >>>
        >>> # Estimate with cost (AWS Bedrock)
        >>> info = transformer.estimate_tokens(df, cost_per_1M_tokens=0.02)
        >>> print(f"Estimated cost: ${info['estimated_cost']:.2f}")
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame')

        # Prepare text using same logic as transform
        Xstr = self.prep_X(X)

        # Count characters
        total_chars = sum(len(s) for s in Xstr)

        # Estimate tokens
        estimated_tokens = total_chars / chars_per_token

        # Estimate cost if pricing provided
        estimated_cost = None
        if cost_per_1M_tokens is not None:
            estimated_cost = (estimated_tokens / 1_000_000) * cost_per_1M_tokens

        return {
            'n_samples': len(X),
            'total_chars': total_chars,
            'estimated_tokens': int(estimated_tokens),
            'estimated_cost': estimated_cost
        }

    
    def fit(self, X, y=None):
        """
        Fit the transformer on the data.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        # Perform argument validation and preparation
        self._validate_and_prepare_args()

        # Set fitted attribute (sklearn standard: trailing underscore)
        self.n_features_in_ = X.shape[1]

        # No fitting necessary for this transformer
        return self

    def transform(self, X):
        """
        Transform the data into embeddings.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to transform.

        Returns
        -------
        embeddings : pandas.DataFrame
            The transformed embeddings.
        """
        # Check if the estimator is fitted
        check_is_fitted(self, 'n_features_in_')

        Xstr = self.prep_X(X)

        if not Xstr:
            # Handle empty input gracefully
            return pd.DataFrame()

        try:
            # Generate embeddings
            ret_list = self.model.embed_documents(Xstr)
        except Exception as e:
            raise RuntimeError(f'Error during embedding: {e}') from e

        # Convert to DataFrame
        ret_df = pd.DataFrame(ret_list)
        ret_df.columns = [f'{self.prefix}{i}' for i in range(ret_df.shape[1])]

        if self.normalize:
            ret_df = pd.DataFrame(
                sklearn_normalize(ret_df, norm='l2'),
                columns=ret_df.columns
            )

        return ret_df

    # recommended method for scikit-learn transformers that modify feature names
    # the argument 'input_features' is ignored, but required for compatibility
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str, optional
            Input features (ignored).

        Returns
        -------
        output_feature_names : array-like of str
            Output feature names.
        """
        # Assuming embedding dimension is known after embedding a sample document
        sample_embedding = self.model.embed_documents([''])[0]
        embedding_dim = len(sample_embedding)
        return np.array([f'{self.prefix}{i}' for i in range(embedding_dim)])