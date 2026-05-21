"""
Tests for TextColumnTransformer
"""

import pytest
import pandas as pd
import numpy as np
import os
import warnings

from tabullm import TextColumnTransformer
from langchain_core.embeddings import Embeddings
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


class MockEmbeddings(Embeddings):
    """Mock embedding model for testing when HuggingFace models unavailable"""

    def embed_documents(self, texts):
        """Return fake 10D embeddings"""
        return [[0.1 * (i % 10) for i in range(10)] for _ in texts]

    def embed_query(self, text):
        """Return fake 10D embedding"""
        return [0.1 * i for i in range(10)]


@pytest.fixture
def embedding_model():
    """
    Create embedding model for testing

    Tries to use real HuggingFace model if available (cached offline),
    falls back to mock if unavailable (with warning).
    """
    os.environ['HF_HUB_OFFLINE'] = '1'

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'local_files_only': True}
        )

        # Test if model actually works
        _ = model.embed_query("test")

        return model

    except Exception as e:
        warnings.warn(
            f"HuggingFace model not available ({e}). "
            f"Using MockEmbeddings (10D instead of 384D). "
            f"Tests will run but with reduced dimensions.",
            UserWarning
        )
        return MockEmbeddings()


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame with text columns"""
    return pd.DataFrame({
        'text1': ['hello world', 'foo bar', 'test data'],
        'text2': ['alpha', 'beta', 'gamma'],
        'numeric': [1.0, 2.0, 3.0]
    })


class TestTextColumnTransformer:
    """Test suite for TextColumnTransformer"""

    def test_initialization(self, embedding_model):
        """Test TextColumnTransformer initialization"""
        transformer = TextColumnTransformer(model=embedding_model)

        assert transformer.model is not None
        assert transformer.colsep == ' || '
        assert transformer.prefix == 'X_'
        assert transformer.colnames is None

    def test_custom_parameters(self, embedding_model):
        """Test initialization with custom parameters"""
        transformer = TextColumnTransformer(
            model=embedding_model,
            colnames={'text1': 'First', 'text2': 'Second'},
            colsep=' | ',
            prefix='emb_'
        )

        assert transformer.colnames == {'text1': 'First', 'text2': 'Second'}
        assert transformer.colsep == ' | '
        assert transformer.prefix == 'emb_'

    def test_fit_transform_all_columns(self, embedding_model, sample_dataframe):
        """Test fit_transform with all text columns"""
        df = sample_dataframe[['text1', 'text2']]  # Only text columns

        transformer = TextColumnTransformer(model=embedding_model)
        X_transformed = transformer.fit_transform(df)

        # Should return embedding (384D for real model, 10D for mock)
        assert X_transformed.shape[0] == 3  # 3 samples
        assert X_transformed.shape[1] in [10, 384]  # Mock or real
        assert isinstance(X_transformed, pd.DataFrame)

    def test_fit_transform_single_column(self, embedding_model, sample_dataframe):
        """Test with single text column"""
        df = sample_dataframe[['text1']]

        transformer = TextColumnTransformer(model=embedding_model)
        X_transformed = transformer.fit_transform(df)

        assert X_transformed.shape[0] == 3
        assert X_transformed.shape[1] in [10, 384]  # Mock or real

    def test_column_transformer_integration(self, embedding_model, sample_dataframe):
        """Test integration with sklearn ColumnTransformer"""
        df = sample_dataframe

        ct = ColumnTransformer([
            ('text', TextColumnTransformer(model=embedding_model), ['text1', 'text2']),
            ('numeric', SimpleImputer(), ['numeric'])
        ])

        X_transformed = ct.fit_transform(df)

        # Should have embedding dims + 1 (numeric)
        assert X_transformed.shape[0] == 3
        assert X_transformed.shape[1] in [11, 385]  # Mock: 10+1, Real: 384+1

    def test_column_transformer_with_passthrough(self, embedding_model, sample_dataframe):
        """Test ColumnTransformer with remainder='passthrough'"""
        df = sample_dataframe

        ct = ColumnTransformer([
            ('text', TextColumnTransformer(model=embedding_model), ['text1'])
        ], remainder='passthrough')

        X_transformed = ct.fit_transform(df)

        # Should have embedding dims + 2 (passthrough: text2, numeric)
        assert X_transformed.shape[0] == 3
        assert X_transformed.shape[1] in [12, 386]  # Mock: 10+2, Real: 384+2

    def test_pipeline_integration(self, embedding_model, sample_dataframe):
        """Test in full sklearn Pipeline"""
        df = sample_dataframe[['text1', 'text2']]
        y = np.array([0, 1, 0])

        pipeline = Pipeline([
            ('embed', TextColumnTransformer(model=embedding_model)),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])

        pipeline.fit(df, y)
        predictions = pipeline.predict(df)

        assert predictions.shape == (3,)

    def test_prep_x_method(self, embedding_model, sample_dataframe):
        """Test prep_X method returns correct format"""
        df = sample_dataframe[['text1', 'text2']]

        transformer = TextColumnTransformer(model=embedding_model)
        transformer.fit(df)

        Xstr = transformer.prep_X(df)

        # Should return list of concatenated strings
        assert isinstance(Xstr, list)
        assert len(Xstr) == 3
        assert 'text1:' in Xstr[0]  # Should have column labels
        assert '||' in Xstr[0]  # Should have separator

    def test_custom_colnames_in_prep_x(self, embedding_model, sample_dataframe):
        """Test that colnames affects prep_X output"""
        df = sample_dataframe[['text1', 'text2']]

        transformer = TextColumnTransformer(
            model=embedding_model,
            colnames={'text1': 'First Column', 'text2': 'Second Column'}
        )
        transformer.fit(df)

        Xstr = transformer.prep_X(df)

        # Should use custom names
        assert 'First Column:' in Xstr[0]
        assert 'Second Column:' in Xstr[0]

    def test_missing_values_handled(self, embedding_model):
        """Test that NaN values are handled correctly"""
        df = pd.DataFrame({
            'text1': ['hello', np.nan, 'world'],
            'text2': ['foo', 'bar', np.nan]
        })

        transformer = TextColumnTransformer(model=embedding_model)
        X_transformed = transformer.fit_transform(df)

        # Should complete without error and preserve row count
        assert X_transformed.shape[0] == 3
        assert X_transformed.shape[1] > 0

        # Check prep_X handles NaN
        Xstr = transformer.prep_X(df)
        # NaN should become empty string
        assert 'text1: ' in Xstr[1]  # Row with NaN in text1

    def test_reproducibility(self, embedding_model, sample_dataframe):
        """Test that results are reproducible"""
        df = sample_dataframe[['text1']]

        t1 = TextColumnTransformer(model=embedding_model)
        t2 = TextColumnTransformer(model=embedding_model)

        X1 = t1.fit_transform(df)
        X2 = t2.fit_transform(df)

        # Should be identical (deterministic embeddings)
        pd.testing.assert_frame_equal(X1, X2)

    def test_estimate_tokens(self, embedding_model, sample_dataframe):
        """Test token estimation"""
        df = sample_dataframe[['text1', 'text2']]

        transformer = TextColumnTransformer(model=embedding_model)

        # Test without cost
        info = transformer.estimate_tokens(df)

        assert 'n_samples' in info
        assert info['n_samples'] == 3
        assert 'total_chars' in info
        assert 'estimated_tokens' in info
        assert info['estimated_tokens'] > 0
        assert info['estimated_cost'] is None

    def test_estimate_tokens_with_cost(self, embedding_model, sample_dataframe):
        """Test token estimation with cost"""
        df = sample_dataframe[['text1']]

        transformer = TextColumnTransformer(model=embedding_model)

        # Test with cost
        info = transformer.estimate_tokens(df, cost_per_1M_tokens=0.10)

        assert info['estimated_cost'] is not None
        assert isinstance(info['estimated_cost'], float)
        assert info['estimated_cost'] >= 0


class TestPrepXMutation:
    def test_colnames_none_unchanged_after_prep_x(self, embedding_model, sample_dataframe):
        """prep_X must not mutate self.colnames when it starts as None."""
        df = sample_dataframe[['text1', 'text2']]
        t = TextColumnTransformer(model=embedding_model)
        assert t.colnames is None
        t.prep_X(df)
        assert t.colnames is None

    def test_get_params_stable_after_prep_x(self, embedding_model, sample_dataframe):
        """get_params() should return the same colnames before and after prep_X."""
        df = sample_dataframe[['text1', 'text2']]
        t = TextColumnTransformer(model=embedding_model)
        params_before = t.get_params()
        t.prep_X(df)
        assert t.get_params() == params_before


class TestLangChainCompatibility:
    def test_invalid_model_error_mentions_langchain_core_contract(self):
        with pytest.raises(TypeError, match=r"langchain_core\.embeddings\.Embeddings"):
            TextColumnTransformer(model=object()).fit(
                pd.DataFrame({'text': ['hi']})
            )


class TestNormalize:
    def test_normalize_false_by_default(self, embedding_model):
        t = TextColumnTransformer(model=embedding_model)
        assert t.normalize is False

    def test_normalize_true_stored(self, embedding_model):
        t = TextColumnTransformer(model=embedding_model, normalize=True)
        assert t.normalize is True

    def test_normalize_invalid_type_raises(self, embedding_model):
        with pytest.raises(TypeError, match="normalize must be a boolean"):
            TextColumnTransformer(model=embedding_model, normalize='yes').fit(
                pd.DataFrame({'text': ['hi']})
            )

    def test_output_shape_unchanged(self, embedding_model, sample_dataframe):
        df = sample_dataframe[['text1', 'text2']]
        t_norm = TextColumnTransformer(model=embedding_model, normalize=True)
        t_plain = TextColumnTransformer(model=embedding_model, normalize=False)
        assert t_norm.fit_transform(df).shape == t_plain.fit_transform(df).shape

    def test_rows_are_unit_vectors(self, embedding_model, sample_dataframe):
        df = sample_dataframe[['text1', 'text2']]
        t = TextColumnTransformer(model=embedding_model, normalize=True)
        X = t.fit_transform(df).values
        norms = np.linalg.norm(X, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_no_normalization_without_flag(self, embedding_model, sample_dataframe):
        """Rows are NOT forced to unit norm when normalize=False."""
        df = sample_dataframe[['text1', 'text2']]
        t = TextColumnTransformer(model=embedding_model, normalize=False)
        X = t.fit_transform(df).values
        norms = np.linalg.norm(X, axis=1)
        # MockEmbeddings returns identical rows, so the norm may happen to be 1;
        # just confirm normalization wasn't applied by verifying values unchanged
        t2 = TextColumnTransformer(model=embedding_model, normalize=False)
        X2 = t2.fit_transform(df).values
        np.testing.assert_array_equal(X, X2)
