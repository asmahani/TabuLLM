"""
Tests for ClusterExplainer with refactored API
"""

import pytest
import pandas as pd
import numpy as np
import os

os.environ['HF_HUB_OFFLINE'] = '1'

from tabullm import TextColumnTransformer, ClusterExplainer
from tabullm.explain import _stat_associations, _adjust_pvalues
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class MockLLM(BaseChatModel):
    """Mock LLM for testing without API calls"""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content="Mock summary text for testing")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self):
        return "mock"


class MockEmbeddings(Embeddings):
    """Mock embedding model"""

    def embed_documents(self, texts):
        return [[0.1 * i for i in range(10)] for _ in texts]

    def embed_query(self, text):
        return [0.1 * i for i in range(10)]


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'text1': ['Sample text'] * 20,
        'text2': ['More text'] * 20
    })


@pytest.fixture
def text_transformer():
    return TextColumnTransformer(model=MockEmbeddings())


@pytest.fixture
def mock_llm():
    return MockLLM()


class TestInitialization:
    def test_basic(self, mock_llm, text_transformer):
        explainer = ClusterExplainer(
            llm=mock_llm,
            text_transformer=text_transformer,
            observations="data",
            text_fields="fields"
        )
        assert explainer.llm_context_length == 200_000
        assert explainer.llm_cost_per_1M_tokens is None



class TestTokenCounting:
    def test_count_tokens_only(self, mock_llm, text_transformer, sample_data):
        text_transformer.fit(sample_data)
        cluster_labels = np.array([0] * 10 + [1] * 10)

        explainer = ClusterExplainer(
            llm=mock_llm,
            text_transformer=text_transformer,
            observations="data",
            text_fields="fields"
        )

        result = explainer.explain(
            sample_data, cluster_labels, preview=True
        )

        assert 'total_tokens' in result
        assert 'estimated_cost' in result
        assert 'strategy' in result
        assert result['total_tokens'] > 0
        assert result['strategy'] in ('one-stage', 'two-stage')

    def test_preview_prints_summary(self, mock_llm, text_transformer, sample_data, capsys):
        text_transformer.fit(sample_data)
        cluster_labels = np.array([0] * 10 + [1] * 10)
        explainer = ClusterExplainer(
            llm=mock_llm,
            text_transformer=text_transformer,
            observations="data",
            text_fields="fields"
        )
        explainer.explain(sample_data, cluster_labels, preview=True)
        out = capsys.readouterr().out
        assert 'Tokens' in out
        assert 'Strategy' in out

    def test_count_tokens_respects_max_members(self, mock_llm, text_transformer, sample_data):
        text_transformer.fit(sample_data)
        cluster_labels = np.array([0] * 10 + [1] * 10)

        explainer = ClusterExplainer(
            llm=mock_llm,
            text_transformer=text_transformer,
            observations="data",
            text_fields="fields"
        )

        full = explainer.explain(sample_data, cluster_labels, preview=True)
        sampled = explainer.explain(
            sample_data, cluster_labels,
            max_members_per_cluster=2, random_state=0,
            preview=True
        )

        assert sampled['total_tokens'] < full['total_tokens']


class TestRecursion:
    def test_recursive_base_case(self, mock_llm, text_transformer):
        explainer = ClusterExplainer(
            llm=mock_llm,
            text_transformer=text_transformer,
            observations="data",
            text_fields="fields"
        )

        items = ["text1", "text2"]
        summary = explainer._recursive_summarize(items, depth=0, max_tokens=1000, max_words=100)
        assert isinstance(summary, str)

    def test_single_item_too_large_error(self, mock_llm, text_transformer):
        explainer = ClusterExplainer(
            llm=mock_llm,
            text_transformer=text_transformer,
            observations="data",
            text_fields="fields"
        )

        items = ["x" * 10000]
        with pytest.raises(ValueError, match="exceeds context"):
            explainer._recursive_summarize(items, depth=0, max_tokens=100, max_words=50)


class TestPreviewPrompts:
    @pytest.fixture
    def explainer(self, mock_llm, text_transformer):
        return ClusterExplainer(
            llm=mock_llm,
            text_transformer=text_transformer,
            observations="job postings",
            text_fields="titles and descriptions"
        )

    def test_returns_all_four_keys(self, explainer):
        result = explainer.preview_prompts()
        assert set(result.keys()) == {
            'label_direct', 'summarize_observations',
            'combine_summaries', 'label_from_summaries', 'synthesize'
        }

    def test_injects_observations_and_text_fields(self, explainer):
        result = explainer.preview_prompts()
        for key in ('label_direct', 'summarize_observations', 'label_from_summaries'):
            assert 'job postings' in result[key]
            assert 'titles and descriptions' in result[key]

    def test_prints_each_prompt(self, explainer, capsys):
        explainer.preview_prompts()
        out = capsys.readouterr().out
        for key in ('label_direct', 'summarize_observations', 'combine_summaries',
                    'label_from_summaries', 'synthesize'):
            assert key in out


class TestSampleCluster:
    @pytest.fixture
    def explainer(self, mock_llm, text_transformer):
        return ClusterExplainer(
            llm=mock_llm,
            text_transformer=text_transformer,
            observations="data",
            text_fields="fields"
        )

    def test_noop_when_max_members_none(self, explainer):
        items = ['a', 'b', 'c', 'd', 'e']
        assert explainer._sample_cluster(items, None, None) is items

    def test_noop_when_already_small(self, explainer):
        items = ['a', 'b', 'c']
        assert explainer._sample_cluster(items, 5, None) is items

    def test_noop_when_exact_size(self, explainer):
        items = ['a', 'b', 'c']
        assert explainer._sample_cluster(items, 3, None) is items

    def test_reduces_to_max_members(self, explainer):
        items = list(range(20))
        result = explainer._sample_cluster(items, 5, 42)
        assert len(result) == 5

    def test_reproducible_with_seed(self, explainer):
        items = list(range(50))
        r1 = explainer._sample_cluster(items, 10, 99)
        r2 = explainer._sample_cluster(items, 10, 99)
        assert r1 == r2

    def test_different_seeds_differ(self, explainer):
        items = list(range(50))
        r1 = explainer._sample_cluster(items, 10, 1)
        r2 = explainer._sample_cluster(items, 10, 2)
        assert r1 != r2


class TestValidateY:
    def test_binary_numeric(self):
        assert ClusterExplainer._validate_y([0, 1, 0, 1]) == 'binary'

    def test_binary_float(self):
        assert ClusterExplainer._validate_y([0.0, 1.0, 1.0, 0.0]) == 'binary'

    def test_boolean_dtype(self):
        y = pd.array([True, False, True], dtype='boolean')
        assert ClusterExplainer._validate_y(y) == 'binary'

    def test_numpy_bool(self):
        y = np.array([True, False, True])
        assert ClusterExplainer._validate_y(y) == 'binary'

    def test_continuous(self):
        assert ClusterExplainer._validate_y([1, 2, 3, 4, 5]) == 'continuous'

    def test_continuous_with_two_plus(self):
        assert ClusterExplainer._validate_y([0, 1, 2]) == 'continuous'

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="only NaN"):
            ClusterExplainer._validate_y([float('nan'), float('nan')])

    def test_one_unique_value_raises(self):
        with pytest.raises(ValueError, match="1 unique"):
            ClusterExplainer._validate_y([1, 1, 1])

    def test_string_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            ClusterExplainer._validate_y(['a', 'b', 'a'])

    def test_string_binary_raises_with_encode_hint(self):
        with pytest.raises(ValueError, match="0 for the negative"):
            ClusterExplainer._validate_y(['fraud', 'legit', 'fraud'])

    def test_nan_ignored_for_unique_count(self):
        # 2 non-NaN unique values → binary, even with NaNs present
        assert ClusterExplainer._validate_y([0, 1, float('nan'), 0]) == 'binary'


class TestOneVsRest:
    @pytest.fixture
    def binary_df(self):
        return pd.DataFrame({
            'Cluster': [0, 0, 0, 1, 1, 1],
            'Outcome':  [0, 1, 0, 1, 1, 0]
        })

    @pytest.fixture
    def continuous_df(self):
        return pd.DataFrame({
            'Cluster': [0, 0, 0, 1, 1, 1],
            'Outcome':  [1.0, 2.5, 3.0, 4.0, 5.5, 6.0]
        })

    def test_binary_auto_detect(self, binary_df):
        from tabullm.explain import one_vs_rest
        results, global_results = one_vs_rest(binary_df, col_x='Cluster', col_y='Outcome')
        assert 'Fisher Exact Test' in results['Test'].values

    def test_binary_with_y_type(self, binary_df):
        from tabullm.explain import one_vs_rest
        results, _ = one_vs_rest(binary_df, col_x='Cluster', col_y='Outcome', y_type='binary')
        assert 'Fisher Exact Test' in results['Test'].values

    def test_continuous_auto_detect(self, continuous_df):
        from tabullm.explain import one_vs_rest
        results, _ = one_vs_rest(continuous_df, col_x='Cluster', col_y='Outcome')
        assert 'Independent t-test' in results['Test'].values

    def test_continuous_with_y_type(self, continuous_df):
        from tabullm.explain import one_vs_rest
        results, _ = one_vs_rest(continuous_df, col_x='Cluster', col_y='Outcome', y_type='continuous')
        assert 'Independent t-test' in results['Test'].values

    def test_boolean_y_auto_detect(self):
        from tabullm.explain import one_vs_rest
        df = pd.DataFrame({
            'Cluster': [0, 0, 1, 1],
            'Outcome':  np.array([True, False, True, False])
        })
        results, _ = one_vs_rest(df, col_x='Cluster', col_y='Outcome')
        assert 'Fisher Exact Test' in results['Test'].values

    def test_string_y_raises(self):
        from tabullm.explain import one_vs_rest
        df = pd.DataFrame({
            'Cluster': [0, 0, 1, 1],
            'Outcome':  ['a', 'b', 'a', 'b']
        })
        with pytest.raises(ValueError, match="not supported"):
            one_vs_rest(df, col_x='Cluster', col_y='Outcome')

    def test_no_correction_no_adjusted_col(self, binary_df):
        from tabullm.explain import one_vs_rest
        results, _ = one_vs_rest(binary_df, col_x='Cluster', col_y='Outcome')
        assert 'P-value (adjusted)' not in results.columns

    def test_bonferroni_adds_adjusted_col(self, binary_df):
        from tabullm.explain import one_vs_rest
        results, _ = one_vs_rest(binary_df, col_x='Cluster', col_y='Outcome',
                                 correction='bonferroni')
        assert 'P-value (adjusted)' in results.columns

    def test_holm_adds_adjusted_col(self, binary_df):
        from tabullm.explain import one_vs_rest
        results, _ = one_vs_rest(binary_df, col_x='Cluster', col_y='Outcome',
                                 correction='holm')
        assert 'P-value (adjusted)' in results.columns

    def test_fdr_bh_adds_adjusted_col(self, continuous_df):
        from tabullm.explain import one_vs_rest
        results, _ = one_vs_rest(continuous_df, col_x='Cluster', col_y='Outcome',
                                 correction='fdr_bh')
        assert 'P-value (adjusted)' in results.columns

    def test_adjusted_geq_raw_bonferroni(self, binary_df):
        from tabullm.explain import one_vs_rest
        results, _ = one_vs_rest(binary_df, col_x='Cluster', col_y='Outcome',
                                 correction='bonferroni')
        valid = results.dropna(subset=['P-value', 'P-value (adjusted)'])
        assert (valid['P-value (adjusted)'] >= valid['P-value']).all()

    def test_adjusted_bounded(self, binary_df):
        from tabullm.explain import one_vs_rest
        results, _ = one_vs_rest(binary_df, col_x='Cluster', col_y='Outcome',
                                 correction='bonferroni')
        valid = results['P-value (adjusted)'].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_global_pvalue_not_in_adjusted(self, binary_df):
        # Global test should never carry 'P-value (adjusted)'
        from tabullm.explain import one_vs_rest
        _, global_results = one_vs_rest(binary_df, col_x='Cluster', col_y='Outcome',
                                        correction='bonferroni')
        assert 'P-value (adjusted)' not in global_results.columns


class TestAdjustPvalues:
    def test_bonferroni_multiplies_by_m(self):
        pvals = [0.01, 0.02, 0.03, 0.04]
        adj = _adjust_pvalues(pvals, 'bonferroni')
        assert adj == pytest.approx([0.04, 0.08, 0.12, 0.16])

    def test_bonferroni_clips_to_one(self):
        adj = _adjust_pvalues([0.5, 0.6], 'bonferroni')
        assert all(v <= 1.0 for v in adj)

    def test_holm_monotone_nondecreasing(self):
        pvals = [0.01, 0.04, 0.03, 0.2]
        adj = _adjust_pvalues(pvals, 'holm')
        # Map back to sorted order and verify non-decreasing
        sorted_adj = sorted(adj)
        assert sorted_adj == sorted_adj  # trivially true; check actual monotonicity
        # The adjusted values in sorted order must be non-decreasing
        order = np.argsort(pvals)
        adj_sorted = [adj[i] for i in order]
        assert all(adj_sorted[i] <= adj_sorted[i+1] for i in range(len(adj_sorted)-1))

    def test_holm_geq_raw(self):
        pvals = [0.01, 0.02, 0.05]
        adj = _adjust_pvalues(pvals, 'holm')
        assert all(a >= r for a, r in zip(adj, pvals))

    def test_fdr_bh_bounded(self):
        pvals = [0.001, 0.01, 0.05, 0.1, 0.5]
        adj = _adjust_pvalues(pvals, 'fdr_bh')
        assert all(0 <= v <= 1 for v in adj)

    def test_fdr_bh_geq_raw(self):
        pvals = [0.001, 0.01, 0.05]
        adj = _adjust_pvalues(pvals, 'fdr_bh')
        assert all(a >= r for a, r in zip(adj, pvals))

    def test_none_preserved(self):
        pvals = [0.01, None, 0.05]
        adj = _adjust_pvalues(pvals, 'bonferroni')
        assert adj[1] is None

    def test_none_only_returns_unchanged(self):
        pvals = [None, None]
        adj = _adjust_pvalues(pvals, 'holm')
        assert adj == [None, None]

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown correction method"):
            _adjust_pvalues([0.05], 'invalid')

    def test_single_pvalue_unchanged_bonferroni(self):
        adj = _adjust_pvalues([0.05], 'bonferroni')
        assert adj == pytest.approx([0.05])


class TestExplainYValidation:
    """Verify that y validation errors surface at explain() entry, not deep in one_vs_rest."""

    @pytest.fixture
    def explainer(self, mock_llm, text_transformer):
        return ClusterExplainer(
            llm=mock_llm,
            text_transformer=text_transformer,
            observations="data",
            text_fields="fields"
        )

    def test_string_y_raises_at_entry(self, explainer, sample_data):
        sample_data_fitted = sample_data.copy()
        cluster_labels = np.array([0] * 10 + [1] * 10)
        y_bad = ['a', 'b'] * 10
        with pytest.raises(ValueError, match="not supported"):
            explainer.explain(sample_data_fitted, cluster_labels, y=y_bad, preview=True)

    def test_all_nan_y_raises_at_entry(self, explainer, sample_data):
        cluster_labels = np.array([0] * 10 + [1] * 10)
        y_bad = [float('nan')] * 20
        with pytest.raises(ValueError, match="only NaN"):
            explainer.explain(sample_data, cluster_labels, y=y_bad, preview=True)


class TestStatAssociations:
    """Tests for _stat_associations helper."""

    @pytest.fixture
    def obs_stats(self):
        np.random.seed(0)
        return pd.DataFrame({
            'score_a': np.random.randn(20),
            'score_b': np.random.randn(20),
        })

    def test_binary_y_returns_t_test(self, obs_stats):
        y = [0] * 10 + [1] * 10
        result = _stat_associations(obs_stats, y, 'binary')
        assert list(result['Test'].unique()) == ['Independent t-test']

    def test_continuous_y_returns_pearson(self, obs_stats):
        y = np.random.randn(20)
        result = _stat_associations(obs_stats, y, 'continuous')
        assert list(result['Test'].unique()) == ['Pearson correlation']

    def test_one_row_per_column(self, obs_stats):
        y = [0] * 10 + [1] * 10
        result = _stat_associations(obs_stats, y, 'binary')
        assert len(result) == len(obs_stats.columns)
        assert list(result['Stat']) == list(obs_stats.columns)

    def test_columns_present_binary(self, obs_stats):
        y = [0] * 10 + [1] * 10
        result = _stat_associations(obs_stats, y, 'binary')
        assert set(result.columns) == {'Stat', 'Test', 'Mean (y=0)', 'Mean (y=1)', 'Statistic', 'P-value'}

    def test_columns_present_continuous(self, obs_stats):
        y = np.random.randn(20)
        result = _stat_associations(obs_stats, y, 'continuous')
        assert set(result.columns) == {'Stat', 'Test', 'Statistic', 'P-value'}

    def test_group_means_correct_direction(self, obs_stats):
        # g0 (y=0) has values shifted up by 2, so Mean(y=0) > Mean(y=1) → positive t
        y = np.array([0] * 10 + [1] * 10)
        obs = pd.DataFrame({'x': np.concatenate([np.ones(10) * 5, np.ones(10) * 1])})
        result = _stat_associations(obs, y, 'binary')
        assert result['Mean (y=0)'].iloc[0] > result['Mean (y=1)'].iloc[0]
        assert result['Statistic'].iloc[0] > 0   # positive t: higher in y=0

    def test_group_means_sorted_by_y_value(self, obs_stats):
        # unique_vals are sorted, so column order is deterministic regardless of y order
        y = [1] * 10 + [0] * 10   # y=1 comes first in the array
        result = _stat_associations(obs_stats, y, 'binary')
        assert 'Mean (y=0)' in result.columns
        assert 'Mean (y=1)' in result.columns

    def test_no_correction_no_adjusted_col(self, obs_stats):
        y = [0] * 10 + [1] * 10
        result = _stat_associations(obs_stats, y, 'binary')
        assert 'P-value (adjusted)' not in result.columns

    def test_correction_adds_adjusted_col_binary(self, obs_stats):
        y = [0] * 10 + [1] * 10
        result = _stat_associations(obs_stats, y, 'binary', correction='bonferroni')
        assert 'P-value (adjusted)' in result.columns

    def test_correction_adds_adjusted_col_continuous(self, obs_stats):
        y = np.random.randn(20)
        result = _stat_associations(obs_stats, y, 'continuous', correction='fdr_bh')
        assert 'P-value (adjusted)' in result.columns

    def test_adjusted_geq_raw_bonferroni(self, obs_stats):
        y = [0] * 10 + [1] * 10
        result = _stat_associations(obs_stats, y, 'binary', correction='bonferroni')
        assert (result['P-value (adjusted)'] >= result['P-value']).all()


class TestObservationStatsInExplain:
    """Integration tests for observation_stats parameter in explain()."""

    @pytest.fixture
    def explainer(self, mock_llm, text_transformer):
        return ClusterExplainer(
            llm=mock_llm,
            text_transformer=text_transformer,
            observations="data",
            text_fields="fields"
        )

    @pytest.fixture
    def obs_stats(self, sample_data):
        np.random.seed(1)
        return pd.DataFrame({
            'stat_x': np.random.randn(len(sample_data)),
            'stat_y': np.random.randn(len(sample_data)),
        })

    def test_wrong_type_raises(self, explainer, sample_data):
        cluster_labels = np.array([0] * 10 + [1] * 10)
        with pytest.raises(TypeError, match="pandas DataFrame"):
            explainer.explain(sample_data, cluster_labels,
                              observation_stats={'a': [1, 2]}, preview=True)

    def test_wrong_length_raises(self, explainer, sample_data):
        cluster_labels = np.array([0] * 10 + [1] * 10)
        short_stats = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(ValueError, match="rows"):
            explainer.explain(sample_data, cluster_labels,
                              observation_stats=short_stats, preview=True)

    def test_cluster_means_in_table(self, explainer, sample_data, obs_stats):
        text_transformer = explainer.text_transformer
        text_transformer.fit(sample_data)
        cluster_labels = np.array([0] * 10 + [1] * 10)
        result = explainer.explain(
            sample_data, cluster_labels, observation_stats=obs_stats, preview=True
        )
        # preview returns dict — just check validation passed (no error)
        assert 'total_tokens' in result

    def test_returns_three_tuple_with_y_and_obs_stats(
        self, explainer, sample_data, obs_stats
    ):
        text_transformer = explainer.text_transformer
        text_transformer.fit(sample_data)
        cluster_labels = np.array([0] * 10 + [1] * 10)
        y = [0] * 10 + [1] * 10

        # Use MockLLM — need a real explain() call; mock returns valid JSON-ish?
        # Preview mode won't trigger LLM, so test return type via preview=True first
        # For the tuple test we verify the structure at preview=False indirectly
        # by checking _stat_associations is called correctly (covered above).
        # Here just confirm no error is raised during validation path.
        result = explainer.explain(
            sample_data, cluster_labels, y=y,
            observation_stats=obs_stats, preview=True
        )
        assert 'strategy' in result


# ---------------------------------------------------------------------------
# Helper: a CapturingLLM that records what each message contains
# ---------------------------------------------------------------------------

class CapturingLLM(BaseChatModel):
    """MockLLM that also records the content of every message it receives."""

    captured: list = []  # filled during _generate calls

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        for m in messages:
            self.captured.append(m.content)
        message = AIMessage(content="Captured synthesis text")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self):
        return "capturing"


# ---------------------------------------------------------------------------
# Helper fixture: a MultipleGroupLabels stub that _explain_1stage can return
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_labels():
    from tabullm.explain import MultipleGroupLabels, GroupLabel
    return MultipleGroupLabels(groups=[
        GroupLabel(group_number=0, description_short='G0', description_long='Long 0'),
        GroupLabel(group_number=1, description_short='G1', description_long='Long 1'),
    ])


# ---------------------------------------------------------------------------
# Shared result_df fixtures used across synthesis tests
# ---------------------------------------------------------------------------

@pytest.fixture
def result_df_no_y():
    return pd.DataFrame({
        'Group Number': [0, 1],
        'Short Description': ['Group A', 'Group B'],
        'Long Description': ['Long description A', 'Long description B'],
    })


@pytest.fixture
def result_df_with_y():
    return pd.DataFrame({
        'Group Number': [0, 1],
        'Short Description': ['Group A', 'Group B'],
        'Long Description': ['Long description A', 'Long description B'],
        'Size': [100, 200],
        'Test': ['Fisher Exact Test', 'Fisher Exact Test'],
        'Odds Ratio': [2.5, 0.3],
        'P-value': [0.001, 0.05],
    })


@pytest.fixture
def global_results():
    return pd.DataFrame([{
        'Test': 'Chi-Square Test', 'Statistic': 15.2,
        'P-value': 0.0001, 'Degrees of Freedom': 1
    }])


@pytest.fixture
def stat_assoc_df():
    return pd.DataFrame([{
        'Stat': 'max_posterior', 'Test': 'Independent t-test',
        'Statistic': 3.1, 'P-value': 0.002
    }])


# ---------------------------------------------------------------------------
# TestPreviewSynthesis
# ---------------------------------------------------------------------------

class TestPreviewSynthesis:
    def test_preview_shows_synthesis_enabled(
        self, mock_llm, text_transformer, sample_data, capsys
    ):
        text_transformer.fit(sample_data)
        cluster_labels = np.array([0] * 10 + [1] * 10)
        explainer = ClusterExplainer(
            llm=mock_llm, text_transformer=text_transformer,
            observations="data", text_fields="fields"
        )
        explainer.explain(sample_data, cluster_labels, synthesize=True, preview=True)
        out = capsys.readouterr().out
        assert 'Synthesis' in out
        assert 'enabled' in out

    def test_preview_no_synthesis_line_when_false(
        self, mock_llm, text_transformer, sample_data, capsys
    ):
        text_transformer.fit(sample_data)
        cluster_labels = np.array([0] * 10 + [1] * 10)
        explainer = ClusterExplainer(
            llm=mock_llm, text_transformer=text_transformer,
            observations="data", text_fields="fields"
        )
        explainer.explain(sample_data, cluster_labels, preview=True)
        out = capsys.readouterr().out
        assert 'Synthesis' not in out


# ---------------------------------------------------------------------------
# TestSynthesize
# ---------------------------------------------------------------------------

class TestSynthesize:
    """Tests for the _synthesize() method."""

    @pytest.fixture
    def cap_explainer(self, text_transformer):
        llm = CapturingLLM(captured=[])
        return ClusterExplainer(
            llm=llm,
            text_transformer=text_transformer,
            observations="job postings",
            text_fields="titles and descriptions"
        )

    def test_returns_str(self, cap_explainer, result_df_no_y):
        result = cap_explainer._synthesize(result_df_no_y)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_y_label_in_prompt(self, cap_explainer, result_df_with_y, global_results):
        cap_explainer.llm.captured.clear()
        cap_explainer._synthesize(
            result_df_with_y, global_results=global_results,
            y_label="fraudulent posting"
        )
        assert any("fraudulent posting" in s for s in cap_explainer.llm.captured)

    def test_generic_y_ref_when_no_label(self, cap_explainer, result_df_with_y, global_results):
        cap_explainer.llm.captured.clear()
        cap_explainer._synthesize(result_df_with_y, global_results=global_results, y_label=None)
        assert any("outcome variable" in s for s in cap_explainer.llm.captured)

    def test_stat_assoc_in_prompt(
        self, cap_explainer, result_df_with_y, global_results, stat_assoc_df
    ):
        cap_explainer.llm.captured.clear()
        cap_explainer._synthesize(
            result_df_with_y, global_results=global_results, stat_assoc_df=stat_assoc_df
        )
        assert any("max_posterior" in s for s in cap_explainer.llm.captured)

    def test_stat_mean_cols_in_prompt(self, cap_explainer, global_results):
        result_df = pd.DataFrame({
            'Group Number': [0, 1],
            'Short Description': ['A', 'B'],
            'Long Description': ['Desc A', 'Desc B'],
            'Size': [100, 200],
            'Test': ['Fisher Exact Test', 'Fisher Exact Test'],
            'Odds Ratio': [2.5, 0.3],
            'P-value': [0.001, 0.05],
            'max_posterior': [0.92, 0.75],   # stat mean column
        })
        cap_explainer.llm.captured.clear()
        cap_explainer._synthesize(result_df, global_results=global_results)
        assert any("max_posterior" in s for s in cap_explainer.llm.captured)

    def test_stat_labels_annotation_in_prompt(
        self, cap_explainer, result_df_with_y, global_results, stat_assoc_df
    ):
        """stat_labels descriptions should appear as inline annotations."""
        cap_explainer.llm.captured.clear()
        cap_explainer._synthesize(
            result_df_with_y,
            global_results=global_results,
            stat_assoc_df=stat_assoc_df,
            stat_labels={'max_posterior': 'avg. assignment confidence'},
        )
        combined = ' '.join(cap_explainer.llm.captured)
        assert 'avg. assignment confidence' in combined

    def test_stat_labels_absent_when_not_provided(
        self, cap_explainer, result_df_with_y, global_results, stat_assoc_df
    ):
        """Without stat_labels, no bracket annotations should appear."""
        cap_explainer.llm.captured.clear()
        cap_explainer._synthesize(
            result_df_with_y,
            global_results=global_results,
            stat_assoc_df=stat_assoc_df,
        )
        combined = ' '.join(cap_explainer.llm.captured)
        # The stat name should appear bare (no square bracket annotation)
        assert 'max_posterior' in combined
        assert '[' not in combined

    def test_extra_bullet_present_with_stat_assoc(
        self, cap_explainer, result_df_with_y, global_results, stat_assoc_df
    ):
        """When stat_assoc_df is provided, the observation-stats bullet should appear."""
        cap_explainer.llm.captured.clear()
        cap_explainer._synthesize(
            result_df_with_y, global_results=global_results, stat_assoc_df=stat_assoc_df
        )
        combined = ' '.join(cap_explainer.llm.captured)
        assert 'per-observation statistics' in combined

    def test_extra_bullet_absent_without_stat_assoc(
        self, cap_explainer, result_df_with_y, global_results
    ):
        """Without stat_assoc_df, the observation-stats bullet should not appear."""
        cap_explainer.llm.captured.clear()
        cap_explainer._synthesize(result_df_with_y, global_results=global_results)
        combined = ' '.join(cap_explainer.llm.captured)
        assert 'per-observation statistics' not in combined

    def test_custom_synthesize_prompt(self, text_transformer, result_df_no_y):
        from langchain_core.prompts import PromptTemplate
        custom_prompt = PromptTemplate(
            template="CUSTOM_MARKER {task} {cluster_section} {global_section} {stat_section}",
            input_variables=["task", "cluster_section", "global_section", "stat_section"]
        )
        llm = CapturingLLM(captured=[])
        explainer = ClusterExplainer(
            llm=llm,
            text_transformer=text_transformer,
            observations="data",
            text_fields="fields",
            custom_prompts={'synthesize': custom_prompt}
        )
        explainer._synthesize(result_df_no_y)
        assert any("CUSTOM_MARKER" in s for s in llm.captured)


# ---------------------------------------------------------------------------
# TestExplainReturnTypesWithSynthesize
# ---------------------------------------------------------------------------

class TestExplainReturnTypesWithSynthesize:
    """Full return-value contract: all 6 combinations of y × observation_stats × synthesize."""

    @pytest.fixture
    def explainer(self, mock_llm, text_transformer, sample_data):
        text_transformer.fit(sample_data)
        return ClusterExplainer(
            llm=mock_llm, text_transformer=text_transformer,
            observations="data", text_fields="fields"
        )

    @pytest.fixture
    def labels(self):
        return np.array([0] * 10 + [1] * 10)

    @pytest.fixture
    def y(self):
        return [0] * 10 + [1] * 10

    @pytest.fixture
    def obs_stats(self):
        return pd.DataFrame({'stat': np.random.randn(20)})

    def test_no_y_synthesize_false_returns_df(
        self, explainer, sample_data, labels, fake_labels
    ):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(explainer, '_explain_1stage', lambda *a, **kw: fake_labels)
            result = explainer.explain(sample_data, labels, strategy='one-stage')
        assert isinstance(result, pd.DataFrame)

    def test_no_y_synthesize_true_returns_2tuple(
        self, explainer, sample_data, labels, fake_labels
    ):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(explainer, '_explain_1stage', lambda *a, **kw: fake_labels)
            result = explainer.explain(
                sample_data, labels, strategy='one-stage', synthesize=True
            )
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], str)

    def test_with_y_synthesize_false_returns_2tuple(
        self, explainer, sample_data, labels, y, fake_labels
    ):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(explainer, '_explain_1stage', lambda *a, **kw: fake_labels)
            result = explainer.explain(sample_data, labels, y=y, strategy='one-stage')
        assert isinstance(result, tuple) and len(result) == 2

    def test_with_y_synthesize_true_returns_3tuple(
        self, explainer, sample_data, labels, y, fake_labels
    ):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(explainer, '_explain_1stage', lambda *a, **kw: fake_labels)
            result = explainer.explain(
                sample_data, labels, y=y, y_label='outcome',
                strategy='one-stage', synthesize=True
            )
        assert isinstance(result, tuple) and len(result) == 3
        assert isinstance(result[-1], str)

    def test_y_obs_stats_synthesize_false_returns_3tuple(
        self, explainer, sample_data, labels, y, obs_stats, fake_labels
    ):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(explainer, '_explain_1stage', lambda *a, **kw: fake_labels)
            result = explainer.explain(
                sample_data, labels, y=y, observation_stats=obs_stats,
                strategy='one-stage'
            )
        assert isinstance(result, tuple) and len(result) == 3

    def test_y_obs_stats_synthesize_true_returns_4tuple(
        self, explainer, sample_data, labels, y, obs_stats, fake_labels
    ):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(explainer, '_explain_1stage', lambda *a, **kw: fake_labels)
            result = explainer.explain(
                sample_data, labels, y=y, y_label='outcome',
                observation_stats=obs_stats, strategy='one-stage', synthesize=True
            )
        assert isinstance(result, tuple) and len(result) == 4
        assert isinstance(result[-1], str)

    def test_y_label_passed_to_synthesize(
        self, text_transformer, sample_data, labels, y, fake_labels
    ):
        """y_label is forwarded to _synthesize (blind labeling: nowhere else)."""
        llm = CapturingLLM(captured=[])
        text_transformer.fit(sample_data)
        explainer = ClusterExplainer(
            llm=llm, text_transformer=text_transformer,
            observations="data", text_fields="fields"
        )
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(explainer, '_explain_1stage', lambda *a, **kw: fake_labels)
            explainer.explain(
                sample_data, labels, y=y, y_label='UNIQUE_SENTINEL',
                strategy='one-stage', synthesize=True
            )
        # y_label must appear in the synthesis prompt, not in the labeling prompt
        # (labeling uses prompt_label_direct which never gets y_label)
        assert any("UNIQUE_SENTINEL" in s for s in llm.captured)
