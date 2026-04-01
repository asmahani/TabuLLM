# Copyright (c) 2024 Alireza S. Mahani and Mansour T.A. Sharabiani
# Licensed under the MIT License. See LICENSE file in the project root.

"""
LLM-based cluster explanation and interpretation.

Provides ClusterExplainer for generating natural language explanations of
text clusters using LLMs. Handles large datasets automatically via recursive
summarization, provides cost transparency, and supports prompt customization.

Key features:
- Auto strategy selection (one-stage vs two-stage)
- Recursive summarization (handles any dataset size)
- Token/cost estimation before LLM calls
- Customizable prompts (class and instance level)
- Statistical association tests (optional)

Classes
-------
GroupLabel : Single cluster description (Pydantic model)
MultipleGroupLabels : Collection of cluster labels with to_df() conversion
ClusterExplainer : Main class for generating cluster explanations
"""

from typing import List, Optional
import random
import pandas as pd
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency, f_oneway, ttest_ind, pearsonr
import warnings


class _SilentDict(dict):
    """dict subclass that suppresses REPL/notebook repr to avoid double-printing."""
    def __repr__(self):
        return ''


class GroupLabel(BaseModel):
    """
    Represents a label for a group in clustering analysis.

    Attributes
    ----------
    group_number : int
        The identifier for the group.
    description_short : str
        A short description or label for the group.
    description_long : str
        A detailed description of the group.
    """
    group_number: int
    description_short: str
    description_long: str

class MultipleGroupLabels(BaseModel):
    """
    Represents multiple group labels for clustering analysis.

    Attributes
    ----------
    groups : List[GroupLabel]
        A list of `GroupLabel` instances.

    Methods
    -------
    to_df() -> pd.DataFrame
        Converts the group labels to a pandas DataFrame.
    """
    groups: List[GroupLabel]
    
    def to_df(self) -> pd.DataFrame:
        """
        Convert the group labels to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the group labels.
        """
        if not self.groups:
            return pd.DataFrame(columns=['Group Number', 'Short Description', 'Long Description'])
        return (
            pd.DataFrame([group.model_dump() for group in self.groups])
            .sort_values('group_number')
            .rename(columns={
                'group_number': 'Group Number',
                'description_short': 'Short Description',
                'description_long': 'Long Description'
            })
            .reset_index(drop=True)
        )

class ClusterExplainer:
    """
    Generate LLM-based explanations for text data clusters.

    Provides automatic handling of large datasets through recursive summarization,
    cost transparency via token counting, and flexible prompt customization.

    Parameters
    ----------
    llm : BaseLanguageModel
        LangChain language model for generating explanations.
    text_transformer : TextColumnTransformer
        Fitted or unfitted TextColumnTransformer. Uses prep_X() for consistent text formatting.
    observations : str
        Description of what observations represent (e.g., "job postings", "patient records").
        Injected into prompts for LLM context.
    text_fields : str
        Description of text content (e.g., "titles and descriptions").
        Injected into prompts for LLM context.
    llm_context_length : int, default=200_000
        LLM context window in tokens (Claude: 200K, GPT-4: 128K, Gemini 1.5: 1M).
        Enables automatic recursion for large datasets.
    llm_cost_per_1M_tokens : float or None, default=None
        Cost in USD per 1M tokens (as of Feb 2026). If provided, cost estimates
        included in count_tokens_only output.
    custom_prompts : dict or None, default=None
        Override default prompts. Valid keys:

        - ``'label_direct'``: overrides ``prompt_label_direct`` (used in ``_explain_1stage``)
        - ``'label_from_summaries'``: overrides ``prompt_label_from_summaries`` (used in ``_explain_2stage`` stage 2)
        - ``'summarize_observations'``: overrides ``prompt_summarize_observations`` (used in ``_recursive_summarize`` at depth 0)
        - ``'combine_summaries'``: overrides ``prompt_combine_summaries`` (used in ``_recursive_summarize`` at depth > 0)
        - ``'synthesize'``: overrides ``prompt_synthesize`` (used in ``_synthesize`` when ``synthesize=True``)

    Examples
    --------
    >>> from tabullm import TextColumnTransformer, GMMFeatureExtractor, ClusterExplainer
    >>>
    >>> # Embed and cluster
    >>> transformer = TextColumnTransformer(model=emb_model)
    >>> X_embedded = transformer.fit_transform(df[['description', 'title']])
    >>> gmm = GMMFeatureExtractor(n_components=10)
    >>> cluster_labels = gmm.fit_predict(X_embedded)
    >>>
    >>> # Initialize explainer
    >>> explainer = ClusterExplainer(
    ...     llm=chat_model,
    ...     text_transformer=transformer,
    ...     observations="job postings",
    ...     text_fields="titles and descriptions",
    ...     llm_cost_per_1M_tokens=0.80
    ... )
    >>>
    >>> # Estimate cost first
    >>> info = explainer.explain(df, cluster_labels, count_tokens_only=True)
    >>> print(f"Cost: ${info['estimated_cost']:.2f}")
    >>>
    >>> # Generate explanations
    >>> explanations = explainer.explain(df, cluster_labels, strategy='auto')
    """
    
    parser = PydanticOutputParser(pydantic_object=MultipleGroupLabels)
    CONTEXT_SAFETY_MARGIN = 0.8  # Fraction of context window used as effective limit

    prompt_label_direct = PromptTemplate(
        template=(
            "Below are {nobs} {observations} organised into {ngroups} groups. "
            "Each entry contains {text_fields}. "
            "Assign each group a concise label and a description of no more than {nwords} words "
            "that captures what makes that group distinctive. "
            "Labels should be mutually distinct across groups.\n{format_instructions}\n\n{payload}"
        ),
        input_variables=["nobs", "observations", "text_fields", "ngroups", "nwords", "payload"],
        partial_variables={'format_instructions': parser.get_format_instructions()}
    )

    prompt_label_from_summaries = PromptTemplate(
        template=(
            "Below are summaries of {ngroups} groups of {observations}, "
            "characterised by their {text_fields}. "
            "Assign each group a concise label and a description of no more than {nwords} words "
            "that captures what makes that group distinctive. "
            "Labels should be mutually distinct across groups.\n{format_instructions}\n\n{payload}"
        ),
        input_variables=['ngroups', 'observations', 'text_fields', 'nwords', 'payload'],
        partial_variables={'format_instructions': parser.get_format_instructions()}
    )

    prompt_summarize_observations = PromptTemplate(
        template=(
            "Below is a sample of {observations}. Each entry contains {text_fields}. "
            "Write a summary of no more than {nwords} words that captures the common themes "
            "and distinguishing characteristics of this group.\n\n{payload}"
        ),
        input_variables=["observations", "text_fields", "nwords", "payload"]
    )

    prompt_combine_summaries = PromptTemplate(
        template=(
            "The following are partial summaries of the same group of {observations}. "
            "Write a single coherent summary of no more than {nwords} words that captures "
            "the common themes and distinguishing characteristics of the group.\n\n{payload}"
        ),
        input_variables=["observations", "nwords", "payload"]
    )

    prompt_synthesize = PromptTemplate(
        template=(
            "{task}\n\n"
            "--- Cluster descriptions ---\n"
            "{cluster_section}\n"
            "{global_section}"
            "{stat_section}"
        ),
        input_variables=["task", "cluster_section", "global_section", "stat_section"]
    )

    def __init__(
        self,
        llm,
        text_transformer,
        observations: str,
        text_fields: str,
        llm_context_length: int = 200_000,
        llm_cost_per_1M_tokens: float = None,
        custom_prompts: dict = None
    ):
        """
        Initialize the ClusterExplainer.

        Parameters
        ----------
        llm : BaseLanguageModel
            The language model to use for generating explanations.
        text_transformer : TextColumnTransformer
            Fitted or unfitted TextColumnTransformer for consistent text formatting.
            Uses prep_X() method to format DataFrame text columns.
        observations : str
            A description of what the observations represent (e.g., "job postings").
        text_fields : str
            A description of the text fields in the data (e.g., "titles and descriptions").
        llm_context_length : int, default=200_000
            LLM context window size in tokens. Default 200K (Claude, Gemini Pro).
            Used for automatic recursion when data exceeds limit.
        llm_cost_per_1M_tokens : float or None, default=None
            Cost in USD per 1 million tokens (as of Feb 2026).
            If provided, cost estimates included in count_tokens_only output.
            Examples: Claude Haiku (0.80), GPT-4 (15.00), Gemini (0.35)
        custom_prompts : dict or None, default=None
            Optional custom prompt templates. Valid keys: 'label_direct', 'label_from_summaries',
            'summarize_observations', 'combine_summaries'
        """
        self.llm = llm
        self.text_transformer = text_transformer
        self.observations = observations
        self.text_fields = text_fields
        self.llm_context_length = llm_context_length
        self.llm_cost_per_1M_tokens = llm_cost_per_1M_tokens

        # Validate parameters
        if not hasattr(text_transformer, 'prep_X'):
            raise TypeError('text_transformer must have prep_X method')
        if llm_context_length <= 0:
            raise ValueError('llm_context_length must be positive')
        if llm_cost_per_1M_tokens is not None and llm_cost_per_1M_tokens < 0:
            raise ValueError('llm_cost_per_1M_tokens must be non-negative')

        # Handle custom prompts (override class-level templates if provided)
        if custom_prompts is not None:
            if 'label_direct' in custom_prompts:
                self.prompt_label_direct = custom_prompts['label_direct']
            if 'label_from_summaries' in custom_prompts:
                self.prompt_label_from_summaries = custom_prompts['label_from_summaries']
            if 'summarize_observations' in custom_prompts:
                self.prompt_summarize_observations = custom_prompts['summarize_observations']
            if 'combine_summaries' in custom_prompts:
                self.prompt_combine_summaries = custom_prompts['combine_summaries']
            if 'synthesize' in custom_prompts:
                self.prompt_synthesize = custom_prompts['synthesize']

    @staticmethod
    def _validate_y(y):
        """
        Validate the outcome variable and return its detected type.

        Parameters
        ----------
        y : array-like
            Outcome variable to validate.

        Returns
        -------
        str
            ``'binary'`` if y has exactly 2 unique non-NaN values or boolean dtype;
            ``'continuous'`` if y has more than 2 unique numeric values.

        Raises
        ------
        ValueError
            If y is all-NaN, has fewer than 2 unique non-NaN values, has only 1
            unique value, or has a non-numeric dtype.
        """
        s = pd.Series(y)
        non_null = s.dropna()

        if len(non_null) == 0:
            raise ValueError(
                "y contains only NaN values. Provide y with at least one non-null value."
            )

        if pd.api.types.is_bool_dtype(s):
            return 'binary'

        if pd.api.types.is_numeric_dtype(s):
            n_unique = non_null.nunique()
            if n_unique == 1:
                raise ValueError(
                    "y has only 1 unique non-null value. At least 2 unique values are required."
                )
            return 'binary' if n_unique == 2 else 'continuous'

        raise ValueError(
            f"y has dtype '{s.dtype}', which is not supported. "
            "Non-numeric y must be encoded as numeric before passing "
            "(e.g., use 0 for the negative class and 1 for the positive class)."
        )

    def preview_prompts(self) -> dict:
        """
        Print and return all fully rendered prompt strings.

        Useful for inspecting what will be sent to the LLM before calling explain().
        Injects the instance's ``observations`` and ``text_fields``; remaining
        variables are filled with descriptive placeholders.

        Returns
        -------
        dict[str, str]
            Rendered prompt strings keyed by prompt name:
            ``'label_direct'``, ``'summarize_observations'``,
            ``'combine_summaries'``, ``'label_from_summaries'``.
        """
        prompts = _SilentDict({
            'label_direct': self.prompt_label_direct.format(
                observations=self.observations,
                text_fields=self.text_fields,
                nobs='<N>',
                ngroups='<K>',
                nwords='<max_words>',
                payload='<... cluster data ...>'
            ),
            'summarize_observations': self.prompt_summarize_observations.format(
                observations=self.observations,
                text_fields=self.text_fields,
                nwords='<max_words>',
                payload='<... cluster data ...>'
            ),
            'combine_summaries': self.prompt_combine_summaries.format(
                observations=self.observations,
                nwords='<max_words>',
                payload='<... summaries ...>'
            ),
            'label_from_summaries': self.prompt_label_from_summaries.format(
                observations=self.observations,
                text_fields=self.text_fields,
                ngroups='<K>',
                nwords='<max_words>',
                payload='<... cluster summaries ...>'
            ),
            'synthesize': self.prompt_synthesize.format(
                task=(
                    f'Below are <K> clusters of {self.observations}, described from their '
                    f'{self.text_fields} WITHOUT prior knowledge of "<y_label>". '
                    f'Statistical associations with "<y_label>" are shown alongside each cluster.\n\n'
                    f'Write a coherent analytical narrative of no more than <max_words> words that:\n'
                    f'1. Summarises the overall landscape of clusters\n'
                    f'2. Identifies which clusters are most strongly associated with "<y_label>"\n'
                    f'3. Explains what content characteristics might account for these associations'
                ),
                cluster_section='<... cluster rows ...>',
                global_section='<... global test result ...>\n\n',
                stat_section='<... stat association table ...>\n\n',
            ),
        })
        for name, text in prompts.items():
            print(f"{name}")
            print("─" * len(name))
            print(text)
            print()
        return prompts

    def _sample_cluster(self, items, max_members, random_state):
        """
        Randomly subsample cluster members.

        Parameters
        ----------
        items : list
            Cluster members to sample from.
        max_members : int or None
            Maximum number of members to keep. If None or >= len(items), returns items unchanged.
        random_state : int or None
            Seed for reproducibility. None means non-deterministic.

        Returns
        -------
        list
            Subsampled (or original) list of items.
        """
        if max_members is None or len(items) <= max_members:
            return items
        return random.Random(random_state).sample(items, max_members)

    def _recursive_summarize(self, items, depth=0, max_tokens=None, max_words=200, verbose=False):
        """
        Recursively summarize items until fits in context.

        Uses depth-aware prompts: different templates for raw observations vs summaries.
        Implements true recursion: splits until fits, handles all corner cases.

        Parameters
        ----------
        items : List[str]
            Observations (depth=0) or summaries (depth>0)
        depth : int, default=0
            Recursion depth (0=observations, >0=summaries)
        max_tokens : int or None
            Context limit. If None, uses self.llm_context_length * 0.8
        max_words : int, default=200
            Max words for output summary

        Returns
        -------
        str : Summary text

        Raises
        ------
        ValueError : If single item exceeds context (cannot split further)
        """
        if max_tokens is None:
            max_tokens = int(self.llm_context_length * self.CONTEXT_SAFETY_MARGIN)

        estimated = self._estimate_tokens(items)

        if estimated < max_tokens:
            # Base case: fits in context
            if verbose and depth > 0:
                print(f"  [Stage {depth}] Combining {len(items)} summaries")

            if depth == 0:
                # Summarizing raw observations
                prompt_template = self.prompt_summarize_observations
                prompt_args = {
                    "observations": self.observations,
                    "text_fields": self.text_fields,
                    "nwords": max_words,
                    "payload": '\n'.join(items)
                }
            else:
                # Combining summaries
                prompt_template = self.prompt_combine_summaries
                prompt_args = {
                    "observations": self.observations,
                    "nwords": max_words,
                    "payload": '\n\n'.join(items)
                }

            # Invoke LLM
            chain = prompt_template | self.llm
            try:
                summary = chain.invoke(prompt_args)
                # Extract content (LLM might return AIMessage object)
                if hasattr(summary, 'content'):
                    return summary.content
                return str(summary)
            except Exception as e:
                raise RuntimeError(f"LLM invocation failed at depth {depth}: {e}")

        else:
            # Recursive case: split and recurse
            mid = len(items) // 2

            if mid == 0:
                # Single item too large - cannot split further
                raise ValueError(
                    f"Single item ({estimated:,} tokens) exceeds context limit ({max_tokens:,} tokens). "
                    f"Suggestions:\n"
                    f"  1. Reduce text columns (use fewer or shorter columns)\n"
                    f"  2. Truncate individual texts before passing to explain()\n"
                    f"  3. Use a model with larger context (e.g., Gemini 1.5: 1M tokens)\n"
                    f"Cannot proceed with this data size."
                )

            if verbose:
                print(f"  [Stage {depth}] Splitting {len(items)} items ({estimated:,} tokens > {max_tokens:,})")

            # Recurse on each half
            left_summary = self._recursive_summarize(items[:mid], depth, max_tokens, max_words, verbose)
            right_summary = self._recursive_summarize(items[mid:], depth, max_tokens, max_words, verbose)

            # Combine summaries (recurse at depth+1)
            return self._recursive_summarize([left_summary, right_summary], depth+1, max_tokens, max_words, verbose)

    def _estimate_tokens(self, items, chars_per_token=4):
        """
        Estimate token count for list of text items.

        Parameters
        ----------
        items : List[str]
            Text items (observations or summaries)
        chars_per_token : float, default=4
            Character-to-token ratio heuristic

        Returns
        -------
        int : Estimated token count
        """
        total_chars = sum(len(item) for item in items)
        return int(total_chars / chars_per_token)

    def _explain_1stage(
        self,
        Xstr: List[str],
        cluster_labels,
        max_label_words=50,
        max_members=None,
        random_state=None,
        verbose=False
    ):
        """
        Generate cluster explanations using one-stage process (internal method).

        Parameters
        ----------
        Xstr : List[str]
            Pre-formatted text data (from text_transformer.prep_X)
        cluster_labels : array-like
            Cluster assignments
        max_label_words : int
            Maximum words for cluster labels
        max_members : int or None
            Maximum cluster members to include. None uses all members.
        random_state : int or None
            Seed for subsampling reproducibility.
        verbose : bool
            Print progress

        Returns
        -------
        MultipleGroupLabels : Pydantic object with cluster descriptions
        """
        unique_labels = sorted(set(cluster_labels))
        group_payloads = []
        for label in unique_labels:
            cluster_data = [Xstr[i] for i in range(len(Xstr)) if cluster_labels[i] == label]
            cluster_data = self._sample_cluster(cluster_data, max_members, random_state)
            group_payloads.append('\n'.join(cluster_data))
        payload_1stage = '\n\n'.join([f'Group {n}:\n\n{payload}' for n, payload in zip(unique_labels, group_payloads)])

        prompt_args = {
            "ngroups": len(unique_labels),
            "observations": self.observations,
            "text_fields": self.text_fields,
            "nwords": max_label_words,
            "payload": payload_1stage,
            "nobs": len(Xstr)
        }

        chain_1stage = self.prompt_label_direct | self.llm | self.parser

        try:
            group_labels = chain_1stage.invoke(prompt_args)
            return group_labels
        except Exception as e:
            raise RuntimeError(f"Error during LLM invocation: {e}")

    def _explain_2stage(
        self,
        Xstr: List[str],
        cluster_labels,
        max_summary_words=200,
        max_label_words=50,
        max_members=None,
        random_state=None,
        verbose=False
    ):
        """
        Generate cluster explanations using two-stage process with automatic recursion (internal).

        Uses recursive summarization to handle clusters exceeding context length.

        Parameters
        ----------
        Xstr : List[str]
            Pre-formatted text data (from text_transformer.prep_X)
        cluster_labels : array-like
            Cluster assignments
        max_summary_words : int
            Maximum words for cluster summaries (stage 1)
        max_label_words : int
            Maximum words for final labels (stage 2)
        max_members : int or None
            Maximum cluster members to include. None uses all members.
        random_state : int or None
            Seed for subsampling reproducibility.
        verbose : bool
            Print progress

        Returns
        -------
        MultipleGroupLabels : Pydantic object with cluster descriptions

        Notes
        -----
        Automatically uses recursive summarization if individual clusters
        exceed context length. True recursion: splits until fits.
        """
        unique_labels = sorted(set(cluster_labels))
        group_summaries = []

        # Stage 1: Summarize each cluster (with automatic recursion)
        for label in unique_labels:
            cluster_data = [Xstr[i] for i in range(len(Xstr)) if cluster_labels[i] == label]
            cluster_data = self._sample_cluster(cluster_data, max_members, random_state)

            # Use recursive summarization (automatically handles large clusters)
            summary = self._recursive_summarize(
                cluster_data,
                depth=0,  # These are raw observations
                max_tokens=int(self.llm_context_length * self.CONTEXT_SAFETY_MARGIN),
                max_words=max_summary_words,
                verbose=verbose
            )

            group_summaries.append(summary)

            if verbose:
                print(f'Processed cluster {label}')

        # Stage 2: Combine cluster summaries into labels
        # Check if combination exceeds context, recurse if needed
        if self._estimate_tokens(group_summaries) > self.llm_context_length * self.CONTEXT_SAFETY_MARGIN:
            # Too many summaries, recursively combine
            if verbose:
                print(f'Cluster summaries exceed context, using recursion...')

            combined_summary = self._recursive_summarize(
                group_summaries,
                depth=1,  # These are summaries
                max_tokens=int(self.llm_context_length * self.CONTEXT_SAFETY_MARGIN),
                max_words=max_label_words
            )

            # This gives us one combined summary, need to parse it differently
            # For now, raise error - this edge case needs more design
            raise NotImplementedError(
                "Recursive combination of cluster summaries not yet implemented. "
                "Too many clusters for context. Reduce number of clusters."
            )

        # Standard case: all cluster summaries fit in context
        payload_stage2 = '\n\n'.join([f'Group {n}:\n\n{summary}' for n, summary in zip(unique_labels, group_summaries)])
        prompt_args_stage2 = {
            "ngroups": len(unique_labels),
            "observations": self.observations,
            "text_fields": self.text_fields,
            "nwords": max_label_words,
            "payload": payload_stage2
        }

        chain_stage2 = self.prompt_label_from_summaries | self.llm | self.parser

        try:
            group_labels = chain_stage2.invoke(prompt_args_stage2)
            return group_labels
        except Exception as e:
            raise RuntimeError(f"Error during LLM invocation in stage 2: {e}")

    def _synthesize(
        self,
        result_df: pd.DataFrame,
        global_results: pd.DataFrame = None,
        stat_assoc_df: pd.DataFrame = None,
        y_label: str = None,
        stat_labels: dict = None,
        max_words: int = 500,
    ) -> str:
        """
        Generate a synthesis narrative from explain() output using the LLM.

        Parameters
        ----------
        result_df : pd.DataFrame
            Output from the main explain() pipeline (cluster labels + merged stats).
        global_results : pd.DataFrame or None
            Global association test result (one row). None when y was not provided.
        stat_assoc_df : pd.DataFrame or None
            Per-stat association table from _stat_associations(). None when not computed.
        y_label : str or None
            Natural-language label for the outcome variable. Used only in the task
            instruction; withheld from all upstream labeling prompts.
        stat_labels : dict or None
            Human-readable descriptions for observation_stats columns. Keys are column
            names; values are short descriptions used as inline annotations in the
            synthesis prompt to give the LLM context about what each statistic means.
        max_words : int, default=500
            Maximum words for the synthesis narrative.

        Returns
        -------
        str
            Synthesis narrative text from the LLM.
        """
        n_groups = len(result_df)
        has_outcome = global_results is not None
        has_stat_assoc = stat_assoc_df is not None and not stat_assoc_df.empty
        stat_labels = stat_labels or {}

        # Columns that are part of the structural / statistical schema, not stat means
        KNOWN_COLS = {
            'Group Number', 'Short Description', 'Long Description',
            'Size', 'Test', 'Odds Ratio', 'T-Statistic', 'P-value',
        }
        stat_mean_cols = [c for c in result_df.columns if c not in KNOWN_COLS]

        def _annotate(col):
            """Return 'col [description]' if a label is available, else 'col'."""
            desc = stat_labels.get(col)
            return f"{col} [{desc}]" if desc else col

        # --- cluster section ---
        cluster_lines = []
        for _, row in result_df.iterrows():
            header = f"Group {row['Group Number']}"
            if 'Size' in result_df.columns and pd.notna(row.get('Size')):
                header += f" (n={int(row['Size'])})"
            header += f': "{row["Short Description"]}"'
            cluster_lines.append(header)
            cluster_lines.append(f"  {row['Long Description']}")

            if has_outcome and 'P-value' in result_df.columns and pd.notna(row.get('P-value')):
                if 'Odds Ratio' in result_df.columns and pd.notna(row.get('Odds Ratio')):
                    cluster_lines.append(
                        f"  Outcome association: OR={row['Odds Ratio']:.3f}, p={row['P-value']:.4f}"
                    )
                elif 'T-Statistic' in result_df.columns and pd.notna(row.get('T-Statistic')):
                    cluster_lines.append(
                        f"  Outcome association: t={row['T-Statistic']:.3f}, p={row['P-value']:.4f}"
                    )

            if stat_mean_cols:
                stat_strs = [
                    f"{_annotate(c)}={row[c]:.3f}"
                    for c in stat_mean_cols
                    if pd.notna(row.get(c))
                ]
                if stat_strs:
                    cluster_lines.append(f"  Cluster stats: {', '.join(stat_strs)}")

            cluster_lines.append("")  # blank line between clusters

        cluster_section = '\n'.join(cluster_lines)

        # --- global test section ---
        global_section = ""
        if global_results is not None and not global_results.empty:
            g = global_results.iloc[0]
            stat_str = f"{g['Statistic']:.3f}" if pd.notna(g.get('Statistic')) else 'N/A'
            p_str = f"{g['P-value']:.4f}" if pd.notna(g.get('P-value')) else 'N/A'
            global_section = f"Global test ({g['Test']}): statistic={stat_str}, p={p_str}\n\n"

        # --- stat association section ---
        stat_section = ""
        if has_stat_assoc:
            # Detect group-mean columns produced by _stat_associations for binary y
            mean_cols = [c for c in stat_assoc_df.columns if str(c).startswith('Mean (y=')]
            lines = ["Association of per-observation statistics with outcome:"]
            for _, row in stat_assoc_df.iterrows():
                stat_str = f"{row['Statistic']:.3f}" if pd.notna(row.get('Statistic')) else 'N/A'
                p_str = f"{row['P-value']:.4f}" if pd.notna(row.get('P-value')) else 'N/A'
                means_str = (
                    ', '.join(f"{c}={row[c]:.3f}" for c in mean_cols if pd.notna(row.get(c)))
                    if mean_cols else ''
                )
                detail = f"{means_str}, t={stat_str}" if means_str else f"r={stat_str}"
                lines.append(
                    f"  {_annotate(row['Stat'])} — {row['Test']}: {detail}, p={p_str}"
                )
            stat_section = '\n'.join(lines) + '\n\n'

        # --- task instruction (changes based on whether outcome data is present) ---
        y_ref = f'"{y_label}"' if y_label else "the outcome variable"
        if has_outcome:
            bullets = [
                f"1. Summarise the overall landscape of clusters — what are the main themes "
                f"and how do they differ from one another?",
                f"2. Interpret the cluster-level associations with {y_ref}: which clusters are "
                f"enriched or depleted, how strong is the evidence (cite odds ratios or "
                f"t-statistics where relevant), and is the pattern coherent across clusters?",
            ]
            if has_stat_assoc:
                bullets.append(
                    f"3. Discuss what the per-observation statistics and their associations with "
                    f"{y_ref} reveal — e.g. whether outcome-associated clusters show "
                    f"systematically higher or lower values, and what that implies."
                )
            bullets.append(
                f"{len(bullets) + 1}. Explain what content characteristics distinguish clusters "
                f"with strong {y_ref} signal from those without."
            )
            task = (
                f"Below are {n_groups} clusters of {self.observations}, described from their "
                f"{self.text_fields} WITHOUT prior knowledge of {y_ref}. "
                f"Statistical associations with {y_ref} are shown alongside each cluster.\n\n"
                f"Write a coherent analytical narrative of no more than {max_words} words that:\n"
                + '\n'.join(bullets)
            )
        else:
            task = (
                f"Below are {n_groups} clusters of {self.observations}, "
                f"described from their {self.text_fields}.\n\n"
                f"Write a coherent narrative of no more than {max_words} words that summarises "
                f"the overall landscape: what the main themes are, how clusters differ from each "
                f"other, and what the data as a whole looks like."
            )

        chain = self.prompt_synthesize | self.llm
        try:
            result = chain.invoke({
                "task": task,
                "cluster_section": cluster_section,
                "global_section": global_section,
                "stat_section": stat_section,
            })
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed during synthesis: {e}")

    def explain(
        self,
        X,
        cluster_labels,
        y=None,
        y_label=None,
        strategy: str = 'auto',
        max_summary_words: int = 200,
        max_label_words: int = 50,
        max_members_per_cluster: int = None,
        random_state: int = None,
        observation_stats: pd.DataFrame = None,
        stat_labels: dict = None,
        correction: str = None,
        synthesize: bool = False,
        preview: bool = False,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Generate cluster explanations using LLM with automatic context handling.

        Parameters
        ----------
        X : DataFrame
            Original data with text columns. Uses text_transformer.prep_X() for formatting.
        cluster_labels : array-like
            Cluster assignments for each sample.
        y : array-like or None, default=None
            Optional outcome variable for statistical association tests.
            Accepted formats:

            - Numeric with exactly 2 unique non-NaN values → binary
              (Fisher's exact test per cluster, Chi-square globally)
            - Boolean dtype → binary
            - Numeric with more than 2 unique values → continuous
              (independent t-test per cluster, one-way ANOVA globally)

            Non-numeric y (strings, categories) is not supported; encode as
            numeric before passing (e.g., 0/1 for binary labels).
            Validated at call time with a descriptive error for unsupported inputs.
        y_label : str or None, default=None
            Natural-language description of the outcome variable
            (e.g., ``"fraudulent job posting (1=fraud, 0=legitimate)"``).
            Used **only** in the synthesis step when ``synthesize=True``.
            Deliberately withheld from all cluster-labeling prompts so that
            cluster descriptions reflect text content, not the outcome.
            When ``y`` is provided but ``y_label`` is omitted, synthesis uses
            the generic phrase "the outcome variable".
        strategy : {'auto', 'one-stage', 'two-stage'}, default='auto'
            Explanation strategy:
            - 'auto': Intelligently selects based on data size
            - 'one-stage': Direct explanation of all clusters
            - 'two-stage': Hierarchical (cluster summaries → final labels)
        max_summary_words : int, default=200
            Maximum words for cluster summaries (stage 1 in two-stage).
        max_label_words : int, default=50
            Maximum words for final cluster labels.
        max_members_per_cluster : int or None, default=None
            Maximum number of members to include per cluster. If a cluster has more
            members, a random subsample of this size is used. None uses all members.
        random_state : int or None, default=None
            Seed for cluster subsampling reproducibility. Only relevant when
            max_members_per_cluster is set.
        observation_stats : pd.DataFrame or None, default=None
            Optional per-observation statistics, one row per sample in X.
            When provided:

            - Cluster-level means are computed and appended as columns to the
              output table (always).
            - Each column is tested for correlation with ``y`` when ``y`` is
              also provided: t-test for binary ``y``, Pearson correlation for
              continuous ``y``. Results are returned as a third element.

            Intended use: pass ``GMMFeatureExtractor.assignment_confidence_stats(X)``
            here to include GMM-derived diagnostics in the explanation output.
        stat_labels : dict or None, default=None
            Human-readable descriptions for ``observation_stats`` columns, used as
            inline annotations in the synthesis prompt so the LLM knows what each
            statistic measures.  Keys are column names; values are short descriptions.
            Ignored when ``synthesize=False`` or ``observation_stats`` is None.
            Example::

                stat_labels = {
                    'max_posterior':    'avg. cluster assignment confidence',
                    'entropy':          'avg. membership entropy across clusters',
                    'log_joint_margin': 'avg. log-joint margin over nearest rival',
                }
        correction : {'bonferroni', 'holm', 'fdr_bh'} or None, default=None
            Multiple testing correction for the K per-cluster hypothesis tests.
            ``'bonferroni'`` is the most conservative; ``'holm'`` controls the
            family-wise error rate with greater power; ``'fdr_bh'`` (Benjamini-Hochberg)
            controls the false discovery rate and is recommended for exploratory work.
            When provided, a ``P-value (adjusted)`` column is appended to ``result_df``
            (and to ``stat_assoc_df`` when ``observation_stats`` is also supplied).
            The global association p-value is a single omnibus test and is not adjusted.
            Ignored when ``y=None``.
        synthesize : bool, default=False
            If True, performs a final LLM pass that interprets the full output
            (cluster labels, association statistics, observation_stats results)
            as a coherent narrative.  The synthesis prompt uses ``y_label`` when
            ``y`` is also provided; without ``y`` it produces a descriptive
            landscape summary.  The synthesis text is appended as the last
            element of the return tuple (see Returns below).
        preview : bool, default=False
            If True, prints a cost/strategy summary and returns without calling the LLM.
            Useful for inspecting estimated tokens, cost, and auto-selected strategy
            before committing to a full run.
        verbose : bool, default=False
            Print progress messages during processing.

        Returns
        -------
        DataFrame or tuple or dict
            If ``preview=True``:
                dict with keys: total_tokens, estimated_cost, strategy
            If ``preview=False`` and ``y=None`` and ``observation_stats=None``:
                - ``synthesize=False``: DataFrame with columns: Group Number, Short Description, Long Description
                - ``synthesize=True``: ``(result_df, synthesis_text)``
            If ``preview=False`` and ``y=None`` and ``observation_stats`` provided:
                - ``synthesize=False``: DataFrame with cluster-mean stat columns appended
                - ``synthesize=True``: ``(result_df, synthesis_text)``
            If ``preview=False`` and ``y`` provided and ``observation_stats=None``:
                - ``synthesize=False``: ``(result_df, global_results)``
                - ``synthesize=True``: ``(result_df, global_results, synthesis_text)``
            If ``preview=False`` and both ``y`` and ``observation_stats`` provided:
                - ``synthesize=False``: ``(result_df, global_results, stat_assoc_df)``
                - ``synthesize=True``: ``(result_df, global_results, stat_assoc_df, synthesis_text)``

            ``synthesis_text`` is a string containing the LLM-generated narrative.
            When ``synthesize=False`` (the default), return shapes are identical to
            pre-synthesis behaviour.

        Notes
        -----
        Uses text_transformer.prep_X() for consistent text formatting.
        Automatically handles large datasets with recursive summarization.
        Strategy selection (when strategy='auto') is based on full unsampled data size.
        Token count shown in preview reflects subsampling if max_members_per_cluster is set.
        """
        # Prepare text using text_transformer for consistent formatting
        Xstr = self.text_transformer.prep_X(X)

        # Validate y early so errors surface at the entry point, not deep in one_vs_rest
        y_type = self._validate_y(y) if y is not None else None

        # Validate observation_stats
        if observation_stats is not None:
            if not isinstance(observation_stats, pd.DataFrame):
                raise TypeError('observation_stats must be a pandas DataFrame or None')
            if len(observation_stats) != len(Xstr):
                raise ValueError(
                    f'observation_stats has {len(observation_stats)} rows but X has '
                    f'{len(Xstr)} rows. They must match.'
                )

        # Token count for full dataset (used for strategy selection and one-stage warning)
        all_data_tokens = self._estimate_tokens(Xstr)

        # Token count for reporting (respects subsampling if set)
        if max_members_per_cluster is not None:
            unique_labels = sorted(set(cluster_labels))
            sampled = []
            for label in unique_labels:
                cluster_data = [Xstr[i] for i in range(len(Xstr)) if cluster_labels[i] == label]
                sampled.extend(self._sample_cluster(cluster_data, max_members_per_cluster, random_state))
            total_tokens = self._estimate_tokens(sampled)
        else:
            total_tokens = all_data_tokens

        # Strategy selection (always based on full unsampled data)
        ONE_STAGE_THRESHOLD = 0.3   # Use one-stage if < 30% of context
        if strategy == 'auto':
            actual_strategy = (
                'one-stage' if all_data_tokens < self.llm_context_length * ONE_STAGE_THRESHOLD
                else 'two-stage'
            )
        else:
            actual_strategy = strategy

        # Preview mode: print summary and return without calling LLM
        if preview:
            estimated_cost = (
                (total_tokens / 1_000_000) * self.llm_cost_per_1M_tokens
                if self.llm_cost_per_1M_tokens is not None else None
            )
            strategy_tag = ' (auto-selected)' if strategy == 'auto' else ''
            print("Preview")
            print("───────")
            print(f"Tokens (estimated):   {total_tokens:,}")
            if estimated_cost is not None:
                print(f"Cost (estimated):     ${estimated_cost:.4f}")
            print(f"Strategy:             {actual_strategy}{strategy_tag}")
            if max_members_per_cluster is not None:
                print(f"Max members/cluster:  {max_members_per_cluster:,}")
            if synthesize:
                print(f"Synthesis:            enabled")
            print()
            return {
                'total_tokens': total_tokens,
                'estimated_cost': estimated_cost,
                'strategy': actual_strategy
            }

        if verbose:
            strategy_tag = ' (auto-selected)' if strategy == 'auto' else ''
            print(f"Strategy: {actual_strategy}{strategy_tag} ({total_tokens:,} tokens)")

        # Warn if one-stage may exceed context (but allow user to proceed)
        if actual_strategy == 'one-stage' and all_data_tokens > self.llm_context_length:
            warnings.warn(
                f"Estimated payload ({all_data_tokens:,} tokens) may exceed context limit "
                f"({self.llm_context_length:,} tokens). "
                f"Consider strategy='auto' or 'two-stage' for automatic recursion. "
                f"Note: Estimate is conservative, actual may be lower.",
                UserWarning
            )

        # Generate explanation using selected strategy
        if actual_strategy == 'two-stage':
            explanation_obj = self._explain_2stage(
                Xstr, cluster_labels,
                max_summary_words=max_summary_words,
                max_label_words=max_label_words,
                max_members=max_members_per_cluster,
                random_state=random_state,
                verbose=verbose
            )
        else:  # one-stage
            explanation_obj = self._explain_1stage(
                Xstr, cluster_labels,
                max_label_words=max_label_words,
                max_members=max_members_per_cluster,
                random_state=random_state,
                verbose=verbose
            )

        # Start with base explanation DataFrame
        result_df = explanation_obj.to_df()

        # Append cluster-mean observation_stats columns (always, when provided)
        if observation_stats is not None:
            stat_means = (
                observation_stats
                .assign(_cluster=list(cluster_labels))
                .groupby('_cluster')[list(observation_stats.columns)]
                .mean()
                .reset_index()
                .rename(columns={'_cluster': 'Group Number'})
            )
            result_df = result_df.merge(stat_means, on='Group Number', how='left')

        # One-vs-rest analysis (only when y is provided)
        global_results = None
        if y is not None:
            data_for_analysis = pd.DataFrame({"Cluster": cluster_labels, "Outcome": y})
            one_vs_rest_results, global_results = one_vs_rest(
                data_for_analysis, col_x="Cluster", col_y="Outcome",
                y_type=y_type, correction=correction
            )
            result_df = pd.merge(
                result_df,
                one_vs_rest_results,
                left_on="Group Number",
                right_on="Category",
                how="left"
            ).drop(columns=["Category"])

        # stat_assoc (only when both y and observation_stats are provided)
        stat_assoc_df = None
        if y is not None and observation_stats is not None:
            stat_assoc_df = _stat_associations(observation_stats, y, y_type, correction=correction)

        # Synthesis step (optional — uses y_label only here, never in labeling prompts)
        synthesis_text = None
        if synthesize:
            synthesis_text = self._synthesize(
                result_df,
                global_results=global_results,
                stat_assoc_df=stat_assoc_df,
                y_label=y_label,
                stat_labels=stat_labels,
            )

        # Build return value
        if y is None:
            return (result_df, synthesis_text) if synthesize else result_df

        if observation_stats is None:
            return (result_df, global_results, synthesis_text) if synthesize else (result_df, global_results)

        return (result_df, global_results, stat_assoc_df, synthesis_text) if synthesize else (result_df, global_results, stat_assoc_df)

_VALID_CORRECTIONS = ('bonferroni', 'holm', 'fdr_bh')


def _adjust_pvalues(pvalues, method):
    """Apply multiple testing correction to a list of p-values.

    Parameters
    ----------
    pvalues : list
        Raw p-values; entries may be None (untestable comparisons are preserved).
    method : {'bonferroni', 'holm', 'fdr_bh'}
        Correction method.

    Returns
    -------
    list
        Adjusted p-values, clipped to [0, 1].  None entries are preserved.
    """
    if method not in _VALID_CORRECTIONS:
        raise ValueError(
            f"Unknown correction method '{method}'. "
            f"Choose from {_VALID_CORRECTIONS}."
        )

    valid_idx = [i for i, p in enumerate(pvalues) if p is not None and not np.isnan(p)]
    m = len(valid_idx)
    if m == 0:
        return list(pvalues)

    raw = np.array([pvalues[i] for i in valid_idx], dtype=float)

    if method == 'bonferroni':
        adj = np.minimum(raw * m, 1.0)

    elif method == 'holm':
        # Sort ascending; multiply by (m, m-1, ..., 1); cumulative max; clip
        order = np.argsort(raw)
        multiplied = np.minimum(raw[order] * np.arange(m, 0, -1), 1.0)
        adj_sorted = np.maximum.accumulate(multiplied)
        adj = np.empty(m)
        adj[order] = adj_sorted

    else:  # fdr_bh
        # Sort ascending; multiply by m/(1, 2, ..., m); cummin from right; clip
        order = np.argsort(raw)
        multiplied = np.minimum(raw[order] * m / np.arange(1, m + 1), 1.0)
        adj_sorted = np.minimum.accumulate(multiplied[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = adj_sorted

    result = list(pvalues)
    for i, idx in enumerate(valid_idx):
        result[idx] = float(adj[i])
    return result


def _stat_associations(
    observation_stats: pd.DataFrame,
    y,
    y_type: str,
    correction: str = None,
) -> pd.DataFrame:
    """
    Test each column of observation_stats for association with y.

    Parameters
    ----------
    observation_stats : pd.DataFrame
        Per-observation statistics, one row per sample.
    y : array-like
        Outcome variable (already validated via _validate_y).
    y_type : {'binary', 'continuous'}
        Pre-validated type of y.
    correction : {'bonferroni', 'holm', 'fdr_bh'} or None, default=None
        Multiple testing correction applied across the stat columns.
        When provided, a ``P-value (adjusted)`` column is appended.

    Returns
    -------
    pd.DataFrame
        One row per stat column.  For binary ``y``: columns are
        ``Stat``, ``Test``, ``Mean (y=<val0>)``, ``Mean (y=<val1>)``,
        ``Statistic``, ``P-value``, where ``val0``/``val1`` are the two
        unique y values (sorted).  The group means make the direction of
        the t-statistic immediately legible: a positive t means the stat
        is higher for ``val0``; a negative t means higher for ``val1``.
        For continuous ``y``: columns are ``Stat``, ``Test``,
        ``Statistic`` (Pearson r, already directional), ``P-value``.
        When ``correction`` is provided, a ``P-value (adjusted)`` column
        is appended to the right of ``P-value``.
    """
    y_series = pd.Series(y).reset_index(drop=True)
    results = []

    for col in observation_stats.columns:
        vals = observation_stats[col].reset_index(drop=True)
        mask = y_series.notna() & vals.notna()

        if y_type == 'binary':
            unique_vals = sorted(y_series[mask].unique())
            g0 = vals[mask & (y_series == unique_vals[0])]
            g1 = vals[mask & (y_series == unique_vals[1])]
            t_stat, p_value = ttest_ind(g0, g1, equal_var=False)
            results.append({
                'Stat': col,
                'Test': 'Independent t-test',
                f'Mean (y={unique_vals[0]})': g0.mean(),
                f'Mean (y={unique_vals[1]})': g1.mean(),
                'Statistic': t_stat,
                'P-value': p_value,
            })
        else:  # continuous
            r, p_value = pearsonr(vals[mask], y_series[mask])
            results.append({
                'Stat': col,
                'Test': 'Pearson correlation',
                'Statistic': r,
                'P-value': p_value,
            })

    df = pd.DataFrame(results)
    if correction is not None and not df.empty:
        df['P-value (adjusted)'] = _adjust_pvalues(df['P-value'].tolist(), correction)
    return df


def one_vs_rest(
    dat: pd.DataFrame,
    col_x: Optional[str] = None,
    col_y: Optional[str] = None,
    y_type: Optional[str] = None,
    correction: Optional[str] = None,
) -> pd.DataFrame:
    """
    Perform a one-vs-rest analysis and global association tests on the given dataset.

    This function compares each category in the specified column (`col_x`) against all other categories
    in terms of the response variable (`col_y`). If the response variable is binary, it performs Fisher's
    Exact Test for one-vs-rest comparisons and a Chi-Square Test for global association. If the response
    variable is continuous, it performs an independent t-test for one-vs-rest comparisons and a One-Way
    ANOVA for global association.

    Parameters
    ----------
    dat : pd.DataFrame
        The input DataFrame containing the data.
    col_x : str, optional
        The name of the column containing the categories to compare.
        If None, the first column is used.
    col_y : str, optional
        The name of the column containing the response variable.
        If None, the second column is used.
    y_type : {'binary', 'continuous'} or None, default=None
        Pre-validated type of the response variable. When provided, type detection
        is skipped. If None, type is inferred via ``ClusterExplainer._validate_y()``.
    correction : {'bonferroni', 'holm', 'fdr_bh'} or None, default=None
        Multiple testing correction applied to the K per-cluster p-values.
        When provided, a ``P-value (adjusted)`` column is appended to the
        returned results DataFrame.  The global association p-value is not
        adjusted (it is a single omnibus test).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the one-vs-rest analysis and global tests,
        including the category, test type, statistic, and p-value.

    Raises
    ------
    ValueError
        If the specified columns are not in the DataFrame or if `col_y` has unsupported data types.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'Group': ['A', 'A', 'B', 'B', 'C', 'C'],
    ...     'Outcome': [1, 0, 1, 1, 0, 0]
    ... })
    >>> one_vs_rest(data, col_x='Group', col_y='Outcome')
    """

    # Validate column names
    if col_x is None:
        col_x = dat.columns[0]
    if col_y is None:
        col_y = dat.columns[1]
    if col_x not in dat.columns:
        raise ValueError(f"Column '{col_x}' not found in the DataFrame.")
    if col_y not in dat.columns:
        raise ValueError(f"Column '{col_y}' not found in the DataFrame.")

    results = []
    categories = sorted(dat[col_x].dropna().unique())

    if y_type is None:
        y_type = ClusterExplainer._validate_y(dat[col_y])
    is_binary = (y_type == 'binary')

    for category in categories:
        group_mask = dat[col_x] == category
        if is_binary:
            # Create contingency table for each category vs. rest
            table = pd.crosstab(group_mask, dat[col_y])
            if table.shape != (2, 2):
                # Not enough data to perform Fisher's Exact Test
                odds_ratio, p_value = None, None
            else:
                # Calculate Fisher's Exact Test
                odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
            results.append({
                'Category': category,
                'Size': group_mask.sum(),
                'Test': 'Fisher Exact Test',
                'Odds Ratio': odds_ratio,
                'P-value': p_value
            })
        else:
            # Split the data into the category of interest and the rest
            group1 = dat.loc[group_mask, col_y].dropna()
            group2 = dat.loc[~group_mask, col_y].dropna()
            if len(group1) < 2 or len(group2) < 2:
                t_stat, p_value = None, None
            else:
                # Perform an independent t-test
                t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            results.append({
                'Category': category,
                'Size': len(group1),
                'Test': 'Independent t-test',
                'T-Statistic': t_stat,
                'P-value': p_value
            })

    # Add global association test
    if is_binary:
        # Global association using Chi-Square Test
        contingency_table = pd.crosstab(dat[col_x], dat[col_y])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        global_test = {
            'Test': 'Chi-Square Test',
            'Statistic': chi2,
            'P-value': p_value,
            'Degrees of Freedom': dof
        }
    else:
        # Global association using One-Way ANOVA
        groups = [dat.loc[dat[col_x] == category, col_y].dropna() for category in categories]
        if any(len(group) < 2 for group in groups):  # Not enough data for ANOVA
            global_test = {
                'Test': 'One-Way ANOVA',
                'Statistic': None,
                'P-value': None
            }
        else:
            f_stat, p_value = f_oneway(*groups)
            global_test = {
                'Test': 'One-Way ANOVA',
                'Statistic': f_stat,
                'P-value': p_value
            }

    # Create DataFrame for one-vs-rest results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Category').reset_index(drop=True)

    # Apply multiple testing correction to per-cluster p-values
    if correction is not None and not results_df.empty:
        results_df['P-value (adjusted)'] = _adjust_pvalues(
            results_df['P-value'].tolist(), correction
        )

    # Append global test results to the DataFrame
    global_test_df = pd.DataFrame([global_test])
    return results_df, global_test_df