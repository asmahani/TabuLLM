# TabuLLM

Python package for feature extraction and interpretation of text columns in tabular data using large language models.

## Overview

TabuLLM integrates LLM-based text embeddings into scikit-learn pipelines for tabular datasets containing text columns. Built on LangChain and scikit-learn, it provides sklearn-compatible transformers for embedding, dimensionality reduction, and cluster interpretation.

## Installation

```bash
pip install tabullm
```


## Core Components

**TextColumnTransformer** - Wraps LangChain embedding models (OpenAI, Anthropic, HuggingFace, etc.) with a sklearn interface. Handles multiple text columns with configurable concatenation and optional L2 normalization (`normalize=True`). Use `estimate_tokens()` to preview API cost before embedding.

**GMMFeatureExtractor** - Extends sklearn's `GaussianMixture` with a `transform()` method that returns per-cluster log-joint features $\log p(\mathbf{x}, c_k)$ — the quantity the GMM maximises for hard assignment — enabling use in sklearn pipelines. An optional `include_log_density` parameter appends the marginal log-density as an explicit outlier score. A companion `assignment_confidence_stats()` method returns per-observation cluster quality diagnostics (`max_posterior`, `entropy`, `log_joint_margin`, `log_density`).

**SphericalKMeans** - K-means clustering with cosine distance for L2-normalized embeddings. For normalized embeddings, mathematically equivalent to sklearn's `KMeans`. Available as an alternative hard-clustering option when GMM-based features are not needed.

**ClusterExplainer** - Generates natural language cluster descriptions using LLMs with automatic recursive summarization that scales to arbitrarily large datasets. Supports:
- Cost preview (`preview=True`) before LLM calls
- Optional outcome-based statistical testing (`y`) to characterize which clusters associate with a target variable
- Per-observation covariates (`observation_stats`) — e.g., from `assignment_confidence_stats()` — appended to the association table
- A synthesis step (`synthesize=True`) that produces a coherent interpretive narrative across all cluster results
- An outcome label (`y_label`) used only in the synthesis prompt; cluster descriptions are generated without knowledge of `y` (*blind labeling* principle)

**load_fraud()** - Data utility that downloads and caches the fraud detection dataset from Zenodo (no credentials required), returning features, labels, and metadata.

## Quick Example

```python
from tabullm import TextColumnTransformer, GMMFeatureExtractor, ClusterExplainer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Embed text columns
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
text_transformer = TextColumnTransformer(
    model=embedding_model,
    colnames={'title': 'Title', 'description': 'Description'}
)

# Build pipeline: Embed → Reduce → Classify
pipeline = Pipeline([
    ('embed', text_transformer),
    ('reduce', GMMFeatureExtractor(n_components=10)),
    ('classify', RandomForestClassifier(n_estimators=100))
])

# Fit and predict
pipeline.fit(df[['title', 'description']], y)
predictions = pipeline.predict(df_new[['title', 'description']])

# Interpret clusters
explainer = ClusterExplainer(
    llm=ChatOpenAI(model='gpt-4o-mini'),
    text_transformer=text_transformer,
    observations='job postings',
    text_fields='titles and descriptions'
)

gmm = pipeline.named_steps['reduce']
cluster_labels = gmm.labels_

# Cluster descriptions only
result_df = explainer.explain(df, cluster_labels)

# With outcome association + synthesis narrative
result_df, global_stats, synthesis = explainer.explain(
    df, cluster_labels,
    y=y,
    y_label='fraudulent posting (1=fraud, 0=legitimate)',
    synthesize=True
)

# Include GMM cluster quality diagnostics in the association table
obs_stats = gmm.assignment_confidence_stats(
    pipeline.named_steps['embed'].transform(df)
)
result_df, global_stats, stat_assoc_df, synthesis = explainer.explain(
    df, cluster_labels,
    y=y,
    y_label='fraudulent posting (1=fraud, 0=legitimate)',
    observation_stats=obs_stats,
    synthesize=True
)
```

## Key Features

- sklearn-compatible API (Pipeline, ColumnTransformer, GridSearchCV)
- Access to 50+ embedding models via LangChain
- Multi-column text handling with flexible concatenation
- Optional L2 normalization of embedding vectors
- Token and cost estimation before embedding API calls
- GMM-based dimensionality reduction with per-cluster log-joint features
- Optional marginal log-density feature for explicit outlier scoring
- Per-observation cluster quality diagnostics (max posterior, entropy, log-joint margin, log density)
- Automatic recursive summarization for arbitrarily large datasets
- Cost estimation for LLM explanation calls
- Outcome-based cluster characterization (binary and continuous outcomes)
- User-supplied per-observation covariates in the association table
- Synthesis narrative connecting cluster descriptions to outcome patterns
- Blind labeling: cluster descriptions generated without knowledge of outcome vector

## Release Notes

**1.1.0** — Added multiple testing correction to `explain()` via the `correction` parameter (`'bonferroni'`, `'holm'`, `'fdr_bh'`). When set, a `P-value (adjusted)` column is appended to the per-cluster results and, when `observation_stats` is provided, to the stat-association table. Backward compatible: default is `None` (no correction).

**1.0.3** — Fixed broken package installation (1.0.2 wheel was published without Python source files).

**1.0.2** — Fixed `__version__` mismatch; aligned `__init__.py` with `pyproject.toml`.

**1.0.1** — Switched fraud dataset download from Kaggle to Zenodo (no credentials required).

**1.0.0** — Initial release.

## Citation

Sharabiani, M.T.A., Mahani, A.S., Bottle, A. et al. (2025). GenAI exceeds clinical experts in predicting acute kidney injury following paediatric cardiopulmonary bypass. Scientific Reports, 15, 20847. https://doi.org/10.1038/s41598-025-04651-8
