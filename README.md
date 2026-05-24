# TabuLLM

[![Tests](https://github.com/asmahani/TabuLLM/actions/workflows/tests.yml/badge.svg)](https://github.com/asmahani/TabuLLM/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/asmahani/TabuLLM/branch/main/graph/badge.svg)](https://codecov.io/gh/asmahani/TabuLLM)
[![PyPI version](https://img.shields.io/pypi/v/tabullm.svg)](https://pypi.org/project/tabullm/)
[![Python versions](https://img.shields.io/pypi/pyversions/tabullm.svg)](https://pypi.org/project/tabullm/)
[![License](https://img.shields.io/github/license/asmahani/TabuLLM.svg)](https://github.com/asmahani/TabuLLM/blob/main/LICENSE)

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

**ClusterExplainer** - Generates natural language cluster descriptions using LLMs with automatic recursive summarization that scales to arbitrarily large datasets. Supports:
- Cost preview (`preview=True`) before LLM calls
- Optional outcome-based statistical testing (`y`) to characterize which clusters associate with a target variable
- Per-observation covariates (`observation_stats`) — e.g., from `assignment_confidence_stats()` — appended to the association table
- A synthesis step (`synthesize=True`) that produces a coherent interpretive narrative across all cluster results
- An outcome label (`y_label`) used only in the synthesis prompt; cluster descriptions are generated without knowledge of `y` (*blind labeling* principle)

**load_fraud()** - Data utility that downloads and caches the fraud detection dataset from Zenodo (no credentials required), returning features, labels, and metadata.

## Quick Example

```python
from tabullm import load_fraud, TextColumnTransformer, GMMFeatureExtractor, ClusterExplainer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
X, y, metadata = load_fraud()
text_cols = ['title', 'location', 'department', 'company_profile',
             'description', 'requirements', 'benefits']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Embed text columns
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
text_transformer = TextColumnTransformer(model=embedding_model)

# Build pipeline: Embed → Reduce → Classify
pipeline = Pipeline([
    ('embed', text_transformer),
    ('reduce', GMMFeatureExtractor(n_components=10, random_state=42)),
    ('classify', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit and predict
pipeline.fit(X_train[text_cols], y_train)
y_pred = pipeline.predict_proba(X_test[text_cols])[:, 1]

# Interpret clusters
explainer = ClusterExplainer(
    llm=ChatOpenAI(model='gpt-4o-mini'),
    text_transformer=text_transformer,
    observations='job postings',
    text_fields='title, location, department, company profile, '
               'description, requirements, and benefits'
)

gmm = pipeline.named_steps['reduce']
cluster_labels = gmm.labels_

# Cluster descriptions only
result_df = explainer.explain(X_train[text_cols], cluster_labels)

# With outcome association + synthesis narrative
result_df, global_stats, synthesis = explainer.explain(
    X_train[text_cols], cluster_labels,
    y=y_train,
    y_label='fraudulent posting (1=fraud, 0=legitimate)',
    synthesize=True
)

# Include GMM cluster quality diagnostics in the association table
obs_stats = gmm.assignment_confidence_stats(
    pipeline.named_steps['embed'].transform(X_train[text_cols])
)
result_df, global_stats, stat_assoc_df, synthesis = explainer.explain(
    X_train[text_cols], cluster_labels,
    y=y_train,
    y_label='fraudulent posting (1=fraud, 0=legitimate)',
    observation_stats=obs_stats,
    synthesize=True
)
```

## Examples

The [`examples/`](examples/) folder contains Jupyter notebooks demonstrating common workflows:

- [`01_fraud_detection_walkthrough.ipynb`](examples/01_fraud_detection_walkthrough.ipynb) — core TabuLLM workflow on the fraud detection dataset: TF-IDF vs. LLM embeddings, GMM-based dimensionality reduction with cluster quality diagnostics, full `ClusterExplainer` usage (cost preview, outcome-based testing, per-observation diagnostics, narrative synthesis), and a predictive pipeline combining text and structured features
- [`02_advanced_pipelines.ipynb`](examples/02_advanced_pipelines.ipynb) — advanced pipeline patterns: forward/backward column sweep to measure marginal contribution of each text column, and stacking ensembles (single-split and multi-split) that process column groups independently and combine predictions via a meta-learner

## Reproducibility assets

The [`paper_experiments/`](paper_experiments/) folder contains the scripts, pinned environment, archived summaries, and reproduction instructions used for the SoftwareX fraud-benchmark experiments.

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

See [CHANGELOG.md](CHANGELOG.md).

## Citation

Sharabiani, M.T.A., Mahani, A.S., Bottle, A. et al. (2025). GenAI exceeds clinical experts in predicting acute kidney injury following paediatric cardiopulmonary bypass. Scientific Reports, 15, 20847. https://doi.org/10.1038/s41598-025-04651-8
