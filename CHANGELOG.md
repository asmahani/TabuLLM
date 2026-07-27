# Changelog

All notable changes to TabuLLM are documented here.

## [1.4.0]

- Added a Sphinx documentation site (Furo theme, MyST Markdown, autodoc/autosummary with Napoleon), published to GitHub Pages at https://asmahani.github.io/TabuLLM/. Covers the README as a landing page, full API reference for `tabullm.__all__`, an examples page linking the two tutorial notebooks with their external requirements, and a paper/reproducibility page. A new `docs` optional dependency group and a `docs.yml` CI workflow build the site (warnings as errors) on every PR/push and deploy it on pushes to `main`.
- Fixed several stale and incorrect examples in public docstrings, found while building the documentation: `ClusterExplainer`'s class docstring called a nonexistent `GMMFeatureExtractor.fit_predict()` and the obsolete `count_tokens_only=True` argument (now `preview=True`); `GMMFeatureExtractor`'s example passed an unsupported `text_columns=` argument to `TextColumnTransformer`; several examples referenced undefined names (`model`, `RandomForestClassifier`, `df`, `emb_model`, `chat_model`) without importing or defining them. Examples are now self-contained and runnable (credentialed calls are noted in comments).
- Added the `Documentation` link to `[project.urls]`, a documentation badge to the README, and set the GitHub repository homepage to the published docs site.
- Widened the supported `langchain-core` range to `>=1.3,<1.6` (from `<1.5`); CI now tests both the minimum supported line and the latest allowed release explicitly.
- Added the SoftwareX fraud-benchmark reproduction package (`paper_experiments/`): pinned environment, run/summarise/plot scripts, and archived per-fold results, metrics, and rolled-up summaries. Added the published SoftwareX paper citation and DOI to the README.
- Excluded `paper_experiments/paper-requirements.txt` (a pinned snapshot of the environment used for the paper's benchmark runs) from Dependabot, since it is not meant to track upstream releases.

## [1.3.0]

- Added README badges for CI, coverage, PyPI version, supported Python versions, and license, and enabled Codecov uploads from the test workflow.
- Reduced the core runtime dependency surface to `langchain-core`; direct `langchain` and `langchain-community` requirements were dropped because TabuLLM uses core abstractions plus provider-specific integration packages.
- Tightened the supported runtime window for the LangChain stack to the tested `langchain-core` 1.3.x-1.4.x series, with matching caps on `numpy`, `scipy`, `scikit-learn`, and `pydantic`.
- Expanded CI to run the test suite across multiple LangChain minor lines and updated tests to use `langchain_core.embeddings.Embeddings` directly instead of the legacy compatibility shim.
- Added weekly Dependabot checks for Python dependencies and GitHub Actions so upstream changes surface as reviewable PRs before they reach users.
- **Breaking:** the minimum supported `numpy` version is now 1.26, the minimum supported `pydantic` version is now 2.7, and the package no longer installs the umbrella `langchain` / `langchain-community` packages as transitive runtime dependencies.

## [1.2.1]

- Fixed deprecated LangChain import in `embed.py` (`langchain.embeddings.base` → `langchain_core.embeddings`).
- Removed phantom `pillow` runtime dependency.

## [1.2.0]

- Removed `SphericalKMeans` class. For L2-normalized embeddings, sklearn's `KMeans` is mathematically equivalent; `GMMFeatureExtractor` provides strictly richer features for pipeline use.

## [1.1.0]

- Added multiple testing correction to `explain()` via the `correction` parameter (`'bonferroni'`, `'holm'`, `'fdr_bh'`). When set, a `P-value (adjusted)` column is appended to the per-cluster results and, when `observation_stats` is provided, to the stat-association table. Backward compatible: default is `None` (no correction).

## [1.0.3]

- Fixed broken package installation (1.0.2 wheel was published without Python source files).

## [1.0.2]

- Fixed `__version__` mismatch; aligned `__init__.py` with `pyproject.toml`.

## [1.0.1]

- Switched fraud dataset download from Kaggle to Zenodo (no credentials required).

## [1.0.0]

- Initial release.
