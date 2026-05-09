# Changelog

All notable changes to TabuLLM are documented here.

## [1.3.0]

- Tightened the supported runtime window for the LangChain stack to the tested 1.1.x-1.2.x series, with matching caps on `numpy`, `scipy`, `scikit-learn`, and `pydantic`.
- Expanded CI to run the test suite across multiple LangChain minor lines and updated tests to use `langchain_core.embeddings.Embeddings` directly instead of the legacy compatibility shim.
- Added weekly Dependabot checks for Python dependencies and GitHub Actions so upstream changes surface as reviewable PRs before they reach users.

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
