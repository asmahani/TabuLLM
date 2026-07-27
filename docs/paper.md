# Paper and reproducibility

## SoftwareX paper

TabuLLM is described in:

> Sharabiani, M.T.A., Bottle, A., and Mahani, A.S. (2026). *TabuLLM: Feature
> extraction and interpretation of text columns in tabular data using large
> language models*. **SoftwareX, 35**, 102810.
> <https://doi.org/10.1016/j.softx.2026.102810>

## Reproducibility assets

The [`paper_experiments/`](https://github.com/asmahani/TabuLLM/tree/main/paper_experiments)
folder in the repository contains the full reproduction package for the
fraud-detection benchmark experiments reported in the paper:

- [reproduction instructions](https://github.com/asmahani/TabuLLM/blob/main/paper_experiments/README.md),
  covering everything from regenerating figures from committed summaries to
  a full from-scratch rerun of the experiment matrix
- [`paper-requirements.txt`](https://github.com/asmahani/TabuLLM/blob/main/paper_experiments/paper-requirements.txt),
  a pinned `pip freeze` of the exact environment used for the manuscript runs
- run scripts for the two benchmark sweeps
  ([`k_sweep/`](https://github.com/asmahani/TabuLLM/tree/main/paper_experiments/k_sweep),
  [`col_sweep/`](https://github.com/asmahani/TabuLLM/tree/main/paper_experiments/col_sweep))
  and the summarisation/plotting scripts that turn their output into the
  manuscript figures
- [archived results](https://github.com/asmahani/TabuLLM/tree/main/paper_experiments/results)
  (per-fold predictions, metrics, and rolled-up summaries) committed to the
  repository, plus pre-computed embeddings and result bundles archived on
  Zenodo (see the reproduction instructions for record details)

The documentation build does not execute the paper experiments; see the
reproduction instructions above to rerun them.

## Citation

If you use TabuLLM in academic work, please cite:

> Sharabiani, M.T.A., Bottle, A., and Mahani, A.S. (2026). *TabuLLM: Feature
> extraction and interpretation of text columns in tabular data using large
> language models*. **SoftwareX, 35**, 102810.
> <https://doi.org/10.1016/j.softx.2026.102810>

## Related work

> Sharabiani, M.T.A., Mahani, A.S., Bottle, A. et al. (2025). *GenAI exceeds
> clinical experts in predicting acute kidney injury following paediatric
> cardiopulmonary bypass*. **Scientific Reports, 15**, 20847.
> <https://doi.org/10.1038/s41598-025-04651-8>
