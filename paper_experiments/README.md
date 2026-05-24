# paper_experiments

Reproduction package for the fraud-detection experiments in the SoftwareX revision.

## What is in this folder

```text
paper_experiments/
├── README.md
├── .env.example
├── requirements-paper.txt
├── download_embeddings.py
├── download_results.py
├── generate_embeddings.py
├── summarise_results.py
├── k_sweep/
│   ├── run_k_sweep.py
│   └── plot_k_sweep.py
├── col_sweep/
│   ├── run_col_sweep.py
│   └── plot_col_sweep.py
├── embeddings/            # .gitignored; downloaded or locally generated
└── results/
    ├── t1_preds/          # raw prediction files (+ run-script sidecar metrics/configs)
    ├── t2_metrics/        # per-fold metrics from summarise_results.py
    └── t3_rolled/         # rolled-up mean ± SE summaries used by the plot scripts
```

## Setup

### Plotting and summarisation only

This is enough if you only want to regenerate summaries or figures from the committed result files:

```bash
pip install tabullm matplotlib
```

### Full reruns of the paper experiments

Use the pinned environment that produced the manuscript results:

```bash
pip install -r requirements-paper.txt
```

`requirements-paper.txt` is a `pip freeze` from the Linux/amd64 Python 3.13 environment used for the revision runs.

### Credentials

Copy `.env.example` to `.env` only if you need credentials:

- `OPENAI_API_KEY` is required for `openai-small` and `openai-large`
- `HF_MODEL_CACHE_DIR` is optional, and only needed when HuggingFace Hub is not directly reachable

## Reproduction paths

### 1. Fastest: regenerate figures from committed T3 summaries

This path uses the summary CSVs already present in `results/t3_rolled/`.

```bash
python k_sweep/plot_k_sweep.py
python col_sweep/plot_col_sweep.py --rep minilm
```

### 2. Rebuild T2/T3 summaries from committed T1 prediction files

This path recomputes the derived summaries from the committed `results/t1_preds/*_preds.csv` files.

```bash
python summarise_results.py
python k_sweep/plot_k_sweep.py
python col_sweep/plot_col_sweep.py --rep minilm
```

`python summarise_results.py` with no arguments batch-processes all `*_preds.csv` files in `results/t1_preds/`.

### 3. Restore results from Zenodo

Use this if you do not have the local `results/` tree or want to refresh it from the archived bundle:

```bash
python download_results.py
python k_sweep/plot_k_sweep.py
python col_sweep/plot_col_sweep.py --rep minilm
```

By default, `download_results.py` resolves the latest Zenodo record under concept DOI `10.5281/zenodo.18884001`, downloads `results.zip`, and extracts it into this folder.

### 4. Re-run experiments from scratch

#### Minimal rerun for the two paper figures

```bash
python download_embeddings.py --model minilm
python k_sweep/run_k_sweep.py --rep minilm
python col_sweep/run_col_sweep.py --rep minilm
python summarise_results.py
python k_sweep/plot_k_sweep.py
python col_sweep/plot_col_sweep.py --rep minilm
```

Instead of downloading embeddings, you can generate them locally:

```bash
python generate_embeddings.py --model minilm --sweeps
```

#### Full paper run matrix

The manuscript results were built from the following runs:

```bash
python k_sweep/run_k_sweep.py --rep none
python k_sweep/run_k_sweep.py --rep minilm
python k_sweep/run_k_sweep.py --rep bge-large
python k_sweep/run_k_sweep.py --rep openai-small
python k_sweep/run_k_sweep.py --rep openai-large
python k_sweep/run_k_sweep.py --rep tfidf
python k_sweep/run_k_sweep.py --rep tfidf --svd-dim 1024
python k_sweep/run_k_sweep.py --rep tfidf --svd-dim 1536
python k_sweep/run_k_sweep.py --rep tfidf --svd-dim 3072
python col_sweep/run_col_sweep.py --rep minilm
python summarise_results.py
python k_sweep/plot_k_sweep.py
python col_sweep/plot_col_sweep.py --rep minilm
```

## Script roles

- `generate_embeddings.py`: create cached `.npy` embedding files, including forward/backward sweep caches when `--sweeps` is used
- `download_embeddings.py`: download pre-computed fraud embeddings from Zenodo record `20290842`
- `k_sweep/run_k_sweep.py`: generate T1 prediction files for the k-sweep benchmark
- `col_sweep/run_col_sweep.py`: generate T1 prediction files for the MiniLM column-sweep benchmark
- `summarise_results.py`: convert T1 prediction files into T2 per-fold metrics and T3 rolled-up summaries
- `plot_k_sweep.py`, `plot_col_sweep.py`: generate the manuscript figures from T3 summaries
- `download_results.py`: restore archived result bundles when local `results/` outputs are absent

## Reproducibility notes

### Environment used for the paper runs

The sweep experiments were run on x86_64 Linux using the exact versions in `requirements-paper.txt`. Key versions:

- `tabullm==1.3.0`
- `scikit-learn==1.8.0`
- `numpy==2.4.6`
- `langchain-core==1.4.0`

### Embedding sources

HuggingFace embeddings were generated with pinned model revisions:

| Model | HuggingFace ID | Commit hash |
|---|---|---|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | `fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9` |
| BGE-Large | `BAAI/bge-large-en-v1.5` | `d4aa6901d3a41ba39fb536a557fa166f842b0e09` |

Running `generate_embeddings.py` with those revisions reproduces the same local HuggingFace embeddings.

OpenAI embeddings were generated with `text-embedding-3-small` and `text-embedding-3-large` in May 2026. Because OpenAI may update hosted models silently, re-calling the API is not guaranteed to reproduce identical vectors; for exact reproduction, prefer the archived embedding files from Zenodo.

### Convergence warnings

The k-sweep and col-sweep scripts may hit `ConvergenceWarning` from scikit-learn's `GaussianMixture` at higher values of `k` (for example `k >= 50`). Both run scripts suppress these warnings in the main process and in joblib subprocesses. Increasing `max_iter` or `n_init` would likely reduce the warnings, but was not done here because it would require re-running the entire sweep matrix at substantial additional cost.

### Computational requirements

The most expensive run is:

```bash
python k_sweep/run_k_sweep.py --rep openai-large
```

In the paper environment, each fold required up to about 10 GB RAM and more than 6 CPU-hours; all 100 folds were run in parallel on a high-memory server (~1 TB RAM), giving a wall-clock time of roughly 6-7 hours for that run. Users with fewer cores or less memory should reduce `--n-jobs` and expect proportionally longer wall times.

The MiniLM, BGE-Large, OpenAI-small, TF-IDF, and MiniLM col-sweep runs are all substantially cheaper.

## Zenodo assets

| Asset | Source |
|---|---|
| Pre-computed embeddings | `download_embeddings.py` -> Zenodo record `20290842` |
| Results bundle | `download_results.py` -> latest record under concept DOI `10.5281/zenodo.18884001` |
