# Examples

The [`examples/`](https://github.com/asmahani/TabuLLM/tree/main/examples) folder
in the repository contains Jupyter notebooks demonstrating common TabuLLM
workflows. They are linked here rather than rendered inline: several cells
download the fraud detection dataset from Zenodo, download HuggingFace
embedding models, or make paid LLM calls that require provider credentials,
so they are not executed as part of the documentation build.

## [`01_fraud_detection_walkthrough.ipynb`](https://github.com/asmahani/TabuLLM/blob/main/examples/01_fraud_detection_walkthrough.ipynb)

The core TabuLLM workflow on the fraud detection dataset:

- TF-IDF vs. LLM embeddings
- GMM-based dimensionality reduction with cluster quality diagnostics
- Full `ClusterExplainer` usage: cost preview, outcome-based testing,
  per-observation diagnostics, narrative synthesis
- A predictive pipeline combining text and structured features

**External requirements:** downloads the fraud dataset from Zenodo (no
credentials needed); downloads a HuggingFace embedding model; uses an OpenAI
model for cluster explanations (requires an `OPENAI_API_KEY`).

## [`02_advanced_pipelines.ipynb`](https://github.com/asmahani/TabuLLM/blob/main/examples/02_advanced_pipelines.ipynb)

Advanced pipeline patterns:

- Forward/backward column sweep to measure the marginal contribution of each
  text column
- Stacking ensembles (single-split and multi-split) that process column
  groups independently and combine predictions via a meta-learner

**External requirements:** downloads the fraud dataset from Zenodo; downloads
a HuggingFace embedding model. No LLM/provider credentials are required for
this notebook.

## Running the notebooks locally

```bash
pip install "tabullm[examples]"
jupyter lab examples/
```

See each notebook's own setup cells for the specific credentials or model
downloads it requires.
