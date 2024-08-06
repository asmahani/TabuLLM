# What is `TabuLLM`?
A Python package for feature extraction from text columns in tabular data using large language models (LLMs). `TabuLLM` is compatible with the [scikit-learn transformers](https://scikit-learn.org/stable/data_transforms.html) interface, allowing the components to be used as in larger composite pipelines.

# `TabuLLM` Modules
1. `embed` - Unified interface for converting one or more text column(s) in the data to a numeric matrix, using commercial LLMs (OpenAI, Google Vertex AI), open-source LLMs (available on Hugging Face and accessed via the [sentence transformers](https://sbert.net/) package), as well as earlier-generation embedding methods such as [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html).
1. `cluster` - Python implementation of [spherical k-means](https://www.jstatsoft.org/article/view/v050i10) for clustering data points using as features the embedding vectors produced by LLMs. Since embeddings are L2-normalized, i.e., they only contain directional information and their magnitude is not meaningful, it is more appropriate to use spherical k-means, which replaces the Euclidean distance - used in standard k-means - with [cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity). (Note: While the cluster-assignment step in the Lloyd's algorithm for esitmating k-means would be identical using Euclidean vs. cosine distance metrics, the centroid-update step would be different since taking a simple average of L2-normalized vectors does not produce another L2-normalized vector.)
1. `explain` - 1) Prompt generation for soliciting descriptive labels for clusters generating based on the embedding vectors, 2) Wrapper for interacting with text-completion LLMs (currently: OpenAI, Google).
1. `compress` - Applying k-nearest-neighbor - in supervised mode - to predict the outcome using the embedding matrix. Wrapping the KNN in cross-fit allows us to compress the high-dimensional embeddings into a single score - while controlling overfitting - which can subsequently be used alongside other features in the ultimate predictive model.

# `TabuLLM` Tutorials

The following is a list of tutorials provided in the package github repo, along with a list of topics covered in each:

- [Overview + `embed` module](notebooks/tutorial.ipynb):
    - An expanded context/motivation for the package
    - Overview of the `AKI` dataset used throughout the tutorials
    - Introduction of the class `TextColumnTransformer` and how it can be used to generate embedding using various model types, and
    - Options for handling multiple text columns in the data.
- [`cluster` module](notebooks/skmeans.ipynb)
- [`explain` module](notebooks/explain.ipynb)
- [`compress` module](notebooks/compress.ipynb)

# Links
- [*TabuLLM* in Action](https://www.medrxiv.org/content/10.1101/2024.05.14.24307372v1): Applying `TabuLLM` to pediatric cardiopulmonary bypass EHR data to predict the risk of postoperative acute kindey injury.
