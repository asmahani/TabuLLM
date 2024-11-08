from langchain import PromptTemplate
import numpy as np

# Define the main template for the prompt with placeholders
preamble_template_1stage = PromptTemplate(
    input_variables=["n_obs", "prompt_observations", "prompt_texts", "n_cluster"],
    template=(
        "The following is a list of {n_obs} {prompt_observations}. Text lines represent {prompt_texts}. "
        "Data points have been grouped into {n_cluster} groups, according to {prompt_texts}. "
        "Please suggest group labels that are representative of their members and distinct from each other. "
        "For each group return the group number, a short description/group label, "
        "and a long description of less than 50 words."
    )
)

preamble_template_2stage_1 = PromptTemplate(
    input_variables=["prompt_observations", "prompt_texts"],
    template=(
        "The following is a list of {prompt_observations}. Text lines represent {prompt_texts}. "
        "Please summarize the data in less than 200 words."
    )
)
preamble_template_2stage_2 = PromptTemplate(
    input_variables=["prompt_observations", "prompt_texts", "n_cluster"],
    template=(
        "The following are descriptions of {n_cluster} groups of {prompt_observations} based on their {prompt_texts}. "
        "Please suggest group labels that are representative of their members and distinct from each other. "
        "For each group return the group number, a short description/group label, "
        "and a long description of less than 50 words."
    )
)

# Modified function to separate group labels from payloads
def generate_prompt(
    text_list,
    cluster_labels,
    prompt_observations=None,
    prompt_texts=None,
):
    # Validation checks (same as original)
    if not isinstance(text_list, list) or not all(isinstance(item, str) for item in text_list):
        raise TypeError("text_list must be a list of strings.")
    if not isinstance(cluster_labels, np.ndarray):
        raise TypeError("cluster_labels must be a numpy ndarray.")
    if not text_list:
        raise ValueError("text_list cannot be empty.")
    if cluster_labels.size == 0:
        raise ValueError("cluster_labels cannot be empty.")
    n_obs = len(text_list)
    if len(cluster_labels) != n_obs:
        raise ValueError("Number of text strings and cluster labels must be the same.")
    if (not prompt_observations or not prompt_texts):
        raise ValueError("'prompt_observations' and 'prompt_texts' must be provided.")

    my_clusters = np.unique(cluster_labels)
    n_cluster = len(my_clusters)
    if not np.array_equal(my_clusters, np.arange(0, n_cluster)):
        raise ValueError("Cluster labels must be integers 0-N, with N >= 1.")

    # Generate preamble using the prompt template
    preamble_1stage = preamble_template_1stage.format(
        n_obs=n_obs,
        prompt_observations=prompt_observations,
        prompt_texts=prompt_texts,
        n_cluster=n_cluster
    )
    preamble_2stage_1 = preamble_template_2stage_1.format(
        prompt_observations=prompt_observations,
        prompt_texts=prompt_texts
    )
    preamble_2stage_2 = preamble_template_2stage_2.format(
        prompt_observations=prompt_observations,
        prompt_texts=prompt_texts,
        n_cluster=n_cluster
    )

    # Generate individual payloads for each cluster without prepended labels
    group_payloads = []
    for n in range(n_cluster):
        text_entries = '\n'.join([text_list[i] for i in range(n_obs) if cluster_labels[i] == n])
        group_payloads.append(text_entries)

    # Return the preamble and list of group payloads as (label, content) tuples
    return (preamble_1stage, (preamble_2stage_1, preamble_2stage_2), group_payloads)
