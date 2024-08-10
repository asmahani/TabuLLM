import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_ind
import vertexai
from vertexai.generative_models import GenerativeModel
from pydantic import BaseModel

class GroupLabel(BaseModel):
    number: int
    description_short: str
    description_long: str

class MultipleGroupLabels(BaseModel):
    groups: list[GroupLabel]

    # method to convert the response to a DataFrame
    def to_df(self):
        return pd.DataFrame([group.model_dump() for group in self.groups]).sort_values('number').reset_index(drop=True)


def generate_prompt(
    text_list
    , cluster_labels
    , prompt_observations = None
    , prompt_texts = None
    , preamble = ''
):
    """Assembling a prompt to solicit cluster descriptions from a text-completion large language model.
    The returned prompt will consist of two parts: 1- preamble, which provides the context and instructions to
    the text-completion model, and 2- the list of observations, grouped by their clusters, where
    each observation represented by the value of their text field.

    :param text_list: List of text strings associated with a collection of observations.
    :type text_list: list
    :param cluster_labels: Numpy array of cluster memberships associated the same collection of observations.
    :type cluster_labels: numpy.ndarray
    :param prompt_observations: Name/phrase that the observation units should be referred to in the prompt, must be in plural form.
    Used to generate the prompt preamble, and will be ignored if `preamble` is any string other than the empty string.
    :type prompt_observations: str
    :param prompt_texts: What does the text field represent for each observation unit? Must be in plural form.
    Used to generate the prompt preamble, and will be ignored if `preamble` is any string other than the empty string.
    :type prompt_texts: str
    :param preamble: Prompt preamble which provides the context and requested task to the text-completion model.
    If an empty string is provided - which is the default value - preamble will be automatically constructed.
    :type preamble: str
    :return: Full text of the prompt.
    :rtype: str
    """
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
    if preamble == '' and (not prompt_observations or not prompt_texts):
        raise ValueError("'prompt_observations' and 'prompt_texts' must be provided if 'preamble' is to be generated automatically.")

    my_clusters = np.unique(cluster_labels)
    n_cluster = len(my_clusters)
    check_cluster_labels = np.array_equal(
        my_clusters
        , np.arange(0, n_cluster + 0)
    )
    if not check_cluster_labels:
        raise ValueError("Cluster labels must be integers 0-N, with N >= 1.")
    
    if preamble == '':
        preamble = (f"The following is a list of {str(n_obs)} {prompt_observations}. Text lines represent {prompt_texts}."
                  f" {prompt_observations.capitalize()} have been grouped into {str(n_cluster)} groups, according to their {prompt_texts}."
                  " Please suggest group labels that are representative of their members, and also distinct from each other."
                  " Follow the provided template to return - for each group - the group number, a short desciption / group label, and a long description."
                 )

    my_body_list = []
    for n in range(n_cluster):
        sublist = [text_list[i] for i in range(n_obs) if cluster_labels[i] == n]
        substring = '\n'.join(sublist)
        substring = 'Group ' + str(n + 1) + ':\n\n' + substring
        my_body_list.append(substring)

    my_body_string = '\n\n=====\n\n'.join(my_body_list)

    #my_full_prompt = preamble + '\n\n=====\n\n' + my_body_string
    
    return (preamble, my_body_string)


def generate_response(
    prompt_instructions
    , prompt_body
    , model_type = 'openai'
    , openai_client = None
    , google_project_id = None
    , google_location = None
    , openai_model = 'gpt-4-turbo'
    , google_model = 'gemini-1.5-pro-001'
):
    """
    Thin wrapper around OpenAI/Google text completion API call, to be used for submitting
    clustered text fields for cluster labeling.
    
    :param prompt: The prompt text to send to the model.
    :type prompt: str
    :param model_type: The type of text-completion model to use ('openai' or 'google').
    :type type: str, optional
    :param openai_client: An instance of the OpenAI API client.
    :type openai_client: OpenAI.Client, optional
    :param google_project_id: The Google Cloud project ID.
    :type google_project_id: str, optional
    :param google_location: The Google Cloud location.
    :type google_location: str, optional
    :param openai_model: The identifier for the OpenAI text generation model to use.
    :type openai_model: str, optional
    :param google_model: The identifier for the Google text generation model to use.
    :type google_model: str, optional
    :return: The text completion generated by the model.
    :rtype: str
    
    :raises ValueError: If the prompt is empty or not a string, or if the type is not 'openai' or 'google'.
    :raises RuntimeError: If there is an issue with the API call.
    """
    
    if model_type not in ['openai', 'google']:
        raise ValueError("Type must be either 'openai' or 'google'.")
    
    if model_type == 'openai':
        try:
            response = openai_client.beta.chat.completions.parse(
                model = openai_model
                , messages = [
                    {"role": "system", "content": prompt_instructions},
                    {"role": "user", "content": prompt_body}
                ]
            )
            return response.choices[0].message.parsed.to_df()
        except Exception as e:
            raise RuntimeError(f"Failed to generate completion from OpenAI API: {e}") from e
    else:
        try:
            vertexai.init(project=google_project_id, location=google_location)
            model = GenerativeModel(google_model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Failed to generate completion from Google API: {e}") from e
    


def one_vs_rest(dat, col_x=None, col_y=None):
    """
    Perform a one-vs-rest analysis on the given dataset.

    This function compares each category in the specified column (col_x) against all other categories
    in terms of the response variable (col_y). If the response variable is binary, it performs Fisher's
    Exact Test. If the response variable is continuous, it performs an independent t-test.

    :param dat: The input DataFrame containing the data.
    :type dat: pd.DataFrame
    :param col_x: The name of the column containing the categories to compare. If None, the first column is used.
    :type col_x: str, optional
    :param col_y: The name of the column containing the response variable. If None, the second column is used.
    :type col_y: str, optional
    :return: A DataFrame containing the results of the one-vs-rest analysis, including the category, test type, statistic, and p-value.
    :rtype: pd.DataFrame
    """
    if not col_x:
        col_x = dat.columns[0]
    if not col_y:
        col_y = dat.columns[1]
    
    results = []
    categories = sorted(dat[col_x].unique())
    
    # Determine if the response variable is binary or continuous
    is_binary = dat[col_y].nunique() == 2
    
    for category in categories:
        if is_binary:
            # Create contingency table for each category vs. rest
            table = pd.crosstab(dat[col_x] == category, dat[col_y])
            # Calculate Fisher's Exact Test
            odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
            results.append((category, 'Odds Ratio', odds_ratio, p_value))
        else:
            # Split the data into the category of interest and the rest
            group1 = dat[dat[col_x] == category][col_y]
            group2 = dat[dat[col_x] != category][col_y]
            # Perform an independent t-test
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            results.append((category, 'T-Statistic', t_stat, p_value))
    
    results_df = pd.DataFrame(
        results
        , columns=['Category', 'Test Type', 'Statistic', 'P-value']
    ).sort_values('Category').set_index('Category')
    return results_df
