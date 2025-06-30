import llm_query, LoRD3
import numpy as np
import pandas as pd
from inference_classifier import run_inference_algorithm
from discovery_classifier import run_discovery_algorithm

# Variables the user has to provide
data = pd.read_csv('compulsory_school.csv')
question = "How does the level of education affect earnings?"
assignment_style = 'observational' # 'observational', 'randomized'
latent_confounders = False  # Set to True if latent confounders are present

# Variables automated by LLM
context = data.head(3).to_string(index=False)
treatment = llm_query.create_chat_completion(
    messages = [
        {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **treatment variable** — the variable being manipulated or used as the cause in a causal question. Be sure to choose the **single most appropriate column name** that best represents the treatment, and your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words. For example, if the question asks how does rain impact temperature, your response should be the column name most closely related to rain."},
        {"role": "user", "content": question}
    ], temperature = 0.1, thinking = False
)
outcome = llm_query.create_chat_completion(
    messages = [
        {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **outcome variable** — the variable that is affected or influenced by the treatment in a question. Be sure to choose the **single most appropriate column name** that best represents the outcome, and your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words. For example, if the question asks how does rain impact temperature, your response should be the column name most closely related to temperature."},
        {"role": "user", "content": question}
    ], temperature = 0.1, thinking = False
)
time_variable = llm_query.create_chat_completion(
    messages=[
        {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **time variable** — the (often binary) variable used for DiD that represents whether the observation is from before or after the implementation of the policy. Be sure to reply **None** if no variable seems appropriate, or choose the **single most appropriate column name** that best represents the outcome. Your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words."},
        {"role": "user", "content": "What is the time variable in the data? Reply None if there is no time variable."}
    ], temperature = 0.1, thinking = False
)
group_variable = llm_query.create_chat_completion(
    messages=[
        {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **group variable** — the (often binary) variable used for DiD that represents whether the observation is part of the control or treatment group. Be sure to reply **None** if no variable seems appropriate, or choose the **single most appropriate column name** that best represents the outcome. Your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words."},
        {"role": "user", "content": "What is the group variable in the data? Reply None if there is no group variable."}
    ], temperature = 0.1, thinking = False
)
print("treatment:", treatment)
print("outcome:", outcome)
print("time variable:", time_variable)
print("group variable:", group_variable)

# Data preprocessing (Drop non-numeric columns and missing values, and columns with high correlation)
mapping = {
    True:  1,
    False: 0,
    'True':  1,
    'False': 0,
    'Yes': 1,
    'No':  0
}
data=data.replace(mapping, inplace=True)
numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.dropna()
corr = numeric_data.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.999)]
numeric_data = numeric_data.drop(columns=to_drop)

result = LoRD3.detect(
    numeric_data,
    treatment_col=treatment,
    covariate_cols=[],
    poly_degree=2,
    k=50
)
running_variable = str(result.variable)
cutoff_value = str(result.cutoff)
print(running_variable)
print(cutoff_value)

nx_graph = run_discovery_algorithm(numeric_data, latent_confounders)
sample_size = data.shape[0]
covariates = list(set(nx_graph.nodes) - {treatment, outcome})

result = run_inference_algorithm(
    data,
    treatment,
    outcome,
    covariates,
    nx_graph,
    sample_size,
    assignment_style,
    latent_confounders,
    cutoff_value,
    time_variable,
    group_variable,
    running_variable
)

print(result)