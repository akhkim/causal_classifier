import numpy as np
import pandas as pd
import networkx as nx
from llama_cpp import Llama
import statsmodels.formula.api as smf
from causallearn.utils.cit import fisherz
from causallearn.graph.Endpoint import Endpoint
from classifier import recommend_inference_algorithm
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from inference_algorithms import g_computation, propensity_score, double_machine_learning, iv, rdd, did, ols, frontdoor_adjustment

data = pd.read_csv('ajrcomment.csv')
context = data.head(3).to_string(index=False)
llm = Llama(model_path="Llama-3.1-8B-Instruct-BF16.gguf", verbose=False, n_ctx=2048, chat_format="llama-3")

# Variables the user has to provide
question = "How does institutional risk impact the GDP?"
assignment_style = 'observational' # 'observational', 'randomized'
latent_confounders = False  # Set to True if latent confounders are present

treatment = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **treatment variable** — the variable being manipulated or used as the cause in a causal question. Be sure to choose the **single most appropriate column name** that best represents the treatment, and your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words. For example, if the question asks how does rain impact temperature, your response should be the column name most closely related to rain."},
        {"role": "user", "content": question}
    ]
)['choices'][0]['message']['content']
outcome = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **outcome variable** — the variable that is affected or influenced by the treatment in a question. Be sure to choose the **single most appropriate column name** that best represents the outcome, and your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words. For example, if the question asks how does rain impact temperature, your response should be the column name most closely related to temperature."},
        {"role": "user", "content": question}
    ]
)['choices'][0]['message']['content']
time_variable = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **time variable** — the (often binary) variable used for DiD that represents whether the observation is from before or after the implementation of the policy. Be sure to reply **None** if no variable seems appropriate, or choose the **single most appropriate column name** that best represents the outcome. Your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words."},
        {"role": "user", "content": "What is the time variable in the data? Reply None if there is no time variable."}
    ]
)['choices'][0]['message']['content']
group_variable = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **group variable** — the (often binary) variable used for DiD that represents whether the observation is part of the control or treatment group. Be sure to reply **None** if no variable seems appropriate, or choose the **single most appropriate column name** that best represents the outcome. Your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words."},
        {"role": "user", "content": "What is the group variable in the data? Reply None if there is no group variable."}
    ]
)['choices'][0]['message']['content']


# Try to automate
running_variable = "None" # For RDD. Try LoRD3
cutoff_value = "None" # For RDD. Try LoRD3


# Data preprocessing (Drop non-numeric columns and missing values, and columns with high correlation)
numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.dropna()
corr = numeric_data.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.999)]
numeric_data = numeric_data.drop(columns=to_drop)

# Causal discovery
alpha = 0.05
if latent_confounders:
    cg, _ = fci(numeric_data.values, alpha=alpha, ci_test=fisherz)
else:
    cg = pc(numeric_data.values, alpha=alpha, ci_test=fisherz)

def cl_to_nx(names):
    cl2real = {f"X{i+1}": col for i, col in enumerate(names)}

    graph = cg.G if hasattr(cg, "G") else cg
    edges  = (graph.get_graph_edges()
              if hasattr(graph, "get_graph_edges")
              else graph.get_edges())

    G = nx.DiGraph()
    G.add_nodes_from(names)

    for e in edges:
        n1, n2 = e.get_node1().get_name(), e.get_node2().get_name()
        e1, e2 = e.get_endpoint1(), e.get_endpoint2()

        a = cl2real[n1]
        b = cl2real[n2]

        if e1 == Endpoint.TAIL  and e2 == Endpoint.ARROW:
            G.add_edge(a, b)
        elif e1 == Endpoint.ARROW and e2 == Endpoint.TAIL:
            G.add_edge(b, a)

    return G

nx_graph = cl_to_nx(list(numeric_data.columns))
sample_size = data.shape[0]
covariates = list(set(nx_graph.nodes) - {treatment, outcome})

result = recommend_inference_algorithm(
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
    group_variable
)

algorithm = result['recommendation']
if result.get('mediators'):
    mediator = result['mediators']
if result.get('instruments'):
    instruments = result['instruments']
if result.get('adjustment_set'):
    adjustment_set = result['adjustment_set']

match result["recommendation"]:
    case 'G Computation':
        estimate = g_computation.estimate(
            data,
            treatment,
            outcome,
            adjustment_set
        )
    case 'Propensity Score':
        estimate = propensity_score.estimate(
            data,
            treatment,
            outcome,
            adjustment_set
        )
    case 'DML':
        estimate = double_machine_learning.estimate(
            data,
            treatment,
            outcome,
            adjustment_set,
            sample_size
        )
    case 'IV':
        estimate = iv.estimate(
            data,
            treatment,
            outcome,
            instruments,
            covariates
        )
    case 'RDD':
        estimate = rdd.estimate(
            data,
            outcome,
            running_variable,
            cutoff_value,
            covariates,
            order=1,
            bandwidth=None
        )
    case 'DiD':
        estimate = did.estimate(
            data,
            treatment,
            outcome,
            time_variable,
            group_variable,
            covariates
        )
    case 'OLS':
        estimate = ols.estimate(
            data,
            treatment,
            outcome,
            adjustment_set
        )
    case 'Frontdoor Adjustment':
        estimate = frontdoor_adjustment.estimate(
            data,
            treatment,
            mediator,
            adjustment_set,
            outcome
        )
    case 'RCT':
        estimate = ols.estimate(
            data,
            treatment,
            outcome,
            adjustment_set
        )
    case _:
        print(algorithm)
