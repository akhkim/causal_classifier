import numpy as np
import pandas as pd
import networkx as nx
from llm_query import parse_intent
from inference_classifier import run_inference_algorithm, match_algorithm
from discovery_classifier import run_discovery_algorithm
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci
from discovery_algorithms import fci
from discovery_algorithms import notears
from preprocessing import full_preprocess
from inference_algorithms.test_assumptions import run_all_tests

# Variables the user has to provide
data = pd.read_csv('proximity.csv')
question = "How does the education affect earnings?"
assignment_style = 'observational' # 'observational', 'randomized'
latent_confounders = False  # Set to True if latent confounders are present

# treatment = "educ"
# outcome = "wage"
# time_variable = "None"
# group_variable = "None"
# running_variable = "None"
# cutoff_value = "None"
# inference_algorithm = "IV"
# instruments = ["nearc2","nearc4"]
# adjustment_set = []
# mediators = []

numeric_data = full_preprocess(data)

# Variables automated by LLM
context = numeric_data.head(3).to_string(index=False)

treatment, outcome, time_variable, group_variable, inference_algorithm, instruments, mediators, adjustment_set, running_variable, cutoff_value = parse_intent(question, context)
print("treatment:", treatment)
print("outcome:", outcome)
print("time variable:", time_variable)
print("group variable:", group_variable)
print("inference algorithm:", inference_algorithm)
print("instruments:", instruments)
print("mediators:", mediators)

# alpha = 0.05
# if latent_confounders:
#     nx_graph = fci.run(numeric_data.values)
# else:
#     mapping = dict(enumerate(numeric_data.columns))
#     cg = pc(numeric_data.values, alpha=alpha, ci_test=kci)
#     nx_graph = nx.from_numpy_array(cg.G.graph, create_using=nx.DiGraph)
#     nx.relabel_nodes(nx_graph, mapping, copy=False)
# nx_graph = notears.run(numeric_data)
nx_graph = run_discovery_algorithm(numeric_data, latent_confounders)
sample_size = numeric_data.shape[0]
covariates = list(set(nx_graph.nodes) - {treatment, outcome})
for edge in nx_graph.edges():
    print(edge)

run_all_tests(numeric_data, treatment, outcome, covariates, group_variable, instruments, cutoff_value, running_variable, mediators)

if not inference_algorithm:
    result = run_inference_algorithm(
        numeric_data,
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
else:
    result = match_algorithm(inference_algorithm, numeric_data, treatment, outcome, covariates, sample_size,
                    cutoff_value, time_variable, group_variable, running_variable, 
                    mediators, instruments, adjustment_set)

print(result)