import pandas as pd
import networkx as nx
import numpy as np
from classifier import recommend_causal_estimator
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from causallearn.graph.Endpoint import Endpoint
from inference_algorithms import g_computation, propensity_score, double_machine_learning, iv, rdd, did, ols, frontdoor_adjustment

data = pd.read_csv('ajrcomment.csv')
treatment = 'risk'
outcome = 'loggdp'
assignment_style = 'observational' # 'observational', 'randomized', 'cutoff', or 'none'
effect = 'ate' # 'ate', 'att', or 'cate'
latent_confounders = False  # Set to True if latent confounders are present
ml_Q = None # For DML, scikit-learn estimator for Q-model
ml_g = None # For DML, scikit-learn estimator for g-model
n_splits = 5 # For DML
running_variable = None # For RDD
cutoff_value = None # For RDD
time_variable = None # For DiD

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

result = recommend_causal_estimator(
    treatment,
    outcome,
    covariates,
    nx_graph,
    sample_size,
    assignment_style,
    latent_confounders,
    cutoff_value,
    time_variable
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
        estimate = g_computation(
            data,
            treatment,
            outcome,
            adjustment_set,
            effect
        )
    case 'Propensity Score':
        estimate = propensity_score(
            data,
            treatment,
            outcome,
            adjustment_set,
            effect
        )
    case 'DML':
        estimate = double_machine_learning(
            data,
            treatment,
            outcome,
            adjustment_set,
            ml_Q,
            ml_g,
            n_splits
        )
    case 'IV':
        estimate = iv(
            data,
            treatment,
            outcome,
            instruments,
            covariates
        )
    case 'RDD':
        estimate = rdd(
            data,
            outcome,
            running_variable,
            cutoff_value,
            covariates,
            order=1,
            bandwidth=None
        )
    case 'DiD':
        estimate = did(
            data,
            treatment,
            outcome,
            time_variable,
            covariates
        )
    case 'OLS':
        estimate = ols(
            data,
            treatment,
            outcome,
            adjustment_set
        )
    case 'Frontdoor Adjustment':
        estimate = frontdoor_adjustment(
            data,
            treatment,
            mediator,
            outcome,
            adjustment_set
        )
    case _:
        print(algorithm)
