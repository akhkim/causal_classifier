import pandas as pd
import networkx as nx
from classifier import recommend_causal_estimator
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.Endpoint import Endpoint
from networkx.drawing.nx_pydot import from_pydot
from inference_algorithms import g_computation, propensity_score, double_machine_learning, iv, rdd, did, ols, frontdoor_adjustment

data = pd.read_csv('iv_causal.csv')
treatment = 'v0'
outcome = 'y'
assignment_style = 'observational' # 'observational', 'randomized', 'cutoff', or 'none'
effect = 'ate' # 'ate', 'att', or 'cate'
latent_confounders = True  # Set to True if latent confounders are present
ml_Q = None # For DML, scikit-learn estimator for Q-model
ml_g = None # For DML, scikit-learn estimator for g-model
n_splits = 5 # For DML
running_variable = None # For RDD
cutoff_value = None # For RDD
time_variable = None # For DiD

# Causal discovery
alpha = 0.05
if latent_confounders:
    cg, _ = fci(data.values, alpha=alpha, ci_test=fisherz)
else:
    cg = pc(data.values, alpha=alpha, ci_test=fisherz)

def cl_to_nx(causal_graph, names):
    graph = cg.G if hasattr(cg, "G") else cg   # pc-graph vs. fci-graph

    # 3.  grab the edge list (API changed once)
    edges = (graph.get_graph_edges()           # ≥ 0.3.x
             if hasattr(graph, "get_graph_edges")
             else graph.get_edges())           # ≤ 0.2.x

    G = nx.DiGraph()
    G.add_nodes_from(names)

    for e in edges:
        a, b = e.get_node1().get_name(), e.get_node2().get_name()
        e1, e2 = e.get_endpoint1(),       e.get_endpoint2()

        if   e1 == Endpoint.TAIL  and e2 == Endpoint.ARROW:  # a → b
            G.add_edge(a, b)
        elif e1 == Endpoint.ARROW and e2 == Endpoint.TAIL:   # b → a
            G.add_edge(b, a)
        # other endpoint patterns (—, ↔, ◦) can be added here as needed

    return G

nx_graph = cl_to_nx(cg, list(data.columns))
sample_size = data.shape[0]
covariates = list(set(nx_graph.nodes) - {treatment, outcome})

# Get recommendation
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
    case 'g computation':
        estimate = g_computation(
            data,
            treatment,
            outcome,
            adjustment_set,
            effect
        )
    case 'propensity score':
        estimate = propensity_score(
            data,
            treatment,
            outcome,
            adjustment_set,
            effect
        )
    case 'double machine learning':
        estimate = double_machine_learning(
            data,
            treatment,
            outcome,
            adjustment_set,
            ml_Q,
            ml_g,
            n_splits
        )
    case 'instrumental variables':
        estimate = iv(
            data,
            treatment,
            outcome,
            instruments,
            covariates
        )
    case 'regression discontinuity design':
        estimate = rdd(
            data,
            outcome,
            running_variable,
            cutoff_value,
            covariates,
            order=1,
            bandwidth=None
        )
    case 'did':
        estimate = did(
            data,
            treatment,
            outcome,
            time_variable,
            covariates
        )
    case 'ols':
        estimate = ols(
            data,
            treatment,
            outcome,
            adjustment_set
        )
    case 'frontdoor adjustment':
        estimate = frontdoor_adjustment(
            data,
            treatment,
            mediator,
            outcome,
            adjustment_set
        )
    case _:
        print(algorithm)
