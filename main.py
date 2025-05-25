import pandas as pd
from classifier import recommend_causal_estimator
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from causallearn.utils.GraphUtils import GraphUtils
from algorithms import backdoor_adjustment, g_computation, propensity_score, double_machine_learning, iv, rdd, did, ols, frontdoor_adjustment

data = pd.read_csv('data.csv')

# Edit the following variables
treatment = data['treatment']
outcome = data['outcome']
effect = 'ate' # 'ate', 'att', or 'cate'
assignment_style = 'randomized' # 'randomized', 'observational', 'cutoff', or 'none'
latent_confouners = False # True or False

ml_Q = None # For DML, scikit-learn estimator for Q-model
ml_g = None # For DML, scikit-learn estimator for g-model
n_splits = 5 # For DML
running_variable = None # For RDD
cutoff_value = None # For RDD
time_variable = None # For DiD
group_variable = None # For DiD

alpha = 0.05  # Significance level for the causal discovery algorithm

# Create a DAG
if latent_confouners:
    cg = fci(data.values, alpha, ci_test=fisherz)
else:
    cg = pc(data.values, alpha, ci_test=fisherz)
nx_graph = GraphUtils.to_nx_graph(cg.G, labels=list(data.columns))

parents_A = set(nx_graph.predecessors(treatment))
parents_Y = set(nx_graph.predecessors(outcome))
covariates = list((parents_A & parents_Y) - {treatment, outcome})

result = recommend_causal_estimator(treatment, outcome, covariates, assignment_style, latent_confouners, alpha, nx_graph, cutoff_value)

algorithm = result['recommendation']
instrument = result['instrument']
mediator = result['mediator']

match algorithm:
    case 'backdoor adjustment':
        estimate = backdoor_adjustment(
            data,
            treatment,
            outcome,
            covariates,
            effect
        )
    case 'g computation':
        estimate = g_computation(
            data,
            treatment,
            outcome,
            covariates,
            effect
        )
    case 'propensity score':
        estimate = propensity_score(
            data,
            treatment,
            outcome,
            covariates,
            effect
        )
    case 'double machine learning':
        estimate = double_machine_learning(
            data,
            treatment,
            outcome,
            covariates,
            ml_Q,
            ml_g,
            n_splits
        )
    case 'instrumental variables':
        estimate = iv(
            data,
            treatment,
            outcome,
            instrument,
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
            outcome
        )
    case 'frontdoor adjustment':
        estimate = frontdoor_adjustment(
            data,
            treatment,
            mediator,
            outcome
        )
    case _:
        raise ValueError(f"Unknown algorithm: {algorithm}")