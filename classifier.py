import networkx as nx

def recommend_causal_estimator(
    treatment,
    outcome,
    covariates,
    assignment_style='none',
    latent_confounders=False,
    dag=None,
    cutoff_value=None,
    time_variable=None,
    group_variable=None
):
    treat_name = treatment.name if hasattr(treatment, 'name') else treatment
    outcome_name = outcome.name if hasattr(outcome, 'name') else outcome
    instrument_var = None
    mediator_var = None

    if assignment_style == 'randomized':
        rec = 'ols'
    elif assignment_style == 'cutoff' or cutoff_value is not None:
        rec = 'regression discontinuity design'
    elif time_variable and group_variable:
        rec = 'did'
    else:
        if dag is not None:
            for z in dag.predecessors(treat_name):
                if z in {treat_name, outcome_name}:
                    continue
                g2 = dag.copy()
                g2.remove_node(treat_name)
                if not nx.has_path(g2, z, outcome_name):
                    instrument_var = z
                    break
            if instrument_var is None:
                for m in dag.successors(treat_name):
                    if nx.has_path(dag, m, outcome_name):
                        mediator_var = m
                        break
        if assignment_style in ('observational','none'):
            if latent_confounders:
                if instrument_var:
                    rec = 'instrumental variables'
                elif mediator_var:
                    rec = 'frontdoor adjustment'
                else:
                    return "No valid recommendation available due to latent confounders."
            else:
                num_cov = len(covariates or [])
                if num_cov <= 2:
                    rec = 'backdoor adjustment'
                elif num_cov <= 5:
                    rec = 'g computation'
                elif num_cov <= 10:
                    rec = 'propensity score'
                else:
                    rec = 'double machine learning'

    return {
        'recommendation': rec,
        'instrument': instrument_var,
        'mediator': mediator_var
    }
