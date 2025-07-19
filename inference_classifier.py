import networkx as nx
from itertools import combinations
from inference_algorithms import g_computation, propensity_score, double_machine_learning, iv, rdd, did, ols, frontdoor_adjustment

def find_backdoor_set(dag, treatment, outcome, covariates):
    # Check if graph is a DAG first
    if not nx.is_directed_acyclic_graph(dag):
        print("Warning: Graph is not a DAG, backdoor criterion may not be reliable")
        return None
    
    bd_graph = dag.copy()
    bd_graph.remove_edges_from(list(bd_graph.out_edges(treatment)))
    
    # Search for the smallest d-separator that contains no descendants of X
    for k in range(len(covariates) + 1):
        for Z in combinations(covariates, k):
            if set(Z) & nx.descendants(dag, treatment):
                continue
            
            try:
                if nx.d_separated(bd_graph, {treatment}, {outcome}, set(Z)):
                    return set(Z)
            except nx.NetworkXError as e:
                print(f"D-separation test failed in backdoor detection: {e}")
                continue
    
    return None

def find_frontdoor_set(dag, treatment, outcome):
    # Check if graph is a DAG first
    if not nx.is_directed_acyclic_graph(dag):
        print("Warning: Graph is not a DAG, frontdoor criterion may not be reliable")
        return []
    
    frontdoor_candidates = set()
    
    # Find all mediators on directed paths from treatment to outcome
    try:
        for path in nx.all_simple_paths(dag, treatment, outcome):
            if len(path) > 2:
                frontdoor_candidates.update(path[1:-1])
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    
    # Check frontdoor criteria using d-separation
    valid_mediators = []
    for m in frontdoor_candidates:
        try:
            if (nx.d_separated(dag, {treatment}, {outcome}, {m}) and
                not nx.d_separated(dag, {treatment}, {m}, set()) and
                nx.d_separated(dag, {m}, {outcome}, {treatment})):
                valid_mediators.append(m)
        except nx.NetworkXError as e:
            print(f"D-separation test failed in frontdoor detection for mediator {m}: {e}")
            continue
    
    return valid_mediators

def find_instruments(dag, treatment, outcome):
    if not nx.is_directed_acyclic_graph(dag):
        print(f"Warning: Graph contains cycles, attempting to remove cycles for instrument detection")
        
        # Option 1: Try to break cycles by removing some edges
        dag_copy = dag.copy()
        try:
            # Remove cycles by finding and breaking them
            cycles = list(nx.simple_cycles(dag_copy))
            if cycles:
                # Remove one edge from each cycle to break it
                for cycle in cycles:
                    if len(cycle) >= 2:
                        dag_copy.remove_edge(cycle[0], cycle[1])
                
                # Check if it's now a DAG
                if not nx.is_directed_acyclic_graph(dag_copy):
                    print("Could not create valid DAG for instrument detection")
                    return []
                dag = dag_copy
            else:
                print("No simple cycles found, but graph still not DAG")
                return []
        except:
            print("Failed to process cycles in graph")
            return []
    
    latent_pairs = []
    g = dag.copy()
    
    # For every pair (A,B) that shares an un-observed cause,
    # create a fresh latent node U_(A,B) → {A, B}.
    for a, b in latent_pairs:
        latent_name = f"U_{a}_{b}"
        if latent_name in g:
            continue
        g.add_node(latent_name, latent=True)
        g.add_edge(latent_name, a)
        g.add_edge(latent_name, b)
    
    instruments = []
    for node in g.nodes:
        if node in {treatment, outcome} or g.nodes[node].get("latent", False):
            continue
        
        try:
            # IV independence: Z ⫫ Y | X (no open back-door once we intervene on X)
            ind_cond = nx.is_d_separator(g, {node}, {outcome}, {treatment})
            
            # Relevance: Z ∦ X (a path from Z to X is still open)
            rel_cond = not nx.is_d_separator(g, {node}, {treatment}, set())
            
            if ind_cond and rel_cond:
                instruments.append(node)
        except nx.NetworkXError as e:
            print(f"D-separation test failed for node {node}: {e}")
            continue
    
    return instruments

def match_algorithm(algorithm, data, treatment, outcome, covariates, sample_size,
                    cutoff_value, time_variable, group_variable, running_variable, 
                    mediator = None, instruments = None, adjustment_set = None):
    match algorithm:
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
            print("No available inference methods")
            return None

    return estimate

def recommend_inference_algorithm(data, treatment, outcome, covariates, dag, sample_size,
                              assignment_style, latent_confounders,
                              cutoff_value, time_variable, group_variable):

    if assignment_style == 'randomized':
        return {'recommendation': 'RCT'}
    
    if cutoff_value != "None":
        if rdd.diagnose(data, treatment, outcome, cutoff_value, covariates):
            return {'recommendation': 'RDD'}
    
    if time_variable != "None":
        if did.diagnose(data, group_variable, time_variable, treatment, outcome):
            return {'recommendation': 'DiD'}

    instruments = find_instruments(dag, treatment, outcome)
    if len(instruments) > 0:
        return {'recommendation': 'IV',
                'instruments': instruments
        }
    
    mediators = find_frontdoor_set(dag, treatment, outcome)
    if len(mediators) > 0:
        return {
            'recommendation': 'Frontdoor Adjustment',
            'mediators': mediators
        }

    # Backdoor Adjustment Algorithms
    if not latent_confounders:
        backdoor_set = find_backdoor_set(dag, treatment, outcome, covariates)
        if backdoor_set:
            backdoor_size = len(backdoor_set)
            observations_per_variable = sample_size / backdoor_size

            # 4a. small, low-dimensional ⇒ OLS regression
            # Justification: Regression models recommend N ≥ 25 for stable inference with higher variance
            if observations_per_variable >= 25:
                return {
                    'recommendation': 'OLS',
                    'adjustment_set': backdoor_set
                }

            # 4b. moderate dimension ⇒ propensity-score methods
            # Justification: Propensity score can reduce dimensionality and Events Per Variable (EPV) suggests 15-25 observations per variable
            elif 15 <= observations_per_variable < 25:
                return {
                    'recommendation': 'Propensity Score',
                    'adjustment_set': backdoor_set
                }

            # 4c. high dimension with ample observations ⇒ Double ML
            # Justification: High dimensionality causes DML to function better than the others, and cross fitting can help with overfitting
            elif sample_size >= 500:
                return {
                    'recommendation': 'DML',
                    'adjustment_set': backdoor_set
                }

            # 4d. fallback parametric g-formula
            # Justification: Most broadly applicable, but vulnerable to model misspecification
            return {
                'recommendation': 'G Computation',
                'adjustment_set': backdoor_set
            }
        
    return {'recommendation': 'OLS'}

def run_inference_algorithm(data, treatment, outcome, covariates, dag, sample_size,
                            assignment_style, latent_confounders,
                            cutoff_value, time_variable, group_variable, running_variable):

    result = recommend_inference_algorithm(
        data,
        treatment,
        outcome,
        covariates,
        dag,
        sample_size,
        assignment_style,
        latent_confounders,
        cutoff_value,
        time_variable,
        group_variable,
    )

    algorithm = result["recommendation"]
    print(algorithm)
    mediator = None
    instruments = None
    adjustment_set = None

    if result.get('mediators'):
        mediator = result['mediators']
    
    if result.get('instruments'):
        instruments = result['instruments']
    
    if result.get('adjustment_set'):
        adjustment_set = result['adjustment_set']

    print(instruments)

    estimate = match_algorithm(algorithm, data, treatment, outcome, covariates, sample_size,
                               cutoff_value, time_variable, group_variable, running_variable, 
                               mediator, instruments, adjustment_set)

    return estimate