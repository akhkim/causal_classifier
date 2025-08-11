import networkx as nx
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from .inference_algorithms import g_computation, propensity_score, double_machine_learning, iv, rdd, did, ols, frontdoor_adjustment, mr

def assess_study_design_characteristics(data, treatment, outcome, covariates, dag, sample_size,
                                      assignment_style, latent_confounders, cutoff_value, 
                                      time_variable, group_variable, llm_results=None):
    """
    Comprehensive assessment of study design characteristics for optimal algorithm selection.
    
    Returns a detailed characterization that informs the decision tree.
    """
    
    # Extract latent confounders from the DAG if present
    dag_latent_confounders = []
    latent_pairs = []
    
    if dag:
        # Find all U_ nodes (latent confounders identified by FCI)
        for node in dag.nodes():
            if str(node).startswith('U_'):
                dag_latent_confounders.append(node)
                # Extract the variables this latent confounder affects
                affected_vars = list(dag.successors(node))
                if len(affected_vars) == 2:
                    latent_pairs.append((node, affected_vars[0], affected_vars[1]))
        
        # Update latent_confounders flag if FCI found latent confounders
        if dag_latent_confounders:
            latent_confounders = True
            print(f"Found {len(dag_latent_confounders)} latent confounders from FCI: {dag_latent_confounders}")
    
    characteristics = {
        'sample_size': sample_size,
        'n_covariates': len(covariates) if covariates else 0,
        'assignment_style': assignment_style,
        'latent_confounders': latent_confounders,
        'dag_latent_confounders': dag_latent_confounders,
        'latent_pairs': latent_pairs,
        
        # Design-specific indicators
        'has_randomization': assignment_style == 'randomized',
        'has_discontinuity': cutoff_value not in [None, 'None'],
        'has_time_variation': time_variable not in [None, 'None'], 
        'has_group_structure': group_variable not in [None, 'None'],
        
        # Identification strategy availability
        'instruments_available': False,
        'mediators_available': False,
        'valid_backdoor_set': False,
        'sufficient_power': True,
        
        # Data quality indicators
        'treatment_type': 'binary',  # Will be updated based on data
        'outcome_type': 'continuous',  # Will be updated based on data
        'covariate_balance': None,
        'overlap_quality': None,
        
        # Theoretical requirements
        'parallel_trends_likely': None,
        'rdd_assumptions_met': None,
        'iv_assumptions_met': None,
        'no_unmeasured_confounding_likely': not latent_confounders,
        
        # Advanced characteristics
        'is_gwas_data': False,
        'high_dimensional': False,
        'sparse_treatment': False
    }
    
    # Assess treatment and outcome types
    if treatment in data.columns:
        unique_treatment = data[treatment].nunique()
        characteristics['treatment_type'] = 'binary' if unique_treatment == 2 else 'continuous' if unique_treatment > 10 else 'categorical'
        characteristics['sparse_treatment'] = (data[treatment].mean() < 0.1 or data[treatment].mean() > 0.9) if unique_treatment == 2 else False
    
    if outcome in data.columns:
        unique_outcome = data[outcome].nunique()
        characteristics['outcome_type'] = 'binary' if unique_outcome == 2 else 'continuous' if unique_outcome > 10 else 'categorical'
    
    # Check for high-dimensionality
    characteristics['high_dimensional'] = characteristics['n_covariates'] > sample_size / 10
    
    # Assess power considerations
    min_power_size = max(100, 10 * characteristics['n_covariates'])
    characteristics['sufficient_power'] = sample_size >= min_power_size
    
    # Check for GWAS data indicators
    if llm_results:
        characteristics['is_gwas_data'] = llm_results.get('is_questionnaire_with_genetics', {}).get('value') == 'Yes'
    
    # Assess available identification strategies
    characteristics.update(_assess_identification_strategies(data, treatment, outcome, covariates, dag, llm_results))
    
    return characteristics

def _assess_identification_strategies(data, treatment, outcome, covariates, dag, llm_results):
    """Assess the availability and quality of different identification strategies."""
    
    strategies = {
        'instruments_available': False,
        'mediators_available': False, 
        'valid_backdoor_set': False,
        'instruments_list': [],
        'mediators_list': [],
        'backdoor_set': set()
    }
    
    # 1. Instrumental Variables Assessment
    graph_instruments = find_instruments(dag, treatment, outcome) if dag else []
    llm_instruments = []
    
    if llm_results and 'instruments' in llm_results and llm_results['instruments']['value'] != 'None':
        llm_instruments = [inst.strip() for inst in llm_results['instruments']['value'].split(',')]
    
    all_instruments = list(set(graph_instruments + llm_instruments))
    
    if all_instruments:
        # Validate instruments with data-driven tests
        valid_instruments = []
        for inst in all_instruments:
            if inst in data.columns:
                # Basic relevance check (correlation with treatment)
                if treatment in data.columns:
                    relevance = abs(data[inst].corr(data[treatment])) > 0.1
                    if relevance:
                        valid_instruments.append(inst)
        
        strategies['instruments_available'] = len(valid_instruments) > 0
        strategies['instruments_list'] = valid_instruments
    
    # 2. Mediators Assessment  
    graph_mediators = find_frontdoor_set(dag, treatment, outcome) if dag else []
    llm_mediators = []
    
    if llm_results and 'mediators' in llm_results and llm_results['mediators']['value'] != 'None':
        llm_mediators = [med.strip() for med in llm_results['mediators']['value'].split(',')]
    
    all_mediators = list(set(graph_mediators + llm_mediators))
    strategies['mediators_available'] = len(all_mediators) > 0
    strategies['mediators_list'] = all_mediators
    
    # 3. Backdoor Set Assessment
    graph_backdoor = find_backdoor_set(dag, treatment, outcome, covariates) if dag else None
    llm_adjustment = set()
    
    if llm_results and 'adjustment_set' in llm_results and llm_results['adjustment_set']['value'] != 'None':
        llm_adjustment = set([adj.strip() for adj in llm_results['adjustment_set']['value'].split(',')])
    
    if graph_backdoor:
        strategies['backdoor_set'] = graph_backdoor.union(llm_adjustment)
        strategies['valid_backdoor_set'] = True
    elif llm_adjustment:
        strategies['backdoor_set'] = llm_adjustment  
        strategies['valid_backdoor_set'] = True
    elif covariates:
        # Fallback to all covariates if no specific backdoor set found
        strategies['backdoor_set'] = set(covariates)
        strategies['valid_backdoor_set'] = True
    
    return strategies

def _evaluate_rdd_suitability(data, treatment, outcome, cutoff_value, covariates, study_chars):
    """Evaluate suitability of Regression Discontinuity Design with diagnostic tests."""
    
    reasoning = []
    diagnostics = {}
    confidence = 0.5  # Base confidence
    
    try:
        # Import and run RDD diagnostics
        from .inference_algorithms import rdd
        rdd_diag = rdd.diagnose(data, treatment, outcome, cutoff_value, covariates)
        diagnostics.update(rdd_diag)
        
        # Assess diagnostic results
        if rdd_diag.get('overall_valid', False):
            confidence = 0.9
            reasoning.append("RDD diagnostics passed - sharp discontinuity confirmed")
        else:
            failed_tests = [k for k, v in rdd_diag.items() if not v and k != 'overall_valid']
            confidence = max(0.4, 0.9 - 0.1 * len(failed_tests))
            reasoning.append(f"RDD diagnostics partially failed: {failed_tests}")
        
        # Additional theoretical considerations
        if study_chars['sufficient_power']:
            confidence += 0.05
            reasoning.append("Sufficient sample size for RDD")
        
        # Check for manipulation concerns
        if 'manipulation_test' in rdd_diag and not rdd_diag['manipulation_test']:
            confidence -= 0.2
            reasoning.append("Warning: Potential manipulation of running variable")
            
    except Exception as e:
        confidence = 0.3
        reasoning.append(f"RDD diagnostic tests failed: {str(e)}")
        diagnostics['error'] = str(e)
    
    return confidence, reasoning, diagnostics

def _evaluate_did_suitability(data, treatment, outcome, time_variable, group_variable, covariates, study_chars):
    """Evaluate suitability of Difference-in-Differences with diagnostic tests."""
    
    reasoning = []
    diagnostics = {}
    confidence = 0.5  # Base confidence
    
    try:
        # Import and run DiD diagnostics
        from .inference_algorithms import did
        did_diag = did.diagnose(data, group_variable, time_variable, treatment, outcome)
        diagnostics.update(did_diag)
        
        # Assess diagnostic results
        if did_diag.get('overall_valid', False):
            confidence = 0.9
            reasoning.append("DiD diagnostics passed - parallel trends confirmed")
        else:
            failed_tests = [k for k, v in did_diag.items() if not v and k != 'overall_valid']
            confidence = max(0.4, 0.9 - 0.15 * len(failed_tests))  # Harsher penalty for DiD
            reasoning.append(f"DiD diagnostics partially failed: {failed_tests}")
        
        # Additional theoretical considerations
        if study_chars['sufficient_power']:
            confidence += 0.05
            reasoning.append("Sufficient sample size for DiD")
            
        # Check for common shocks assumption
        if 'parallel_trends' in did_diag and did_diag['parallel_trends']:
            confidence += 0.1
            reasoning.append("Parallel trends assumption supported by data")
            
    except Exception as e:
        confidence = 0.3
        reasoning.append(f"DiD diagnostic tests failed: {str(e)}")
        diagnostics['error'] = str(e)
    
    return confidence, reasoning, diagnostics

def _evaluate_iv_suitability(data, treatment, outcome, instruments, covariates, study_chars):
    """Evaluate suitability of Instrumental Variables with strength and validity tests."""
    
    reasoning = []
    diagnostics = {}
    confidence = 0.5  # Base confidence
    recommended_method = 'IV'  # Default
    
    # Choose between IV and MR based on data characteristics
    if study_chars['is_gwas_data']:
        recommended_method = 'MR'
        confidence = 0.7  # Higher base confidence for genetic instruments
        reasoning.append("GWAS data detected - using Mendelian Randomization")
    
    try:
        # Import and run appropriate diagnostics
        if recommended_method == 'MR':
            from .inference_algorithms import mr
            iv_diag = mr.diagnose(data, treatment, outcome, covariates)
        else:
            from .inference_algorithms import iv
            iv_diag = iv.diagnose(data, treatment, outcome, instruments, covariates)
            
        diagnostics.update(iv_diag)
        
        # Assess diagnostic results
        if iv_diag.get('overall_valid', False):
            confidence = min(confidence + 0.2, 0.95)
            reasoning.append(f"{recommended_method} diagnostics passed - strong instruments confirmed")
        else:
            failed_tests = [k for k, v in iv_diag.items() if not v and k != 'overall_valid']
            confidence = max(0.3, confidence - 0.1 * len(failed_tests))
            reasoning.append(f"{recommended_method} diagnostics partially failed: {failed_tests}")
        
        # Additional considerations
        if len(instruments) > 1:
            confidence += 0.05
            reasoning.append(f"Multiple instruments available ({len(instruments)})")
            
        # Check instrument strength
        if 'weak_instruments' in iv_diag and not iv_diag['weak_instruments']:
            confidence += 0.1
            reasoning.append("Strong instruments confirmed (F-stat > 10)")
        elif 'weak_instruments' in iv_diag and iv_diag['weak_instruments']:
            confidence -= 0.2
            reasoning.append("Warning: Weak instruments detected")
            
    except Exception as e:
        confidence = 0.2
        reasoning.append(f"{recommended_method} diagnostic tests failed: {str(e)}")
        diagnostics['error'] = str(e)
    
    return confidence, reasoning, diagnostics, recommended_method

def _evaluate_frontdoor_suitability(data, treatment, outcome, mediators, covariates, study_chars):
    """Evaluate suitability of Frontdoor Adjustment with mediation tests."""
    
    reasoning = []
    diagnostics = {}
    confidence = 0.4  # Lower base confidence as frontdoor is restrictive
    
    if not mediators:
        return 0.0, ["No mediators available"], {}
    
    try:
        # Import and run frontdoor diagnostics
        from .inference_algorithms import frontdoor_adjustment
        frontdoor_diag = frontdoor_adjustment.diagnose(data, treatment, mediators[0], outcome, covariates)
        diagnostics.update(frontdoor_diag)
        
        # Assess diagnostic results
        if frontdoor_diag.get('overall_valid', False):
            confidence = 0.8
            reasoning.append("Frontdoor diagnostics passed - complete mediation confirmed")
        else:
            failed_tests = [k for k, v in frontdoor_diag.items() if not v and k != 'overall_valid']
            confidence = max(0.2, 0.8 - 0.2 * len(failed_tests))
            reasoning.append(f"Frontdoor diagnostics partially failed: {failed_tests}")
        
        # Additional considerations
        if len(mediators) == 1:
            confidence += 0.05
            reasoning.append("Single mediator identified - cleaner identification")
        else:
            reasoning.append(f"Multiple mediators ({len(mediators)}) - may complicate identification")
            
    except Exception as e:
        confidence = 0.1
        reasoning.append(f"Frontdoor diagnostic tests failed: {str(e)}")
        diagnostics['error'] = str(e)
    
    return confidence, reasoning, diagnostics

def _evaluate_backdoor_methods(data, treatment, outcome, backdoor_set, study_chars):
    """Select optimal backdoor method based on data characteristics and diagnostics."""
    
    reasoning = []
    diagnostics = {}
    n_covariates = len(backdoor_set) if backdoor_set else 0
    sample_size = study_chars['sample_size']
    observations_per_covariate = sample_size / max(n_covariates, 1)
    
    # Decision tree for backdoor methods based on empirical guidelines
    
    # High-dimensional case: Use Double Machine Learning
    if study_chars['high_dimensional'] or (n_covariates > 20 and sample_size >= 1000):
        try:
            from .inference_algorithms import double_machine_learning
            dml_diag = double_machine_learning.diagnose(data, treatment, outcome, list(backdoor_set))
            diagnostics.update(dml_diag)
            
            if dml_diag.get('overall_valid', False):
                confidence = 0.9
                reasoning.append(f"DML optimal for high-dimensional data: {n_covariates} covariates")
            else:
                confidence = 0.7
                reasoning.append(f"DML chosen for high-dim but diagnostics partially failed")
                
            return confidence, 'DML', reasoning, diagnostics
            
        except Exception as e:
            reasoning.append(f"DML failed, falling back: {str(e)}")
    
    # Well-powered linear case: Use OLS if assumptions met
    if observations_per_covariate >= 25 and not study_chars['treatment_type'] == 'continuous':
        try:
            from .inference_algorithms import ols
            ols_diag = ols.diagnose(data, treatment, outcome, list(backdoor_set))
            diagnostics.update(ols_diag)
            
            if ols_diag.get('overall_valid', False):
                confidence = 0.9
                reasoning.append(f"OLS optimal: {observations_per_covariate:.1f} obs/covariate, assumptions met")
                return confidence, 'OLS', reasoning, diagnostics
            else:
                reasoning.append("OLS assumptions violated, trying alternatives")
                
        except Exception as e:
            reasoning.append(f"OLS diagnostics failed: {str(e)}")
    
    # Moderate power case: Use Propensity Score methods
    if 10 <= observations_per_covariate < 25 and study_chars['treatment_type'] == 'binary':
        try:
            from .inference_algorithms import propensity_score
            ps_diag = propensity_score.diagnose(data, treatment, outcome, list(backdoor_set))
            diagnostics.update(ps_diag)
            
            if ps_diag.get('overall_valid', False):
                confidence = 0.8
                reasoning.append(f"Propensity Score optimal: binary treatment, {observations_per_covariate:.1f} obs/covariate")
                return confidence, 'Propensity Score', reasoning, diagnostics
            else:
                reasoning.append("Propensity Score diagnostics partially failed")
                
        except Exception as e:
            reasoning.append(f"Propensity Score diagnostics failed: {str(e)}")
    
    # Default to G-Computation as robust alternative
    try:
        from .inference_algorithms import g_computation
        gc_diag = g_computation.diagnose(data, treatment, outcome, list(backdoor_set))
        diagnostics.update(gc_diag)
        
        if gc_diag.get('overall_valid', False):
            confidence = 0.8
            reasoning.append("G-Computation chosen as robust backdoor method")
        else:
            confidence = 0.6
            reasoning.append("G-Computation chosen as fallback despite diagnostic concerns")
            
        return confidence, 'G Computation', reasoning, diagnostics
        
    except Exception as e:
        # Final fallback to OLS
        confidence = 0.4
        reasoning.append(f"All methods failed diagnostics, using OLS as final fallback: {str(e)}")
        return confidence, 'OLS', reasoning, {'error': str(e)}

def _get_backdoor_fallbacks(primary_method, study_chars):
    """Get appropriate fallback methods for backdoor approaches."""
    
    fallbacks = []
    
    if primary_method == 'DML':
        fallbacks = ['G Computation', 'Propensity Score', 'OLS']
    elif primary_method == 'OLS':
        fallbacks = ['G Computation', 'Propensity Score']
    elif primary_method == 'Propensity Score':
        fallbacks = ['G Computation', 'OLS'] 
    elif primary_method == 'G Computation':
        fallbacks = ['OLS']
    else:
        fallbacks = ['OLS']  # Always have OLS as ultimate fallback
        
    return fallbacks

def _select_fallback_method(data, treatment, outcome, covariates, study_chars):
    """Select fallback method when no clear identification strategy is available."""
    
    reasoning = []
    
    # If we have covariates, try the most robust backdoor method
    if covariates:
        reasoning.append("Using covariates with robust method due to unclear identification")
        if study_chars['sample_size'] >= 500:
            return 0.5, 'G Computation', reasoning + ["G-Computation for robustness with moderate sample"]
        else:
            return 0.4, 'OLS', reasoning + ["OLS fallback for small sample size"]
    
    # No covariates available - very limited options
    reasoning.append("No covariates available - causal identification highly questionable")
    
    if study_chars['treatment_type'] == 'binary':
        return 0.3, 'OLS', reasoning + ["Simple difference in means (binary treatment)"]
    else:
        return 0.2, 'OLS', reasoning + ["Simple regression (continuous treatment)"]

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
        
        # Try to break cycles by removing some edges
        dag_copy = dag.copy()
        try:
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
                    mediators=None, instruments=None, adjustment_set=None, llm_results=None):
    """
    Simplified interface for directly executing a specified algorithm.
    This function bypasses the decision tree and directly executes the named algorithm.
    """
    
    print(f"=== DIRECTLY MATCHING ALGORITHM: {algorithm} ===")
    
    try:
        estimate = _execute_inference_algorithm(
            algorithm, data, treatment, outcome, covariates, sample_size,
            cutoff_value, time_variable, group_variable, running_variable,
            adjustment_set or [], instruments or [], mediators or [], llm_results,
            latent_confounders=[]  # No latent confounders info in direct matching
        )
        
        print(f"✓ {algorithm} executed successfully via direct matching")
        
        # Add minimal metadata 
        if isinstance(estimate, dict):
            estimate.update({
                'selected_algorithm': algorithm,
                'algorithm_confidence': 1.0,  # Full confidence since user specified
                'selection_method': 'direct_specification',
                'identification_strategy': 'user_specified'
            })
        
        return estimate
        
    except Exception as e:
        print(f"✗ Direct algorithm execution failed: {str(e)}")
        
        # Create fallback result
        fallback_estimate = {
            'causal_effect': 0.0,
            'confidence_interval': [0.0, 0.0],
            'p_value': 1.0,
            'method': f'{algorithm} (failed)',
            'error': f'Direct execution of {algorithm} failed: {str(e)}',
            'selected_algorithm': algorithm,
            'algorithm_confidence': 0.0,
            'selection_method': 'direct_specification_failed'
        }
        
        return fallback_estimate

def recommend_inference_algorithm(data, treatment, outcome, covariates, dag, sample_size,
                               assignment_style, latent_confounders, cutoff_value, 
                               time_variable, group_variable, llm_results=None):
    """
    Optimized causal inference algorithm recommendation based on theoretical decision tree.
    
    Decision hierarchy:
    1. RANDOMIZATION: If randomized → RCT analysis
    2. QUASI-EXPERIMENTAL DESIGNS: RDD, DiD with diagnostic validation  
    3. INSTRUMENTAL VARIABLES: IV/MR with strength and validity tests
    4. FRONTDOOR CRITERION: When complete mediation is available
    5. BACKDOOR CRITERION: Confounding control with method selection based on data characteristics
    6. FALLBACK: Robust methods when identification is uncertain
    """
    
    # Get comprehensive study characteristics
    study_chars = assess_study_design_characteristics(
        data, treatment, outcome, covariates, dag, sample_size,
        assignment_style, latent_confounders, cutoff_value,
        time_variable, group_variable, llm_results
    )
    
    # Initialize algorithm decision
    algorithm_decision = {
        'primary_algorithm': None,
        'confidence': 0.0,
        'reasoning': [],
        'fallback_algorithms': [],
        'study_characteristics': study_chars,
        'identification_strategy': None,
        'required_assumptions': [],
        'diagnostic_tests': {}
    }
    
    # TIER 1: RANDOMIZED EXPERIMENTS (Highest Internal Validity)
    if study_chars['has_randomization']:
        algorithm_decision.update({
            'primary_algorithm': 'RCT',
            'confidence': 0.98,
            'identification_strategy': 'randomization',
            'reasoning': ['Randomized assignment ensures unbiased treatment effect estimation'],
            'required_assumptions': ['Random assignment maintained', 'No attrition bias', 'SUTVA'],
            'fallback_algorithms': ['OLS']  # Simple fallback for randomized data
        })
        return algorithm_decision
    
    # TIER 2: QUASI-EXPERIMENTAL DESIGNS (Natural Experiments)
    
    # Tier 2A: Regression Discontinuity Design
    if study_chars['has_discontinuity']:
        rdd_confidence, rdd_reasoning, rdd_diagnostics = _evaluate_rdd_suitability(
            data, treatment, outcome, cutoff_value, covariates, study_chars
        )
        
        if rdd_confidence >= 0.7:  # High confidence threshold for RDD
            algorithm_decision.update({
                'primary_algorithm': 'RDD',
                'confidence': rdd_confidence,
                'identification_strategy': 'discontinuity',
                'reasoning': rdd_reasoning,
                'required_assumptions': ['Sharp/fuzzy discontinuity', 'No manipulation of running variable', 'Continuity of potential outcomes'],
                'diagnostic_tests': rdd_diagnostics,
                'fallback_algorithms': ['OLS', 'G Computation']
            })
            return algorithm_decision
        else:
            # RDD available but questionable - keep as fallback
            algorithm_decision['fallback_algorithms'].append('RDD')
    
    # Tier 2B: Difference-in-Differences  
    if study_chars['has_time_variation'] and study_chars['has_group_structure']:
        did_confidence, did_reasoning, did_diagnostics = _evaluate_did_suitability(
            data, treatment, outcome, time_variable, group_variable, covariates, study_chars
        )
        
        if did_confidence >= 0.75:  # High confidence threshold for DiD
            algorithm_decision.update({
                'primary_algorithm': 'DiD', 
                'confidence': did_confidence,
                'identification_strategy': 'difference_in_differences',
                'reasoning': did_reasoning,
                'required_assumptions': ['Parallel trends', 'No anticipation effects', 'SUTVA', 'Common shocks'],
                'diagnostic_tests': did_diagnostics,
                'fallback_algorithms': ['G Computation', 'Propensity Score']
            })
            return algorithm_decision
        else:
            # DiD available but questionable - keep as fallback
            algorithm_decision['fallback_algorithms'].append('DiD')
    
    # TIER 3: INSTRUMENTAL VARIABLES (Addressing Unmeasured Confounding)
    if study_chars['instruments_available'] and len(study_chars['instruments_list']) > 0:
        iv_confidence, iv_reasoning, iv_diagnostics, recommended_iv_method = _evaluate_iv_suitability(
            data, treatment, outcome, study_chars['instruments_list'], covariates, study_chars
        )
        
        if iv_confidence >= 0.7:  # Reasonable confidence threshold for IV
            algorithm_decision.update({
                'primary_algorithm': recommended_iv_method,  # Could be IV or MR
                'confidence': iv_confidence,
                'identification_strategy': 'instrumental_variables',
                'reasoning': iv_reasoning,
                'required_assumptions': ['Instrument relevance', 'Instrument exogeneity', 'Exclusion restriction'],
                'diagnostic_tests': iv_diagnostics,
                'instruments': study_chars['instruments_list'],
                'fallback_algorithms': ['G Computation', 'DML'] if study_chars['sufficient_power'] else ['OLS']
            })
            return algorithm_decision
        else:
            # IV available but weak - keep as fallback
            algorithm_decision['fallback_algorithms'].append(recommended_iv_method)
    
    # TIER 4: FRONTDOOR CRITERION (Complete Mediation)  
    if study_chars['mediators_available'] and len(study_chars['mediators_list']) > 0:
        frontdoor_confidence, frontdoor_reasoning, frontdoor_diagnostics = _evaluate_frontdoor_suitability(
            data, treatment, outcome, study_chars['mediators_list'], covariates, study_chars
        )
        
        if frontdoor_confidence >= 0.6:  # Lower threshold as frontdoor is less common
            algorithm_decision.update({
                'primary_algorithm': 'Frontdoor Adjustment',
                'confidence': frontdoor_confidence,
                'identification_strategy': 'frontdoor_criterion',
                'reasoning': frontdoor_reasoning,
                'required_assumptions': ['Complete mediation', 'No direct effect', 'No confounding of mediator-outcome'],
                'diagnostic_tests': frontdoor_diagnostics,
                'mediators': study_chars['mediators_list'],
                'fallback_algorithms': ['G Computation', 'Propensity Score']
            })
            return algorithm_decision
        else:
            algorithm_decision['fallback_algorithms'].append('Frontdoor Adjustment')
    
    # TIER 5: BACKDOOR CRITERION (Confounding Control)
    # Modified to handle latent confounders appropriately
    if study_chars['valid_backdoor_set']:
        # If latent confounders are present, prefer methods that can handle them
        if study_chars['latent_confounders']:
            print(f"Latent confounders detected: {study_chars.get('dag_latent_confounders', [])}")
            
            # For latent confounders, prefer DML or G-Computation as they're more robust
            if study_chars['sufficient_power'] and study_chars['sample_size'] >= 500:
                backdoor_method = 'DML'
                backdoor_confidence = 0.75  # Reduced confidence due to latent confounders
                backdoor_reasoning = [
                    f"DML selected for robustness with {len(study_chars.get('dag_latent_confounders', []))} latent confounders",
                    "Double ML provides some robustness to unobserved confounding"
                ]
                backdoor_diagnostics = {}
            else:
                backdoor_method = 'G Computation'
                backdoor_confidence = 0.70  # Reduced confidence due to latent confounders
                backdoor_reasoning = [
                    f"G-Computation selected with {len(study_chars.get('dag_latent_confounders', []))} latent confounders",
                    "G-Computation can model some latent confounding patterns"
                ]
                backdoor_diagnostics = {}
            
            algorithm_decision.update({
                'primary_algorithm': backdoor_method,
                'confidence': backdoor_confidence,
                'identification_strategy': 'backdoor_with_latents',
                'reasoning': backdoor_reasoning,
                'required_assumptions': [
                    'Measured confounders sufficient for identification',
                    'Latent confounders do not invalidate identification',
                    'Correct DAG specification including latent structure'
                ],
                'diagnostic_tests': backdoor_diagnostics,
                'adjustment_set': study_chars['backdoor_set'],
                'latent_confounders': study_chars.get('dag_latent_confounders', []),
                'fallback_algorithms': ['G Computation', 'OLS'] if backdoor_method == 'DML' else ['DML', 'OLS'],
                'warning': f'Causal identification complicated by {len(study_chars.get("dag_latent_confounders", []))} latent confounders'
            })
            return algorithm_decision
            
        else:
            # No latent confounders - proceed with standard backdoor methods
            backdoor_confidence, backdoor_method, backdoor_reasoning, backdoor_diagnostics = _evaluate_backdoor_methods(
                data, treatment, outcome, study_chars['backdoor_set'], study_chars
            )
            
            algorithm_decision.update({
                'primary_algorithm': backdoor_method,
                'confidence': backdoor_confidence,
                'identification_strategy': 'backdoor_criterion',
                'reasoning': backdoor_reasoning,
                'required_assumptions': ['No unmeasured confounding', 'Correct DAG specification', 'Adequate covariate measurement'],
                'diagnostic_tests': backdoor_diagnostics,
                'adjustment_set': study_chars['backdoor_set'],
                'fallback_algorithms': _get_backdoor_fallbacks(backdoor_method, study_chars)
            })
            return algorithm_decision
    
    # TIER 6: FALLBACK METHODS (When Identification is Uncertain)
    fallback_confidence, fallback_method, fallback_reasoning = _select_fallback_method(
        data, treatment, outcome, covariates, study_chars
    )
    
    algorithm_decision.update({
        'primary_algorithm': fallback_method,
        'confidence': fallback_confidence,
        'identification_strategy': 'fallback_robust',
        'reasoning': fallback_reasoning,
        'required_assumptions': ['Depends on chosen method - generally strong'],
        'fallback_algorithms': ['OLS'],  # Always have OLS as final fallback
        'warning': 'Causal identification may be compromised - interpret results cautiously'
    })
    
    return algorithm_decision

def run_inference_algorithm(data, treatment, outcome, covariates, dag, sample_size,
                           assignment_style, latent_confounders, cutoff_value, 
                           time_variable, group_variable, running_variable, llm_results=None):
    """
    Execute causal inference with optimized algorithm selection based on theoretical decision tree.
    """
    
    # Get algorithm recommendation using optimized decision tree
    decision = recommend_inference_algorithm(
        data, treatment, outcome, covariates, dag, sample_size,
        assignment_style, latent_confounders, cutoff_value,
        time_variable, group_variable, llm_results
    )
    
    algorithm = decision["primary_algorithm"]
    confidence = decision["confidence"]
    reasoning = decision["reasoning"]
    identification_strategy = decision["identification_strategy"]
    required_assumptions = decision["required_assumptions"]
    fallback_algorithms = decision.get("fallback_algorithms", [])
    
    print(f"=== CAUSAL INFERENCE ALGORITHM SELECTION ===")
    print(f"Selected Algorithm: {algorithm}")
    print(f"Confidence Score: {confidence:.3f}")
    print(f"Identification Strategy: {identification_strategy}")
    
    print(f"\nSelection Reasoning:")
    for i, reason in enumerate(reasoning, 1):
        print(f"  {i}. {reason}")
    
    print(f"\nRequired Assumptions:")
    for i, assumption in enumerate(required_assumptions, 1):
        print(f"  {i}. {assumption}")
    
    if decision.get("warning"):
        print(f"\n⚠️  WARNING: {decision['warning']}")
    
    if fallback_algorithms:
        print(f"\nFallback Options: {', '.join(fallback_algorithms)}")
    
    # Display diagnostic results if available
    diagnostic_tests = decision.get("diagnostic_tests", {})
    if diagnostic_tests:
        print(f"\n=== DIAGNOSTIC TEST RESULTS ===")
        for test_name, test_result in diagnostic_tests.items():
            if test_name != 'error' and test_name != 'overall_valid':
                status = "✓ PASSED" if test_result else "✗ FAILED"
                print(f"  {test_name}: {status}")
    
    # Display study characteristics
    study_chars = decision["study_characteristics"]
    print(f"\n=== STUDY CHARACTERISTICS ===")
    print(f"Sample Size: {study_chars['sample_size']}")
    print(f"Number of Covariates: {study_chars['n_covariates']}")
    print(f"Assignment Style: {study_chars['assignment_style']}")
    print(f"Treatment Type: {study_chars['treatment_type']}")
    print(f"Outcome Type: {study_chars['outcome_type']}")
    print(f"High Dimensional: {study_chars['high_dimensional']}")
    print(f"Sufficient Power: {study_chars['sufficient_power']}")
    
    # Display latent confounder information if present
    if study_chars.get('dag_latent_confounders'):
        print(f"Latent Confounders Detected: {len(study_chars['dag_latent_confounders'])}")
        for i, (latent_node, var1, var2) in enumerate(study_chars.get('latent_pairs', []), 1):
            print(f"  {i}. {latent_node}: Unobserved confounder of {var1} ↔ {var2}")
    else:
        print(f"Latent Confounders: None detected")
    
    print(f"\n=== EXECUTING {algorithm} ===")
    
    # Extract algorithm-specific parameters
    instruments = decision.get("instruments", [])
    mediators = decision.get("mediators", [])
    adjustment_set = decision.get("adjustment_set", set())
    
    # Execute the selected algorithm with error handling and fallbacks
    try:
        estimate = _execute_inference_algorithm(
            algorithm, data, treatment, outcome, covariates, sample_size,
            cutoff_value, time_variable, group_variable, running_variable,
            list(adjustment_set), instruments, mediators, llm_results,
            latent_confounders=study_chars.get('dag_latent_confounders', [])
        )
        
        print(f"✓ {algorithm} completed successfully")
        
        # Add metadata to the result
        _add_inference_metadata_to_result(estimate, decision)
        
        return estimate
        
    except Exception as e:
        print(f"✗ {algorithm} failed: {str(e)}")
        
        # Try fallback algorithms
        for fallback_alg in fallback_algorithms:
            try:
                print(f"Trying fallback algorithm: {fallback_alg}")
                estimate = _execute_inference_algorithm(
                    fallback_alg, data, treatment, outcome, covariates, sample_size,
                    cutoff_value, time_variable, group_variable, running_variable,
                    list(adjustment_set), instruments, mediators, llm_results,
                    latent_confounders=study_chars.get('dag_latent_confounders', [])
                )
                print(f"✓ {fallback_alg} completed successfully")
                
                # Update decision info for the fallback
                decision['primary_algorithm'] = fallback_alg
                decision['confidence'] *= 0.8  # Reduce confidence since we used fallback
                decision['reasoning'].append(f"Fell back to {fallback_alg} due to {algorithm} failure")
                
                _add_inference_metadata_to_result(estimate, decision)
                return estimate
                
            except Exception as fallback_error:
                print(f"✗ {fallback_alg} also failed: {str(fallback_error)}")
                continue
        
        # If all algorithms fail, raise the original error
        raise RuntimeError(f"All algorithms failed. Primary: {e}, Fallbacks attempted: {len(fallback_algorithms)}")

def _execute_inference_algorithm(algorithm_name, data, treatment, outcome, covariates, 
                               sample_size, cutoff_value, time_variable, group_variable, 
                               running_variable, adjustment_set, instruments, mediators, llm_results,
                               latent_confounders=None):
    """Execute a specific causal inference algorithm with proper parameter handling."""
    
    # Handle common algorithm name variations
    algorithm_mapping = {
        'Mendelian Randomization': 'MR',
        'mendelian randomization': 'MR', 
        'Instrumental Variables': 'IV',
        'instrumental variables': 'IV',
        'Double Machine Learning': 'DML',
        'double machine learning': 'DML',
        'Ordinary Least Squares': 'OLS',
        'ordinary least squares': 'OLS',
        'Regression Discontinuity': 'RDD',
        'regression discontinuity': 'RDD',
        'Difference in Differences': 'DiD',
        'difference in differences': 'DiD',
        'Randomized Controlled Trial': 'RCT',
        'randomized controlled trial': 'RCT'
    }
    
    # Clean and map algorithm name
    algorithm_name = str(algorithm_name).strip()
    algorithm_name = algorithm_mapping.get(algorithm_name, algorithm_name)
    
    match algorithm_name:
        case 'RCT':
            return ols.estimate(data, treatment, outcome, adjustment_set)
            
        case 'RDD':
            return rdd.estimate(
                data, outcome, running_variable, cutoff_value, covariates
            )
            
        case 'DiD':
            return did.estimate(
                data, treatment, outcome, time_variable, 
                group_variable, covariates
            )
            
        case 'IV':
            return iv.estimate(
                data, treatment, outcome, instruments, covariates,
                latent_confounders=latent_confounders
            )
            
        case 'MR':
            return mr.estimate(
                data, treatment, outcome, covariates, 
                llm_results=llm_results,
                latent_confounders=latent_confounders
            )
            
        case 'Frontdoor Adjustment':
            return frontdoor_adjustment.estimate(
                data, treatment, mediators, adjustment_set, outcome,
                latent_confounders=latent_confounders
            )
            
        case 'DML':
            return double_machine_learning.estimate(
                data, treatment, outcome, adjustment_set, sample_size,
                latent_confounders=latent_confounders
            )
            
        case 'OLS':
            return ols.estimate(data, treatment, outcome, adjustment_set)
            
        case 'Propensity Score':
            return propensity_score.estimate(
                data, treatment, outcome, adjustment_set
            )
            
        case 'G Computation':
            return g_computation.estimate(
                data, treatment, outcome, adjustment_set,
                latent_confounders=latent_confounders
            )
            
        case _:
            raise ValueError(f"Unknown inference algorithm: {algorithm_name}")

def _add_inference_metadata_to_result(estimate, decision):
    """Add algorithm selection metadata to the resulting estimate."""
    
    if estimate is None:
        return
        
    metadata = {
        'selected_algorithm': decision['primary_algorithm'],
        'algorithm_confidence': decision['confidence'],
        'identification_strategy': decision['identification_strategy'],
        'selection_reasoning': decision['reasoning'],
        'required_assumptions': decision['required_assumptions'],
        'study_characteristics': decision['study_characteristics'],
        'diagnostic_tests': decision.get('diagnostic_tests', {}),
        'fallback_algorithms': decision.get('fallback_algorithms', [])
    }
    
    # Add metadata based on estimate type
    if isinstance(estimate, dict):
        estimate.update(metadata)
    elif hasattr(estimate, '__dict__'):
        # Object with attributes
        for key, value in metadata.items():
            setattr(estimate, key, value)
    
    return estimate