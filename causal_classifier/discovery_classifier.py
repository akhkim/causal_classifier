from collections import Counter
import numpy as np
import pandas as pd
from fitter import Fitter
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from . import discovery_algorithms

def run_discovery_algorithm(df, latent_confounders, variable_analysis=None, use_gpu=False):
    """
    Execute causal discovery with optimized algorithm selection and optional GPU acceleration.
    
    Args:
        df: Input dataframe
        latent_confounders: Whether latent confounders are expected
        use_gpu: Whether to attempt GPU acceleration for supported algorithms
    """
    decision = recommend_discovery_algorithm(df, latent_confounders)
    
    algorithm = decision["primary_algorithm"]
    confidence = decision["confidence"]
    reasoning = decision["reasoning"]
    fallback_algorithms = decision.get("fallback_algorithms", [])
    data_chars = decision["data_characteristics"]
    
    print(f"=== CAUSAL DISCOVERY ALGORITHM SELECTION ===")
    print(f"Selected Algorithm: {algorithm}")
    print(f"Confidence Score: {confidence:.3f}")
    if use_gpu:
        print(f"GPU Acceleration: Requested")
    print(f"\nSelection Reasoning:")
    for i, reason in enumerate(reasoning, 1):
        print(f"  {i}. {reason}")
    
    if fallback_algorithms:
        print(f"\nFallback Options: {', '.join(fallback_algorithms)}")
    
    print(f"\n=== DATA CHARACTERISTICS ===")
    print(f"Samples (n): {data_chars['n_samples']}")
    print(f"Features (p): {data_chars['n_features']}")
    print(f"Sample-to-Feature Ratio: {data_chars['sample_to_feature_ratio']:.2f}")
    
    # Data type summary
    type_counts = Counter(data_chars['data_types'].values())
    print(f"Data Types: {dict(type_counts)}")
    
    if data_chars['dominant_distribution'] != 'no_continuous_vars':
        print(f"Dominant Distribution: {data_chars['dominant_distribution']}")
        print(f"Distribution Agreement: {data_chars['distribution_agreement']:.3f}")
    
    linearity = data_chars['linearity_assessment'].get('is_predominantly_linear', 'Unknown')
    print(f"Predominantly Linear: {linearity}")
    print(f"Estimated Sparsity: {data_chars['sparsity_level']:.3f}")
    print(f"Mixed Data Types: {data_chars['mixed_data']}")
    
    print(f"\n=== EXECUTING {algorithm} ===")
    
    # Execute the selected algorithm with error handling and fallbacks
    try:
        dag = _execute_algorithm(algorithm, df, use_gpu)
        print(f"✓ {algorithm} completed successfully")
        
        # Add metadata to the result
        _add_metadata_to_dag(dag, decision)
        
        nx_graph = dag
        
    except Exception as e:
        print(f"✗ {algorithm} failed: {str(e)}")
        
        # Try fallback algorithms
        for fallback_alg in fallback_algorithms:
            try:
                print(f"Trying fallback algorithm: {fallback_alg}")
                dag = _execute_algorithm(fallback_alg, df, use_gpu)
                print(f"✓ {fallback_alg} completed successfully")
                
                # Update decision info for the fallback
                decision['primary_algorithm'] = fallback_alg
                decision['confidence'] *= 0.8  # Reduce confidence since we used fallback
                decision['reasoning'].append(f"Fell back to {fallback_alg} due to {algorithm} failure")
                
                _add_metadata_to_dag(dag, decision)
                nx_graph = dag

            except Exception as fallback_error:
                print(f"✗ {fallback_alg} also failed: {str(fallback_error)}")
                continue
        
        # If all algorithms fail, raise the original error
        raise RuntimeError(f"All algorithms failed. Primary: {e}, Fallbacks: {[str(fe) for fe in fallback_algorithms]}")
    
    # Add metadata based on variable analysis if provided
    if variable_analysis:
        for node in nx_graph.nodes():
            if str(node) in variable_analysis:
                analysis = variable_analysis[str(node)]
                nx_graph.nodes[node]['variable_type'] = analysis['type']
                nx_graph.nodes[node]['potential_roles'] = analysis['potential_roles']
                nx_graph.nodes[node]['unique_ratio'] = analysis['unique_ratio']
        
        # Suggest likely treatment-outcome pairs based on graph structure
        suggestions = suggest_causal_pairs_from_graph(nx_graph, variable_analysis)
        return nx_graph, suggestions
    
    return nx_graph, []

def suggest_causal_pairs_from_graph(graph, variable_analysis):
    """Suggest likely treatment-outcome pairs based on graph structure"""
    suggestions = []
    
    # Check if graph is directed
    is_directed = hasattr(graph, 'out_degree') and hasattr(graph, 'in_degree')
    
    # Look for binary variables with high out-degree (potential treatments)
    potential_treatments = []
    for node in graph.nodes():
        if str(node) in variable_analysis:
            analysis = variable_analysis[str(node)]
            if 'binary_treatment' in analysis['potential_roles'] or 'binary_indicator' in analysis['potential_roles']:
                if is_directed:
                    out_degree = graph.out_degree(node)
                else:
                    # For undirected graphs, use total degree
                    out_degree = graph.degree(node)
                potential_treatments.append((str(node), out_degree))
    
    # Look for variables with high in-degree (potential outcomes)
    potential_outcomes = []
    for node in graph.nodes():
        if is_directed:
            node_degree = graph.in_degree(node)
        else:
            # For undirected graphs, use total degree
            node_degree = graph.degree(node)
            
        if node_degree > 1:  # Connected to multiple variables
            node_str = str(node)
            if node_str in variable_analysis:
                analysis = variable_analysis[node_str]
                # Boost confidence for income-like variables
                confidence_boost = 0.3 if 'income_variable' in analysis['potential_roles'] else 0.0
                potential_outcomes.append((node_str, node_degree, confidence_boost))
    
    # Create suggestions
    for treatment, out_deg in sorted(potential_treatments, key=lambda x: x[1], reverse=True)[:3]:
        for outcome, in_deg, boost in sorted(potential_outcomes, key=lambda x: x[1], reverse=True)[:3]:
            if treatment != outcome:
                # Calculate confidence based on graph structure and variable analysis
                confidence = min(0.9, (out_deg + in_deg) / 10 + boost)
                
                degree_type = "outgoing/incoming edges" if is_directed else "connections"
                suggestions.append({
                    'treatment': treatment,
                    'outcome': outcome,
                    'confidence': min(confidence, 0.95),
                    'reasoning': f'Graph structure: {treatment} has {out_deg} {degree_type}, {outcome} has {in_deg} {degree_type}'
                })
    
    return suggestions[:5]  # Top 5 suggestions

def detect_data_characteristics(df):
    """
    Enhanced data characterization for optimal algorithm selection.
    Returns comprehensive analysis of data properties relevant for causal discovery.
    """
    if df.shape[0] > 1000:
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df
    
    n, p = df.shape
    characteristics = {
        'n_samples': n,
        'n_features': p,
        'sample_to_feature_ratio': n / p if p > 0 else 0,
        'data_types': {},
        'distributions': {},
        'linearity_assessment': {},
        'sparsity_level': None,
        'mixed_data': False
    }
    
    # Categorize variables by data type
    continuous_vars = []
    discrete_vars = []
    binary_vars = []
    categorical_vars = []
    
    for col in df_sample.columns:
        if pd.api.types.is_numeric_dtype(df_sample[col]):
            unique_vals = df_sample[col].nunique()
            if unique_vals == 2:
                binary_vars.append(col)
                characteristics['data_types'][col] = 'binary'
            elif unique_vals <= 10 and df_sample[col].dtype == 'int64':
                discrete_vars.append(col)
                characteristics['data_types'][col] = 'discrete'
            else:
                continuous_vars.append(col)
                characteristics['data_types'][col] = 'continuous'
        else:
            categorical_vars.append(col)
            characteristics['data_types'][col] = 'categorical'
    
    # Determine if we have mixed data types
    type_counts = len([x for x in [continuous_vars, discrete_vars, binary_vars, categorical_vars] if x])
    characteristics['mixed_data'] = type_counts > 1
    
    # Distribution analysis for continuous variables
    distribution_results = []
    for col in continuous_vars:
        data = df_sample[col].dropna().values
        if len(data) > 10:  # Need sufficient data for distribution fitting
            # Test for normality first (most important for algorithm selection)
            _, normality_p = stats.shapiro(data) if len(data) <= 5000 else stats.jarque_bera(data)
            is_normal = normality_p > 0.05
            
            if is_normal:
                characteristics['distributions'][col] = 'normal'
                distribution_results.append('normal')
            else:
                # Check for common non-normal distributions
                try:
                    f = Fitter(data, distributions=["norm", "expon", "laplace", "gamma", "beta"], timeout=20)
                    f.fit()
                    best_dist = f.get_best(method="sumsquare_error")
                    if best_dist:
                        best_name = list(best_dist.keys())[0]
                        characteristics['distributions'][col] = best_name
                        distribution_results.append(best_name)
                    else:
                        characteristics['distributions'][col] = 'unknown_non_normal'
                        distribution_results.append('unknown_non_normal')
                except:
                    characteristics['distributions'][col] = 'unknown_non_normal'
                    distribution_results.append('unknown_non_normal')
        else:
            characteristics['distributions'][col] = 'insufficient_data'
            distribution_results.append('insufficient_data')
    
    # Overall distribution assessment
    if distribution_results:
        dist_counter = Counter(distribution_results)
        dominant_dist = dist_counter.most_common(1)[0][0]
        dist_agreement = dist_counter.most_common(1)[0][1] / len(distribution_results)
        characteristics['dominant_distribution'] = dominant_dist
        characteristics['distribution_agreement'] = dist_agreement
    else:
        characteristics['dominant_distribution'] = 'no_continuous_vars'
        characteristics['distribution_agreement'] = 1.0
    
    # Assess linearity (important for linear methods vs non-linear)
    if len(continuous_vars) >= 2:
        # Sample pairs of continuous variables to assess linearity
        linear_relationships = []
        for i, var1 in enumerate(continuous_vars[:min(5, len(continuous_vars))]):
            for var2 in continuous_vars[i+1:min(5, len(continuous_vars))]:
                try:
                    data1 = df_sample[var1].dropna().values
                    data2 = df_sample[var2].dropna().values
                    if len(data1) > 10 and len(data2) > 10:
                        # Use mutual information to assess non-linear relationships
                        # Higher MI relative to linear correlation suggests non-linearity
                        valid_idx = ~(np.isnan(data1) | np.isnan(data2))
                        if np.sum(valid_idx) > 10:
                            data1_clean = data1[valid_idx].reshape(-1, 1)
                            data2_clean = data2[valid_idx]
                            
                            linear_corr = abs(np.corrcoef(data1_clean.flatten(), data2_clean)[0, 1])
                            mi_score = mutual_info_regression(data1_clean, data2_clean, random_state=42)[0]
                            
                            # Normalize MI score by entropy for comparison
                            if mi_score > 0 and linear_corr > 0:
                                linearity_ratio = linear_corr / (mi_score + 1e-6)  # Higher ratio suggests more linear
                                linear_relationships.append(linearity_ratio)
                except Exception:
                    continue
        
        if linear_relationships:
            avg_linearity = np.mean(linear_relationships)
            characteristics['linearity_assessment']['avg_linearity_score'] = avg_linearity
            characteristics['linearity_assessment']['is_predominantly_linear'] = avg_linearity > 0.5
        else:
            characteristics['linearity_assessment']['is_predominantly_linear'] = True  # Default assumption
    else:
        characteristics['linearity_assessment']['is_predominantly_linear'] = True
    
    # Assess sparsity (important for NOTEARS and similar methods)
    if continuous_vars:
        # Estimate sparsity by looking at correlation structure
        corr_matrix = df_sample[continuous_vars].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)  # Remove diagonal
        strong_correlations = (corr_matrix > 0.3).sum().sum()
        total_possible = len(continuous_vars) * (len(continuous_vars) - 1)
        if total_possible > 0:
            sparsity_ratio = 1 - (strong_correlations / total_possible)
            characteristics['sparsity_level'] = sparsity_ratio
        else:
            characteristics['sparsity_level'] = 0.5
    else:
        characteristics['sparsity_level'] = 0.5
    
    return characteristics

def recommend_discovery_algorithm(df, latent_confounders=False):
    """
    Optimized algorithm selection based on theoretical foundations and empirical best practices.
    
    Decision tree follows this hierarchy:
    1. Latent confounders → FCI (only option)
    2. Data type considerations → discrete vs continuous vs mixed
    3. Sample size and dimensionality → constraint-based vs score-based vs optimization-based
    4. Distribution assumptions → Gaussian vs non-Gaussian
    5. Linearity assumptions → linear vs non-linear methods
    6. Computational constraints → scalability considerations
    """
    
    # Get comprehensive data characteristics
    data_chars = detect_data_characteristics(df)
    n, p = data_chars['n_samples'], data_chars['n_features']
    
    # Initialize decision variables
    algorithm_decision = {
        'primary_algorithm': None,
        'confidence': 0.0,
        'reasoning': [],
        'fallback_algorithms': [],
        'data_characteristics': data_chars
    }
    
    # DECISION LEVEL 1: Latent Confounders (Hard Constraint)
    if latent_confounders:
        algorithm_decision['primary_algorithm'] = 'FCI'
        algorithm_decision['confidence'] = 0.95
        algorithm_decision['reasoning'].append("Latent confounders present - FCI is the only suitable algorithm")
        algorithm_decision['fallback_algorithms'] = []  # No alternatives when latents are assumed
        return algorithm_decision
    
    # DECISION LEVEL 2: Data Type Analysis
    has_continuous = any(t in ['continuous'] for t in data_chars['data_types'].values())
    has_discrete = any(t in ['discrete', 'binary'] for t in data_chars['data_types'].values())
    has_categorical = any(t == 'categorical' for t in data_chars['data_types'].values())
    
    # DECISION LEVEL 3: Sample Size and Dimensionality Assessment
    sample_adequacy = _assess_sample_adequacy(n, p)
    
    # DECISION LEVEL 4: Distribution and Linearity Assessment
    is_gaussian = data_chars.get('dominant_distribution') == 'normal' and data_chars.get('distribution_agreement', 0) > 0.7
    is_non_gaussian = (data_chars.get('dominant_distribution') not in ['normal', 'no_continuous_vars'] and 
                      data_chars.get('distribution_agreement', 0) > 0.6)
    is_linear = data_chars.get('linearity_assessment', {}).get('is_predominantly_linear', True)
    is_sparse = data_chars.get('sparsity_level', 0.5) > 0.7
    
    # MAIN DECISION TREE
    
    # Branch 1: Purely Discrete/Categorical Data
    if not has_continuous and (has_discrete or has_categorical):
        if sample_adequacy['constraint_based_suitable']:
            algorithm_decision['primary_algorithm'] = 'PC'
            algorithm_decision['confidence'] = 0.85
            algorithm_decision['reasoning'].append("Discrete data with adequate samples for constraint-based method")
            algorithm_decision['fallback_algorithms'] = ['GES']
        else:
            algorithm_decision['primary_algorithm'] = 'GES'
            algorithm_decision['confidence'] = 0.80
            algorithm_decision['reasoning'].append("Discrete data with limited samples - score-based method preferred")
            algorithm_decision['fallback_algorithms'] = ['PC']
    
    # Branch 2: Purely Continuous Data
    elif has_continuous and not has_discrete and not has_categorical:
        
        # Sub-branch 2a: Gaussian Continuous Data
        if is_gaussian:
            if sample_adequacy['constraint_based_suitable'] and is_linear:
                algorithm_decision['primary_algorithm'] = 'PC'
                algorithm_decision['confidence'] = 0.90
                algorithm_decision['reasoning'].append("Gaussian linear data with adequate samples - PC is optimal")
                algorithm_decision['fallback_algorithms'] = ['GES', 'NOTEARS']
            elif sample_adequacy['score_based_suitable']:
                algorithm_decision['primary_algorithm'] = 'GES'
                algorithm_decision['confidence'] = 0.85
                algorithm_decision['reasoning'].append("Gaussian data with moderate samples - GES preferred")
                algorithm_decision['fallback_algorithms'] = ['NOTEARS', 'PC']
            else:
                algorithm_decision['primary_algorithm'] = 'NOTEARS'
                algorithm_decision['confidence'] = 0.80
                algorithm_decision['reasoning'].append("Gaussian data with small samples - NOTEARS handles small n well")
                algorithm_decision['fallback_algorithms'] = ['GES']
        
        # Sub-branch 2b: Non-Gaussian Continuous Data
        elif is_non_gaussian:
            if sample_adequacy['lingam_suitable'] and is_linear:
                algorithm_decision['primary_algorithm'] = 'LiNGAM'
                algorithm_decision['confidence'] = 0.90
                algorithm_decision['reasoning'].append("Non-Gaussian linear data with adequate samples - LiNGAM is optimal")
                algorithm_decision['fallback_algorithms'] = ['NOTEARS', 'GES']
            elif sample_adequacy['score_based_suitable'] and is_sparse:
                algorithm_decision['primary_algorithm'] = 'NOTEARS'
                algorithm_decision['confidence'] = 0.85
                algorithm_decision['reasoning'].append("Non-Gaussian sparse data - NOTEARS handles sparsity well")
                algorithm_decision['fallback_algorithms'] = ['GES', 'LiNGAM']
            else:
                algorithm_decision['primary_algorithm'] = 'GES'
                algorithm_decision['confidence'] = 0.75
                algorithm_decision['reasoning'].append("Non-Gaussian data - GES is distribution-agnostic")
                algorithm_decision['fallback_algorithms'] = ['NOTEARS']
        
        # Sub-branch 2c: Unknown/Mixed Distribution Continuous Data
        else:
            if sample_adequacy['optimization_suitable'] and is_sparse:
                algorithm_decision['primary_algorithm'] = 'NOTEARS'
                algorithm_decision['confidence'] = 0.75
                algorithm_decision['reasoning'].append("Unknown distribution with sparsity - NOTEARS is robust")
                algorithm_decision['fallback_algorithms'] = ['GES']
            elif sample_adequacy['score_based_suitable']:
                algorithm_decision['primary_algorithm'] = 'GES'
                algorithm_decision['confidence'] = 0.70
                algorithm_decision['reasoning'].append("Unknown distribution - GES is most robust score-based method")
                algorithm_decision['fallback_algorithms'] = ['NOTEARS']
            else:
                algorithm_decision['primary_algorithm'] = 'GES'
                algorithm_decision['confidence'] = 0.65
                algorithm_decision['reasoning'].append("Small sample with unknown distribution - GES is most general")
                algorithm_decision['fallback_algorithms'] = ['NOTEARS']
    
    # Branch 3: Mixed Data Types
    else:
        if sample_adequacy['constraint_based_suitable']:
            algorithm_decision['primary_algorithm'] = 'PC'
            algorithm_decision['confidence'] = 0.80
            algorithm_decision['reasoning'].append("Mixed data types with adequate samples - PC handles heterogeneity well")
            algorithm_decision['fallback_algorithms'] = ['GES']
        elif sample_adequacy['score_based_suitable']:
            algorithm_decision['primary_algorithm'] = 'GES'
            algorithm_decision['confidence'] = 0.75
            algorithm_decision['reasoning'].append("Mixed data types with moderate samples - GES is flexible")
            algorithm_decision['fallback_algorithms'] = ['NOTEARS']
        else:
            algorithm_decision['primary_algorithm'] = 'GES'
            algorithm_decision['confidence'] = 0.65
            algorithm_decision['reasoning'].append("Mixed data types with small samples - GES as best general method")
            algorithm_decision['fallback_algorithms'] = ['NOTEARS']
    
    # DECISION LEVEL 5: Computational Constraints and Final Adjustments
    if p > 50 and algorithm_decision['primary_algorithm'] == 'PC':
        # High-dimensional data - switch to more scalable methods
        algorithm_decision['reasoning'].append(f"High dimensionality (p={p}) - switching from constraint-based to score-based")
        algorithm_decision['fallback_algorithms'].insert(0, 'PC')  # PC becomes fallback
        algorithm_decision['primary_algorithm'] = 'GES'
        algorithm_decision['confidence'] *= 0.9
    
    # Ensure confidence is within bounds
    algorithm_decision['confidence'] = min(algorithm_decision['confidence'], 1.0)
    
    return algorithm_decision

def _assess_sample_adequacy(n, p):
    """
    Assess sample size adequacy for different algorithm families.
    Based on theoretical requirements and empirical studies.
    """
    return {
        'constraint_based_suitable': n >= max(200, 5 * p * np.log(p)),  # PC, FCI need substantial samples
        'score_based_suitable': n >= max(50, 3 * p),  # GES needs moderate samples  
        'optimization_suitable': n >= max(30, 2 * p),  # NOTEARS can work with smaller samples
        'lingam_suitable': n >= max(100, 4 * p),  # LiNGAM needs reasonable samples for non-Gaussianity
        'sample_to_feature_ratio': n / p if p > 0 else float('inf'),
        'is_high_dimensional': p > n/5,  # Rule of thumb for high-dimensionality
        'is_very_high_dimensional': p > n/2
    }

def _execute_algorithm(algorithm_name, df, use_gpu=False):
    """Execute a specific causal discovery algorithm with optional GPU acceleration."""
    
    # Determine fast mode based on dataset size
    n_samples, n_features = df.shape
    fast_mode = (n_features > 15) or (n_samples * n_features > 50000)
    
    match algorithm_name:
        case "FCI":
            return discovery_algorithms.fci.run(df, use_gpu=use_gpu, fast_mode=fast_mode)
        case "PC":
            return discovery_algorithms.pc.run(df, use_gpu=use_gpu, fast_mode=fast_mode)
        case "GES":
            # GES doesn't benefit much from GPU acceleration, so no GPU parameter
            return discovery_algorithms.ges.run(df, score_func="local_score_BIC")
        case "LiNGAM":
            return discovery_algorithms.lingam.run(df, use_gpu=use_gpu, fast_mode=fast_mode)
        case "NOTEARS":
            return discovery_algorithms.notears.run(df, use_gpu=use_gpu, fast_mode=fast_mode)
        case _:
            raise ValueError(f"Unknown discovery algorithm: {algorithm_name}")

def _add_metadata_to_dag(dag, decision):
    """Add algorithm selection metadata to the resulting DAG."""
    metadata = {
        'selected_algorithm': decision['primary_algorithm'],
        'algorithm_confidence': decision['confidence'],
        'selection_reasoning': decision['reasoning'],
        'data_characteristics': decision['data_characteristics'],
        'fallback_algorithms': decision.get('fallback_algorithms', [])
    }
    
    # Add metadata based on the DAG type
    if hasattr(dag, 'graph') and hasattr(dag.graph, 'update'):
        # NetworkX graph
        dag.graph.update(metadata)
    elif hasattr(dag, '__dict__'):
        # Object with attributes
        for key, value in metadata.items():
            setattr(dag, key, value)
    elif isinstance(dag, dict):
        # Dictionary-based DAG
        dag.update(metadata)
    
    return dag