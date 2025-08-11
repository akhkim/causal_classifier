from dowhy import CausalModel
import numpy as np
import pandas as pd
import json
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ..llm_query import create_chat_completion

def _select_hyperparameters_llm(data, treatment, outcome, adjustment_set):
    """
    Use LLM to select optimal hyperparameters for G-computation based on data characteristics.
    """
    
    # Gather data characteristics
    n_samples = len(data)
    n_covariates = len(adjustment_set) if adjustment_set else 0
    
    # Check outcome type and distribution
    outcome_data = data[outcome].dropna()
    outcome_type = "binary" if outcome_data.nunique() <= 2 else "continuous"
    outcome_skewness = stats.skew(outcome_data) if outcome_type == "continuous" else 0
    
    # Check treatment type
    treatment_data = data[treatment].dropna()
    treatment_type = "binary" if treatment_data.nunique() <= 2 else "continuous"
    
    # Estimate model complexity needed
    complexity_score = n_covariates / n_samples if n_samples > 0 else 0
    
    context = f"""
    Dataset Characteristics for G-Computation:
    - Sample size: {n_samples}
    - Number of covariates: {n_covariates}
    - Complexity ratio (covariates/samples): {complexity_score:.3f}
    - Treatment type: {treatment_type}
    - Outcome type: {outcome_type}
    - Outcome skewness: {outcome_skewness:.3f}
    - Treatment variable: {treatment}
    - Outcome variable: {outcome}
    """
    
    try:
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": """You are an expert in G-computation (g-formula) hyperparameter selection.
                
                Based on the dataset characteristics, recommend optimal hyperparameters for:
                1. n_simulations: Number of Monte Carlo simulations for g-formula estimation
                2. method_params: Additional parameters for the g-formula method
                
                Guidelines:
                - Small samples (n<500): Use fewer simulations (500-1500) to avoid overfitting
                - Medium samples (500-2000): Use moderate simulations (1000-3000)
                - Large samples (>2000): Can use more simulations (2000-5000) for precision
                - High complexity (many covariates): May need more simulations for stable estimates
                - Binary outcomes: Generally need fewer simulations than continuous
                - Skewed outcomes: May benefit from more simulations for robust estimation
                - Simple models (few covariates): Can use fewer simulations
                
                Return your response as valid JSON with this exact structure:
                {
                    "n_simulations": <int>,
                    "method_name": "gformula",
                    "reasoning": "<brief explanation of choices>"
                }"""},
                {"role": "user", "content": f"Given these G-computation dataset characteristics, what are the optimal hyperparameters?\n\n{context}"}
            ],
            temperature=0.1,
            thinking=False
        )
        
        hyperparams = json.loads(response)
        return hyperparams
        
    except Exception as e:
        print(f"LLM hyperparameter selection failed: {e}")
        # Fallback to rule-based selection
        return _select_hyperparameters_fallback(n_samples, n_covariates, outcome_type, complexity_score)

def _select_hyperparameters_fallback(n_samples, n_covariates, outcome_type, complexity_score):
    """Fallback rule-based hyperparameter selection."""
    
    # Number of simulations
    if n_samples < 500:
        n_simulations = 1000
    elif n_samples < 2000:
        n_simulations = 2000
    else:
        n_simulations = 3000
    
    # Adjust for complexity
    if complexity_score > 0.1:  # High complexity
        n_simulations = min(n_simulations * 1.5, 5000)
    
    # Adjust for outcome type
    if outcome_type == "binary":
        n_simulations = int(n_simulations * 0.8)  # Binary outcomes typically need fewer
    
    return {
        "n_simulations": int(n_simulations),
        "method_name": "gformula",
        "reasoning": "Fallback rule-based selection based on sample size and complexity"
    }

def estimate(
    data,
    treatment,
    outcome,
    adjustment_set,
    latent_confounders=None
):
    """
    G-computation estimation with LLM-optimized hyperparameters.
    Now supports both measured confounders and latent confounders from FCI.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset
    treatment : str
        Treatment variable name
    outcome : str
        Outcome variable name
    adjustment_set : list
        List of measured confounders to adjust for
    latent_confounders : list, optional
        List of latent confounder nodes (U_ format) identified by FCI
    """
    
    # Handle latent confounders information
    if latent_confounders:
        print(f"G-Computation: Handling {len(latent_confounders)} latent confounders: {latent_confounders}")
        # For G-computation, latent confounders affect our confidence in identification
        # but we still proceed with measured confounders
        print("Note: G-computation proceeds with measured confounders; latent confounders noted for interpretation")
    
    # Get optimal hyperparameters using LLM
    hyperparams = _select_hyperparameters_llm(data, treatment, outcome, adjustment_set)
    print(f"G-Computation Hyperparameters selected: {hyperparams.get('reasoning', 'No reasoning provided')}")
    
    # Build DoWhy causal model with measured confounders
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=adjustment_set
    )
    identified_estimand = model.identify_effect()
    
    # Run G-computation with optimized hyperparameters
    est = model.estimate_effect(
        identified_estimand,
        method_name=hyperparams["method_name"],
        method_params={"n_simulations": hyperparams["n_simulations"]}
    )
    
    # Adjust confidence based on latent confounders
    confidence_adjustment = 1.0
    if latent_confounders:
        # Reduce confidence based on number of latent confounders
        confidence_adjustment = max(0.5, 1.0 - 0.1 * len(latent_confounders))
        print(f"Confidence adjusted by factor {confidence_adjustment:.2f} due to latent confounders")
    
    return {
        'estimate': est.value,
        'std_error': est.get_std_error(),
        'model': est,
        'hyperparameters': hyperparams,
        'latent_confounders': latent_confounders or [],
        'confidence_adjustment': confidence_adjustment,
        'interpretation': _generate_interpretation(est.value, latent_confounders, adjustment_set)
    }

def _generate_interpretation(estimate, latent_confounders, adjustment_set):
    """Generate interpretation text considering both measured and latent confounders."""
    
    interpretation = f"Estimated causal effect: {estimate:.4f}\n"
    interpretation += f"Adjusted for {len(adjustment_set)} measured confounders: {adjustment_set}\n"
    
    if latent_confounders:
        interpretation += f"\nIMPORTANT: {len(latent_confounders)} latent confounders detected by FCI:\n"
        for latent in latent_confounders:
            interpretation += f"  - {latent}: Unobserved confounder affecting the causal relationship\n"
        interpretation += "\nThis estimate may be biased due to unobserved confounding. "
        interpretation += "Consider instrumental variables or other methods for unbiased estimation."
    else:
        interpretation += "\nNo latent confounders detected. Estimate assumes no unmeasured confounding."
    
    return interpretation

def diagnose(data, treatment, outcome, adjustment_set, alpha=0.05):
    """
    Comprehensive G-computation diagnostic tests.
    
    Returns:
    --------
    dict : Dictionary with diagnostic test results
    """
    df = data.copy()
    
    diagnostics = {
        'model_fit': True,
        'linearity': True,
        'no_unmeasured_confounding': True,
        'positivity': True,
        'consistency': True,
        'overall_valid': True
    }
    
    try:
        # Prepare data
        covariates = adjustment_set + [treatment]
        X = df[covariates].values
        Y = df[outcome].values
        T = df[treatment].values
        
        # 1. Test model fit quality
        # Cross-validation score for outcome model
        model = LinearRegression()
        cv_scores = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
        model_fit_score = -cv_scores.mean()
        
        # Compare to baseline (mean only)
        baseline_mse = np.var(Y)
        diagnostics['model_fit'] = model_fit_score < baseline_mse * 0.8  # Should improve over baseline
        
        # 2. Test linearity (for linear G-computation)
        # Fit full model and check residuals
        model.fit(X, Y)
        predictions = model.predict(X)
        residuals = Y - predictions
        
        # Shapiro-Wilk test for normality of residuals (proxy for linearity)
        if len(residuals) >= 3:
            _, p_value = stats.shapiro(residuals[:5000])  # Limit sample size for computational efficiency
            diagnostics['linearity'] = p_value > alpha
        
        # 3. Test for positivity (all treatment combinations should be observed)
        # Check if we have observations for different treatment levels
        treatment_levels = df[treatment].unique()
        if len(treatment_levels) >= 2:
            # For each covariate combination, check if we have both treatment levels
            # Simplified: check overall balance
            min_treatment_prop = min(np.mean(T), 1 - np.mean(T))
            diagnostics['positivity'] = min_treatment_prop > 0.05  # At least 5% in each group
        
        # 4. Indirect test for unmeasured confounding
        # Check if treatment assignment appears random given observed covariates
        X_adj = df[adjustment_set].values if adjustment_set else np.ones((len(df), 1))
        
        if X_adj.shape[1] > 0:
            # Fit treatment model
            treatment_model = LinearRegression()
            treatment_model.fit(X_adj, T)
            treatment_pred = treatment_model.predict(X_adj)
            
            # If treatment is well-predicted by observed covariates, 
            # it suggests systematic assignment (potential confounding)
            treatment_r2 = 1 - np.var(T - treatment_pred) / np.var(T)
            diagnostics['no_unmeasured_confounding'] = treatment_r2 < 0.5  # R² should be moderate
        
        # 5. Consistency check (same treatment should give same outcome for same units)
        # This is mainly conceptual but we can check for outliers
        # Look for extreme residuals that might indicate inconsistency
        if len(residuals) > 0:
            residual_z_scores = np.abs(stats.zscore(residuals))
            outlier_proportion = np.mean(residual_z_scores > 3)
            diagnostics['consistency'] = outlier_proportion < 0.05  # Less than 5% extreme outliers
        
        # Overall validity
        diagnostics['overall_valid'] = all([
            diagnostics['model_fit'],
            diagnostics['linearity'],
            diagnostics['no_unmeasured_confounding'],
            diagnostics['positivity'],
            diagnostics['consistency']
        ])
        
    except Exception as e:
        print(f"G-computation diagnostic failed: {e}")
        diagnostics['overall_valid'] = False
    
    return diagnostics


# import numpy as np
# from sklearn.base import clone
# from sklearn.linear_model import LinearRegression

# def estimate(
#     data,
#     treatment,
#     outcome,
#     adjustment_set,
#     effect='ate'
# ):
#     df = data.copy()
#     outcome_model = LinearRegression()

#     # Prepare feature matrix
#     features = adjustment_set + [treatment]
#     X = df[features]
#     Y = df[outcome]

#     # Fit the outcome regression model
#     model = clone(outcome_model).fit(X, Y)

#     # Create data for potential outcomes
#     df1 = df.copy()
#     df1[treatment] = 1
#     X1 = df1[features]

#     df0 = df.copy()
#     df0[treatment] = 0
#     X0 = df0[features]

#     # Predict potential outcomes
#     Y1_pred = model.predict(X1)
#     Y0_pred = model.predict(X0)

#     # Individual treatment effects
#     ITE = Y1_pred - Y0_pred

#     # Compute effect estimate
#     if effect.lower() == 'ate':
#         estimate = np.mean(ITE)
#     elif effect.lower() == 'att':
#         mask = df[treatment] == 1
#         estimate = np.mean(ITE[mask])

#     return {
#         'estimate': estimate,
#         'std_error': None,
#         'model': model
#     }

