import numpy as np
import pandas as pd
import json
from causalml.propensity import ElasticNetPropensityModel
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from ..llm_query import create_chat_completion

def _select_hyperparameters_llm(data, treatment, outcome, covariates):
    """
    Use LLM to select optimal hyperparameters for Propensity Score methods.
    """
    
    # Gather data characteristics
    n_samples = len(data)
    n_features = len(covariates) if covariates else 0
    
    # Treatment balance
    treatment_balance = data[treatment].mean() if treatment in data.columns else 0.5
    treatment_imbalance = abs(treatment_balance - 0.5)
    
    # Feature correlations (measure of confounding complexity)
    if covariates and len(covariates) > 1:
        try:
            covar_corrs = data[covariates].corr().abs()
            avg_correlation = covar_corrs.values[np.triu_indices_from(covar_corrs.values, k=1)].mean()
        except:
            avg_correlation = 0.3
    else:
        avg_correlation = 0.3
    
    context = f"""
    Dataset Characteristics:
    - Sample size: {n_samples}
    - Number of covariates: {n_features}
    - Treatment balance: {treatment_balance:.3f} (imbalance: {treatment_imbalance:.3f})
    - Average covariate correlation: {avg_correlation:.3f}
    - Treatment variable: {treatment}
    - Outcome variable: {outcome}
    """
    
    try:
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": """You are an expert in Propensity Score method hyperparameter selection.
                
                Based on the dataset characteristics, recommend optimal hyperparameters for:
                1. Number of cross-validation folds (n_fold): For propensity score model selection
                2. Clipping bounds (clip): Extreme propensity score values to clip - prevents division by very small numbers
                3. Random state for reproducibility
                
                Guidelines:
                - Small samples (n<200): Use fewer folds (3-5), wider clipping bounds (1e-4, 1-1e-4)
                - Large samples (>1000): Can use more folds (5-10), tighter clipping bounds (1e-3, 1-1e-3)
                - High treatment imbalance (>0.4): Use wider clipping bounds to handle extreme scores
                - High covariate correlation (>0.6): May need more regularization, fewer folds
                - Low correlation (<0.3): Can use more folds, standard clipping
                
                Return your response as valid JSON with this exact structure:
                {
                    "n_fold": <int>,
                    "clip_lower": <float>,
                    "clip_upper": <float>,
                    "random_state": 42,
                    "reasoning": "<brief explanation of choices>"
                }"""},
                {"role": "user", "content": f"Given these dataset characteristics, what are the optimal Propensity Score hyperparameters?\n\n{context}"}
            ],
            temperature=0.1,
            thinking=False
        )
        
        hyperparams = json.loads(response)
        return hyperparams
        
    except Exception as e:
        print(f"LLM hyperparameter selection failed: {e}")
        # Fallback to rule-based selection
        return _select_hyperparameters_fallback(n_samples, treatment_imbalance, avg_correlation)

def _select_hyperparameters_fallback(n_samples, treatment_imbalance, avg_correlation):
    """Fallback rule-based hyperparameter selection."""
    
    # Cross-validation folds
    if n_samples < 200:
        n_fold = 3
    elif n_samples < 1000:
        n_fold = 5
    else:
        n_fold = 10
    
    # Clipping bounds based on sample size and treatment imbalance
    if n_samples < 200 or treatment_imbalance > 0.4:
        clip_lower, clip_upper = 1e-4, 1-1e-4
    else:
        clip_lower, clip_upper = 1e-3, 1-1e-3
    
    return {
        "n_fold": n_fold,
        "clip_lower": clip_lower,
        "clip_upper": clip_upper,
        "random_state": 42,
        "reasoning": "Fallback rule-based selection"
    }

def estimate(data, treatment, outcome, covariates):
    """
    Propensity Score estimation with LLM-optimized hyperparameters.
    """
    
    # Get optimal hyperparameters using LLM
    hyperparams = _select_hyperparameters_llm(data, treatment, outcome, covariates)
    print(f"Propensity Score Hyperparameters selected: {hyperparams.get('reasoning', 'No reasoning provided')}")
    
    # Extract arrays
    Y = data[outcome].to_numpy()
    T = data[treatment].to_numpy()
    X = data[covariates].to_numpy()

    # Build propensity score model with optimized hyperparameters
    clip_bounds = (hyperparams["clip_lower"], hyperparams["clip_upper"])
    
    pm = ElasticNetPropensityModel(
        n_fold=hyperparams["n_fold"],
        random_state=hyperparams["random_state"],
        clip_bounds=clip_bounds
    )
    p_hat = pm.fit_predict(X, T)

    # ATE
    ate_scores = T * Y / p_hat - (1 - T) * Y / (1 - p_hat)
    ate = ate_scores.mean()
    ate_se = ate_scores.std(ddof=1) / np.sqrt(len(Y))

    # ATT
    att_scores = (T - p_hat) * Y / p_hat
    att = att_scores.mean()
    att_se = att_scores.std(ddof=1) / np.sqrt(len(Y))

    return {
        "ATE": float(ate),
        "ATE_std_error": float(ate_se),
        "ATT": float(att),
        "ATT_std_error": float(att_se),
        "model": pm,
        "hyperparameters": hyperparams
    }

def diagnose(data, treatment, outcome, covariates, alpha=0.05):
    """
    Comprehensive propensity score diagnostic tests.
    
    Returns:
    --------
    dict : Dictionary with diagnostic test results
    """
    df = data.copy()
    
    diagnostics = {
        'positivity': True,
        'balance': True,
        'overlap': True,
        'predictability': True,
        'overall_valid': True
    }
    
    try:
        # Prepare data
        X = df[covariates].values
        T = df[treatment].values
        Y = df[outcome].values
        
        # Fit propensity score model
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X, T)
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        
        # 1. Test for positivity (propensity scores should be bounded away from 0 and 1)
        min_ps = propensity_scores.min()
        max_ps = propensity_scores.max()
        diagnostics['positivity'] = (min_ps > 0.01) and (max_ps < 0.99)
        
        # 2. Test for overlap (sufficient overlap in propensity score distributions)
        treated_ps = propensity_scores[T == 1]
        control_ps = propensity_scores[T == 0]
        
        if len(treated_ps) > 0 and len(control_ps) > 0:
            # Check quantile overlap
            treated_q10, treated_q90 = np.percentile(treated_ps, [10, 90])
            control_q10, control_q90 = np.percentile(control_ps, [10, 90])
            
            overlap_range = min(treated_q90, control_q90) - max(treated_q10, control_q10)
            total_range = max(treated_q90, control_q90) - min(treated_q10, control_q10)
            
            overlap_ratio = overlap_range / total_range if total_range > 0 else 0
            diagnostics['overlap'] = overlap_ratio > 0.1  # At least 10% overlap
        
        # 3. Test for balance (standardized mean differences should be small after weighting)
        # Simple balance test using standardized mean differences
        balance_results = []
        for i, covar in enumerate(covariates):
            if covar in df.columns:
                treated_mean = df[df[treatment] == 1][covar].mean()
                control_mean = df[df[treatment] == 0][covar].mean()
                pooled_std = df[covar].std()
                
                if pooled_std > 0:
                    smd = abs(treated_mean - control_mean) / pooled_std
                    balance_results.append(smd < 0.25)  # SMD < 0.25 is generally acceptable
        
        diagnostics['balance'] = all(balance_results) if balance_results else True
        
        # 4. Test predictability of treatment (propensity scores should have some predictive power)
        try:
            auc_score = roc_auc_score(T, propensity_scores)
            # AUC should be significantly different from 0.5 but not too close to 1
            diagnostics['predictability'] = 0.55 < auc_score < 0.95
        except:
            # Fallback: simple correlation test
            corr, p_val = stats.pointbiserialr(T, propensity_scores)
            diagnostics['predictability'] = (abs(corr) > 0.1) and (abs(corr) < 0.9)
        
        # Overall validity
        diagnostics['overall_valid'] = all([
            diagnostics['positivity'],
            diagnostics['balance'],
            diagnostics['overlap'],
            diagnostics['predictability']
        ])
        
    except Exception as e:
        print(f"Propensity score diagnostic failed: {e}")
        diagnostics['overall_valid'] = False
    
    return diagnostics



# import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# def estimate(
#     data,
#     treatment,
#     outcome,
#     covariates,
#     effect
# ):
#     df = data.copy()

#     # Validate inputs
#     if treatment not in df.columns or outcome not in df.columns:
#         raise KeyError("Treatment or outcome column not found in data.")
#     for cov in covariates:
#         if cov not in df.columns:
#             raise KeyError(f"Covariate '{cov}' not found in data.")

#     # Fit propensity score model
#     X_ps = sm.add_constant(df[covariates])
#     ps_model = sm.Logit(df[treatment], X_ps).fit(disp=False)
#     pscore = ps_model.predict(X_ps)
#     # Trim extreme scores
#     pscore = np.clip(pscore, 1e-3, 1-1e-3)
#     df['pscore'] = pscore

#     # Compute weights
#     if effect == 'ate':
#         df['weights'] = df[treatment] / df['pscore'] + (1 - df[treatment]) / (1 - df['pscore'])
#     elif effect == 'att':
#         df['weights'] = np.where(
#             df[treatment] == 1,
#             1.0,
#             df['pscore'] / (1 - df['pscore'])
#         )

#     # Weighted regression of outcome on treatment
#     formula = f"{outcome} ~ {treatment}"
#     outcome_model = smf.wls(formula, data=df, weights=df['weights']).fit()
#     estimate = outcome_model.params[treatment]
#     std_error = outcome_model.bse[treatment]

#     return {
#         'estimate': estimate,
#         'std_error': std_error,
#         'ps_model': ps_model,
#         'outcome_model': outcome_model
#     }
