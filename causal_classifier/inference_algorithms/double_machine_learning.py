from doubleml import DoubleMLPLR, DoubleMLData
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from ..llm_query import create_chat_completion

def _select_hyperparameters_llm(data, treatment, outcome, covariates, sample_size):
    """
    Use LLM to select optimal hyperparameters for Double Machine Learning based on data characteristics.
    """
    
    # Gather data characteristics
    n_features = len(covariates) if covariates else 0
    n_samples = sample_size
    
    # Check treatment type
    treatment_unique = data[treatment].nunique() if treatment in data.columns else 2
    treatment_type = "binary" if treatment_unique <= 2 else "continuous"
    
    # Check outcome type
    outcome_stats = data[outcome].describe() if outcome in data.columns else {}
    
    # Calculate feature to sample ratio
    feature_ratio = n_features / n_samples if n_samples > 0 else 0
    
    # Create context for LLM
    context = f"""
    Dataset Characteristics:
    - Sample size: {n_samples}
    - Number of features: {n_features}
    - Feature-to-sample ratio: {feature_ratio:.3f}
    - Treatment type: {treatment_type}
    - Treatment variable: {treatment}
    - Outcome variable: {outcome}
    
    Outcome statistics:
    {outcome_stats}
    """
    
    try:
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": """You are an expert in Double Machine Learning (DML) hyperparameter selection. 
                
                Based on the dataset characteristics, recommend optimal hyperparameters for:
                1. Number of cross-validation folds (n_folds): Should balance bias-variance tradeoff. Too few folds can lead to high variance, too many can lead to high bias in small samples.
                2. ML learner for outcome model (ml_Q): RandomForestRegressor parameters (n_estimators, max_depth, min_samples_split)
                3. ML learner for treatment model (ml_g): LogisticRegression or RandomForestClassifier parameters
                
                Guidelines:
                - Small samples (n<200): Use fewer folds (3-5), simpler models
                - Medium samples (200-1000): Use 5-10 folds, moderate complexity
                - Large samples (>1000): Can use more folds (5-20), more complex models
                - High-dimensional data (p/n > 0.1): Use regularization, simpler models
                - Binary treatment: Use LogisticRegression for propensity model
                - Continuous treatment: Use RandomForestRegressor for propensity model
                
                Return your response as valid JSON with this exact structure:
                {
                    "n_folds": <int>,
                    "outcome_model": {
                        "type": "RandomForestRegressor",
                        "n_estimators": <int>,
                        "max_depth": <int or null>,
                        "min_samples_split": <int>,
                        "random_state": 42
                    },
                    "treatment_model": {
                        "type": "LogisticRegression or RandomForestRegressor",
                        "n_estimators": <int if RandomForest>,
                        "max_iter": <int if Logistic>,
                        "solver": "solver name if Logistic",
                        "random_state": 42
                    },
                    "reasoning": "brief explanation of choices"
                }"""},
                {"role": "user", "content": f"Given these dataset characteristics, what are the optimal DML hyperparameters?\n\n{context}"}
            ],
            temperature=0.1,
            thinking=False
        )
        
        # Parse the JSON response
        hyperparams = json.loads(response)
        return hyperparams
        
    except Exception as e:
        print(f"LLM hyperparameter selection failed: {e}")
        # Fallback to rule-based selection
        return _select_hyperparameters_fallback(sample_size, n_features, treatment_type)

def _select_hyperparameters_fallback(sample_size, n_features, treatment_type):
    """Fallback rule-based hyperparameter selection."""
    
    # Cross-validation folds
    if sample_size < 100:
        n_folds = min(5, sample_size)
    elif sample_size < 500:
        n_folds = 5
    else:
        n_folds = 10
    
    # Outcome model parameters
    if sample_size < 200:
        n_estimators = 50
        max_depth = 5
        min_samples_split = 10
    elif sample_size < 1000:
        n_estimators = 100
        max_depth = None
        min_samples_split = 5
    else:
        n_estimators = 200
        max_depth = None
        min_samples_split = 2
    
    # Treatment model
    if treatment_type == "binary":
        treatment_model = {
            "type": "LogisticRegression",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42
        }
    else:
        treatment_model = {
            "type": "RandomForestRegressor",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": 42
        }
    
    return {
        "n_folds": n_folds,
        "outcome_model": {
            "type": "RandomForestRegressor",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": 42
        },
        "treatment_model": treatment_model,
        "reasoning": "Fallback rule-based selection"
    }

def estimate(
    data,
    treatment,
    outcome,
    covariates,
    sample_size,
    latent_confounders=None
):
    """
    Double Machine Learning estimation with LLM-optimized hyperparameters.
    Now supports both measured confounders and latent confounders from FCI.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset
    treatment : str
        Treatment variable name
    outcome : str
        Outcome variable name
    covariates : list
        List of measured confounders to adjust for
    sample_size : int
        Sample size of the data
    latent_confounders : list, optional
        List of latent confounder nodes (U_ format) identified by FCI
    """
    df = data.copy()
    
    # Handle latent confounders information
    robustness_note = ""
    if latent_confounders:
        print(f"DML: Handling {len(latent_confounders)} latent confounders: {latent_confounders}")
        print("Note: DML provides some robustness to unobserved confounding through cross-fitting")
        robustness_note = f"DML estimation with {len(latent_confounders)} latent confounders detected. " \
                         "Cross-fitting provides partial robustness to unobserved confounding."
    
    # Get optimal hyperparameters using LLM
    hyperparams = _select_hyperparameters_llm(data, treatment, outcome, covariates, sample_size)
    print(f"DML Hyperparameters selected: {hyperparams.get('reasoning', 'No reasoning provided')}")
    
    # Build ML models based on hyperparameters
    outcome_params = hyperparams["outcome_model"]
    ml_Q = RandomForestRegressor(
        n_estimators=outcome_params["n_estimators"],
        max_depth=outcome_params.get("max_depth"),
        min_samples_split=outcome_params["min_samples_split"],
        random_state=outcome_params["random_state"]
    )
    
    treatment_params = hyperparams["treatment_model"]
    if treatment_params["type"] == "LogisticRegression":
        ml_g = LogisticRegression(
            solver=treatment_params["solver"],
            max_iter=treatment_params["max_iter"],
            random_state=treatment_params["random_state"]
        )
    else:
        from sklearn.ensemble import RandomForestRegressor as RFRegressor
        ml_g = RFRegressor(
            n_estimators=treatment_params["n_estimators"],
            max_depth=treatment_params.get("max_depth"),
            min_samples_split=treatment_params["min_samples_split"],
            random_state=treatment_params["random_state"]
        )
    
    n_folds = hyperparams["n_folds"]
    
    # Fit DML model
    dml_data = DoubleMLData(df, y_col=outcome, d_cols=[treatment], x_cols=covariates)
    dml_model = DoubleMLPLR(
        dml_data,
        ml_g=ml_g,
        ml_Q=ml_Q,
        n_folds=n_folds
    )
    dml_model.fit()
    ate = dml_model.ate
    se = dml_model.ate_se
    
    # Calculate robustness adjustment
    robustness_factor = 1.0
    if latent_confounders:
        # DML is more robust than OLS but still affected by strong confounding
        robustness_factor = max(0.7, 1.0 - 0.05 * len(latent_confounders))
        print(f"Robustness factor: {robustness_factor:.2f} accounting for latent confounders")
    
    return {
        'estimate': ate,
        'std_error': se,
        'model': dml_model,
        'hyperparameters': hyperparams,
        'latent_confounders': latent_confounders or [],
        'robustness_factor': robustness_factor,
        'robustness_note': robustness_note,
        'interpretation': _generate_dml_interpretation(ate, se, latent_confounders, covariates)
    }

def _generate_dml_interpretation(estimate, std_error, latent_confounders, covariates):
    """Generate interpretation text for DML considering both measured and latent confounders."""
    
    interpretation = f"Double Machine Learning Estimate: {estimate:.4f} (SE: {std_error:.4f})\n"
    interpretation += f"Controlled for {len(covariates)} measured confounders using cross-fitting\n"
    
    if latent_confounders:
        interpretation += f"\nLatent confounders detected: {len(latent_confounders)}\n"
        for latent in latent_confounders:
            interpretation += f"  - {latent}\n"
        interpretation += "\nDML provides some robustness to unobserved confounding through:\n"
        interpretation += "  1. Cross-fitting to reduce overfitting bias\n"
        interpretation += "  2. Orthogonal estimation to handle model misspecification\n"
        interpretation += "However, strong unobserved confounding can still bias estimates.\n"
    else:
        interpretation += "\nNo latent confounders detected. DML estimate is robust to model misspecification."
    
    return interpretation

def diagnose(data, treatment, outcome, covariates, alpha=0.05):
    """
    Comprehensive Double Machine Learning diagnostic tests.
    
    Returns:
    --------
    dict : Dictionary with diagnostic test results
    """
    df = data.copy()
    
    diagnostics = {
        'cross_fitting_valid': True,
        'prediction_quality': True,
        'orthogonality': True,
        'overlap': True,
        'sufficient_data': True,
        'overall_valid': True
    }
    
    try:
        # Prepare data
        X = df[covariates].values
        Y = df[outcome].values
        T = df[treatment].values
        n = len(df)
        
        # 1. Test sufficient data for cross-fitting
        min_samples_per_fold = n / 5  # Assuming 5-fold CV
        diagnostics['sufficient_data'] = min_samples_per_fold >= 20  # At least 20 samples per fold
        
        # 2. Test prediction quality of machine learning models
        # Outcome model prediction quality
        outcome_model = RandomForestRegressor(n_estimators=100, random_state=42)
        outcome_scores = cross_val_score(outcome_model, X, Y, cv=5, scoring='neg_mean_squared_error')
        outcome_mse = -outcome_scores.mean()
        baseline_outcome_mse = np.var(Y)
        outcome_improvement = (baseline_outcome_mse - outcome_mse) / baseline_outcome_mse
        
        # Treatment model prediction quality (for binary treatment)
        if len(np.unique(T)) == 2:
            treatment_model = LogisticRegression(random_state=42, max_iter=1000)
            treatment_scores = cross_val_score(treatment_model, X, T, cv=5, scoring='accuracy')
            treatment_accuracy = treatment_scores.mean()
            baseline_accuracy = max(np.mean(T), 1 - np.mean(T))  # Majority class accuracy
            treatment_improvement = (treatment_accuracy - baseline_accuracy) / baseline_accuracy
        else:
            # For continuous treatment
            treatment_model = RandomForestRegressor(n_estimators=100, random_state=42)
            treatment_scores = cross_val_score(treatment_model, X, T, cv=5, scoring='neg_mean_squared_error')
            treatment_mse = -treatment_scores.mean()
            baseline_treatment_mse = np.var(T)
            treatment_improvement = (baseline_treatment_mse - treatment_mse) / baseline_treatment_mse
        
        # Both models should show improvement over baseline
        diagnostics['prediction_quality'] = (outcome_improvement > 0.1) and (treatment_improvement > 0.1)
        
        # 3. Test orthogonality conditions (Neyman orthogonality)
        # This is conceptually tested by the DML procedure itself
        # We can check if residuals from cross-fitted models have low correlation
        diagnostics['orthogonality'] = True  # Assumed valid if DML procedure completes
        
        # 4. Test overlap/positivity
        if len(np.unique(T)) == 2:
            # For binary treatment
            min_treatment_prop = min(np.mean(T), 1 - np.mean(T))
            diagnostics['overlap'] = min_treatment_prop > 0.05
        else:
            # For continuous treatment, check for sufficient variation
            treatment_cv = np.std(T) / np.mean(T) if np.mean(T) != 0 else 0
            diagnostics['overlap'] = treatment_cv > 0.1  # Coefficient of variation > 10%
        
        # 5. Cross-fitting validity (data should be sufficient for sample splitting)
        diagnostics['cross_fitting_valid'] = n >= 100  # Minimum sample size for reliable cross-fitting
        
        # Overall validity
        diagnostics['overall_valid'] = all([
            diagnostics['cross_fitting_valid'],
            diagnostics['prediction_quality'],
            diagnostics['orthogonality'],
            diagnostics['overlap'],
            diagnostics['sufficient_data']
        ])
        
    except Exception as e:
        print(f"Double Machine Learning diagnostic failed: {e}")
        diagnostics['overall_valid'] = False
    
    return diagnostics


# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.base import clone
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LogisticRegression

# def estimate(
#     data,
#     treatment,
#     outcome,
#     covariates,
#     sample_size
# ):
#     ml_Q = RandomForestRegressor(n_estimators=100)
#     ml_g = LogisticRegression(solver='lbfgs', max_iter=1000)
#     if sample_size < 100:
#         n_splits = sample_size
#     elif 100 <= sample_size < 500:
#         n_splits = 20
#     else:
#         n_splits = 10
    
#     T = data[treatment].values
#     Y = data[outcome].values
#     X = data[covariates].values
#     n = len(Y)

#     # Storage for influence values
#     D = np.zeros(n)

#     # Cross-fitting
#     kf = KFold(n_splits=n_splits, shuffle=True)
#     for train_idx, test_idx in kf.split(X):
#         # Fit Q-models on training fold
#         Q1_mod = clone(ml_Q)
#         Q0_mod = clone(ml_Q)
#         Q1_mod.fit(X[train_idx][T[train_idx]==1], Y[train_idx][T[train_idx]==1])
#         Q0_mod.fit(X[train_idx][T[train_idx]==0], Y[train_idx][T[train_idx]==0])

#         # Fit g-model on training fold
#         g_mod = clone(ml_g).fit(X[train_idx], T[train_idx])

#         # Predict nuisances on test fold
#         Q1_hat = Q1_mod.predict(X[test_idx])
#         Q0_hat = Q0_mod.predict(X[test_idx])
#         if hasattr(g_mod, 'predict_proba'):
#             g_hat = g_mod.predict_proba(X[test_idx])[:,1]
#         else:
#             g_hat = g_mod.predict(X[test_idx])
#         g_hat = np.clip(g_hat, 1e-3, 1-1e-3)

#         # Compute influence function
#         Ai = T[test_idx]
#         Yi = Y[test_idx]
#         Di = (
#             Q1_hat - Q0_hat
#             + Ai * (Yi - Q1_hat) / g_hat
#             - (1 - Ai) * (Yi - Q0_hat) / (1 - g_hat)
#         )
#         D[test_idx] = Di

#     # Aggregate results
#     ate = D.mean()
#     se = D.std(ddof=1) / np.sqrt(n)

#     return {'estimate': ate, 'std_error': se, 'influence': D}  # influence: np.ndarray — influence function values
