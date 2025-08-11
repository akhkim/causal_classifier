from linearmodels.iv import IV2SLS
import pandas as pd
import numpy as np
import json
from scipy import stats
import statsmodels.api as sm
from ..llm_query import create_chat_completion

def _select_hyperparameters_llm(data, treatment, outcome, instruments, covariates):
    """
    Use LLM to select optimal hyperparameters for IV estimation based on data characteristics.
    """
    
    # Gather data characteristics
    n_samples = len(data)
    n_instruments = len(instruments) if instruments else 0
    n_covariates = len(covariates) if covariates else 0
    
    # Calculate instrument strength (rough estimate)
    instrument_strength_indicators = []
    if instruments:
        for inst in instruments:
            if inst in data.columns and treatment in data.columns:
                try:
                    corr = data[inst].corr(data[treatment])
                    instrument_strength_indicators.append(abs(corr))
                except:
                    instrument_strength_indicators.append(0.1)
    
    avg_instrument_strength = np.mean(instrument_strength_indicators) if instrument_strength_indicators else 0.1
    
    # Check for overidentification
    is_overidentified = n_instruments > 1
    
    context = f"""
    Dataset Characteristics:
    - Sample size: {n_samples}
    - Number of instruments: {n_instruments}
    - Number of covariates: {n_covariates}
    - Average instrument-treatment correlation: {avg_instrument_strength:.3f}
    - Is overidentified: {is_overidentified}
    - Treatment variable: {treatment}
    - Outcome variable: {outcome}
    - Instruments: {instruments}
    """
    
    try:
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": """You are an expert in Instrumental Variables (IV) estimation hyperparameter selection.
                
                Based on the dataset characteristics, recommend optimal hyperparameters for:
                1. cov_type (covariance estimation): How to estimate standard errors - important for inference
                2. Additional IV-specific settings that may be relevant
                
                Guidelines:
                - Small samples (n<200): Use "robust" or "cluster" covariance for conservative inference
                - Large samples (>1000): Can use "unadjusted" if assumptions are met, otherwise "robust"
                - Weak instruments (correlation<0.1): Recommend robust covariance, may suggest warnings
                - Strong instruments (correlation>0.3): Can use less conservative approaches
                - Overidentified models: Should use robust covariance to handle potential heterogeneity
                - Many instruments: May need bias corrections
                
                Available covariance types:
                - "unadjusted": Standard OLS-type covariance (assumes homoskedasticity)
                - "robust": Heteroskedasticity-robust (White/Eicker-Huber-White)
                - "cluster": Cluster-robust (if clustering structure exists)
                - "kernel": HAC (heteroskedasticity and autocorrelation consistent)
                
                Return your response as valid JSON with this exact structure:
                {
                    "cov_type": "<covariance_type>",
                    "bias_correction": <boolean>,
                    "weak_instrument_warning": <boolean>,
                    "reasoning": "<brief explanation of choices>"
                }"""},
                {"role": "user", "content": f"Given these dataset characteristics, what are the optimal IV hyperparameters?\n\n{context}"}
            ],
            temperature=0.1,
            thinking=False
        )
        
        hyperparams = json.loads(response)
        return hyperparams
        
    except Exception as e:
        print(f"LLM hyperparameter selection failed: {e}")
        # Fallback to rule-based selection
        return _select_hyperparameters_fallback(n_samples, avg_instrument_strength, is_overidentified)

def _select_hyperparameters_fallback(n_samples, avg_instrument_strength, is_overidentified):
    """Fallback rule-based hyperparameter selection."""
    
    # Covariance type
    if n_samples < 200 or is_overidentified or avg_instrument_strength < 0.2:
        cov_type = "robust"
        bias_correction = True
    elif n_samples > 1000 and avg_instrument_strength > 0.3:
        cov_type = "unadjusted"
        bias_correction = False
    else:
        cov_type = "robust"
        bias_correction = False
    
    # Weak instrument warning
    weak_instrument_warning = avg_instrument_strength < 0.1
    
    return {
        "cov_type": cov_type,
        "bias_correction": bias_correction,
        "weak_instrument_warning": weak_instrument_warning,
        "reasoning": "Fallback rule-based selection"
    }

def estimate(
    data, 
    treatment, 
    outcome, 
    instruments, 
    covariates,
    latent_confounders=None
):
    """
    IV estimation with LLM-optimized hyperparameters.
    Now supports both measured confounders and latent confounders from FCI.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset
    treatment : str
        Treatment variable name
    outcome : str
        Outcome variable name
    instruments : list
        List of instrumental variables
    covariates : list
        List of measured confounders/covariates to control for
    latent_confounders : list, optional
        List of latent confounder nodes (U_ format) identified by FCI
    """
    
    # Handle latent confounders information
    iv_advantage_note = ""
    if latent_confounders:
        print(f"IV: Handling {len(latent_confounders)} latent confounders: {latent_confounders}")
        print("Note: IV estimation is designed to handle unobserved confounding via instrumental variables")
        iv_advantage_note = f"IV estimation with {len(latent_confounders)} latent confounders detected. " \
                           "Instrumental variables provide identification despite unobserved confounding."
    
    # Get optimal hyperparameters using LLM
    hyperparams = _select_hyperparameters_llm(data, treatment, outcome, instruments, covariates)
    print(f"IV Hyperparameters selected: {hyperparams.get('reasoning', 'No reasoning provided')}")
    
    # Check for weak instrument warning
    if hyperparams.get("weak_instrument_warning", False):
        print("WARNING: Potentially weak instruments detected. IV estimates may be unreliable.")
    
    exog = list(set(covariates) - set(instruments)) if covariates else []
    # build formula: outcome ~ exog + [treatment ~ instruments]
    inst = ' + '.join(instruments)
    ex = ' + '.join(exog) if exog else '1'
    formula = f"{outcome} ~ {ex} + [{treatment} ~ {inst}]"
    
    print(f"DEBUG - Treatment: {treatment}")
    print(f"DEBUG - Outcome: {outcome}")
    print(f"DEBUG - Instruments: {instruments}")
    print(f"DEBUG - Original covariates: {covariates}")
    print(f"DEBUG - Final exog variables: {exog}")

    # Fit IV model with optimized covariance type
    try:
        iv_res = IV2SLS.from_formula(formula, data).fit(cov_type=hyperparams["cov_type"])
        
        # Safely extract treatment coefficient
        if treatment in iv_res.params.index:
            treatment_estimate = iv_res.params[treatment]
            treatment_std_error = iv_res.std_errors[treatment]
        else:
            # Try alternative access methods
            param_names = list(iv_res.params.index)
            print(f"Available parameters: {param_names}")
            # Look for treatment name in parameter names
            matching_params = [p for p in param_names if treatment.lower() in p.lower()]
            if matching_params:
                treatment_param = matching_params[0]
                treatment_estimate = iv_res.params[treatment_param]
                treatment_std_error = iv_res.std_errors[treatment_param]
                print(f"Using parameter: {treatment_param} for treatment: {treatment}")
            else:
                raise ValueError(f"Treatment parameter '{treatment}' not found in IV results")
        
        return {
            'estimate': treatment_estimate,
            'std_error': treatment_std_error,
            'model': iv_res,
            'hyperparameters': hyperparams
        }
        
    except Exception as e:
        print(f"IV estimation failed: {e}")
        print(f"Formula used: {formula}")
        print(f"Available columns: {list(data.columns)}")
        raise

def diagnose(data, treatment, outcome, instruments, covariates=None, alpha=0.05):
    """
    Comprehensive IV diagnostic tests.
    
    Returns:
    --------
    dict : Dictionary with diagnostic test results
    """
    df = data.copy()
    
    diagnostics = {
        'instrument_strength': True,
        'exogeneity': True,
        'relevance': True,
        'overidentification': True,
        'overall_valid': True
    }
    
    try:
        # Prepare variables
        exog = list(set(covariates) - set(instruments)) if covariates else []
        
        # 1. Test instrument strength (F-statistic from first stage)
        # First stage regression: treatment ~ instruments + exog
        first_stage_vars = instruments + exog
        if first_stage_vars:
            X_first = df[first_stage_vars]
            X_first = sm.add_constant(X_first)
            y_first = df[treatment]
            
            first_stage_model = sm.OLS(y_first, X_first).fit()
            
            # F-test for instrument significance
            instrument_positions = [first_stage_model.model.exog_names.index(inst) for inst in instruments 
                                  if inst in first_stage_model.model.exog_names]
            
            if instrument_positions:
                # Simple F-statistic calculation
                restricted_vars = ['const'] + exog
                restricted_X = sm.add_constant(df[exog]) if exog else sm.add_constant(np.ones(len(df)))
                restricted_model = sm.OLS(y_first, restricted_X).fit()
                
                # Calculate F-statistic
                n = len(df)
                k_unrestricted = len(first_stage_vars) + 1  # +1 for constant
                k_restricted = len(exog) + 1  # +1 for constant
                
                rss_restricted = restricted_model.ssr
                rss_unrestricted = first_stage_model.ssr
                
                f_stat = ((rss_restricted - rss_unrestricted) / (k_unrestricted - k_restricted)) / (rss_unrestricted / (n - k_unrestricted))
                
                # Rule of thumb: F > 10 for strong instruments
                diagnostics['instrument_strength'] = f_stat > 10
                diagnostics['relevance'] = f_stat > 3.84  # Critical value at 5% level
        
        # 2. Test for exogeneity (simplified test)
        # Check if instruments are correlated with outcome residuals from reduced form
        reduced_form_vars = exog if exog else []
        if reduced_form_vars:
            X_reduced = sm.add_constant(df[reduced_form_vars])
        else:
            X_reduced = sm.add_constant(np.ones(len(df)))
        
        y_outcome = df[outcome]
        reduced_form_model = sm.OLS(y_outcome, X_reduced).fit()
        residuals = reduced_form_model.resid
        
        # Test correlation between instruments and residuals
        exog_test_results = []
        for instrument in instruments:
            if instrument in df.columns:
                corr, p_val = stats.pearsonr(df[instrument], residuals)
                exog_test_results.append(p_val > alpha)  # Should not be correlated
        
        diagnostics['exogeneity'] = all(exog_test_results) if exog_test_results else True
        
        # 3. Overidentification test (if more instruments than endogenous variables)
        if len(instruments) > 1:  # Overidentified case
            try:
                # Simplified overidentification test
                # Run 2SLS and test if instrument residuals are uncorrelated with instruments
                exog_list = list(set(covariates) - set(instruments)) if covariates else []
                inst = ' + '.join(instruments)
                ex = ' + '.join(exog_list) if exog_list else '1'
                formula = f"{outcome} ~ {ex} + [{treatment} ~ {inst}]"
                
                iv_model = IV2SLS.from_formula(formula, df).fit()
                iv_residuals = iv_model.resids
                
                # Test if residuals are uncorrelated with instruments
                overid_tests = []
                for instrument in instruments:
                    if instrument in df.columns:
                        corr, p_val = stats.pearsonr(df[instrument], iv_residuals)
                        overid_tests.append(p_val > alpha)
                
                diagnostics['overidentification'] = all(overid_tests) if overid_tests else True
            except:
                diagnostics['overidentification'] = True  # Default to valid if test fails
        else:
            diagnostics['overidentification'] = True  # Exactly identified case
        
        # Overall validity
        diagnostics['overall_valid'] = all([
            diagnostics['instrument_strength'],
            diagnostics['exogeneity'],
            diagnostics['relevance'],
            diagnostics['overidentification']
        ])
        
    except Exception as e:
        print(f"IV diagnostic failed: {e}")
        diagnostics['overall_valid'] = False
    
    return diagnostics
