import rdd
import json
import numpy as np
import pandas as pd
from scipy import stats
from ..llm_query import create_chat_completion

def _select_hyperparameters_llm(data, outcome, running_variable, cutoff_value):
    """
    Use LLM to select optimal hyperparameters for RDD based on data characteristics.
    """
    
    # Gather data characteristics
    n_samples = len(data)
    
    # Analyze running variable distribution around cutoff
    running_data = data[running_variable].dropna()
    cutoff_distance = np.abs(running_data - cutoff_value)
    
    # Calculate data density around cutoff
    within_1sd = np.sum(cutoff_distance <= running_data.std())
    within_05sd = np.sum(cutoff_distance <= 0.5 * running_data.std())
    
    # Calculate optimal bandwidth using rule-of-thumb
    h_rot = 1.06 * running_data.std() * (len(running_data) ** (-1/5))
    
    # Check for potential discontinuities/bunching around cutoff
    near_cutoff = running_data[cutoff_distance <= 0.1 * running_data.std()]
    bunching_score = len(near_cutoff) / len(running_data) if len(running_data) > 0 else 0
    
    context = f"""
    Dataset Characteristics for RDD:
    - Sample size: {n_samples}
    - Running variable: {running_variable}
    - Cutoff value: {cutoff_value}
    - Running variable std: {running_data.std():.3f}
    - Rule-of-thumb bandwidth: {h_rot:.3f}
    - Observations within 1 SD of cutoff: {within_1sd}
    - Observations within 0.5 SD of cutoff: {within_05sd}
    - Potential bunching at cutoff: {bunching_score:.3f}
    - Running variable range: {running_data.min():.2f} to {running_data.max():.2f}
    """
    
    try:
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": """You are an expert in Regression Discontinuity Design (RDD) hyperparameter selection.
                
                Based on the dataset characteristics, recommend optimal hyperparameters for:
                1. bandwidth: Controls the window around the cutoff used for estimation. Too small = high variance, too large = bias
                2. kernel: Type of kernel for weighting observations by distance from cutoff
                
                Guidelines:
                - Small samples (n<500): Use wider bandwidth (0.3-0.5) to ensure sufficient observations
                - Large samples (>2000): Can use narrower bandwidth (0.1-0.25) for precise estimation
                - High density near cutoff: Can use narrower bandwidth
                - Low density near cutoff: Need wider bandwidth
                - Evidence of bunching: Use wider bandwidth to avoid manipulation issues
                - Sharp discontinuity: Can use narrower bandwidth
                - Fuzzy discontinuity: May need wider bandwidth
                
                Available kernels:
                - "triangular": Linear decay, good default choice
                - "rectangular": Uniform weights, simple but can be noisy
                - "epanechnikov": Quadratic decay, optimal in MSE sense
                - "gaussian": Smooth decay, good for continuous outcomes
                
                Return your response as valid JSON with this exact structure:
                {
                    "bandwidth": <float>,
                    "kernel": "<kernel_name>",
                    "reasoning": "<brief explanation of choices>"
                }"""},
                {"role": "user", "content": f"Given these RDD dataset characteristics, what are the optimal hyperparameters?\n\n{context}"}
            ],
            temperature=0.1,
            thinking=False
        )
        
        hyperparams = json.loads(response)
        return hyperparams
        
    except Exception as e:
        print(f"LLM hyperparameter selection failed: {e}")
        # Fallback to rule-based selection
        return _select_hyperparameters_fallback(n_samples, within_1sd, bunching_score, h_rot)

def _select_hyperparameters_fallback(n_samples, within_1sd, bunching_score, h_rot):
    """Fallback rule-based hyperparameter selection."""
    
    # Bandwidth selection
    if n_samples < 500:
        bandwidth = max(0.4, h_rot)  # Wider for small samples
    elif n_samples > 2000:
        bandwidth = min(0.2, h_rot)  # Narrower for large samples
    else:
        bandwidth = max(0.25, min(0.35, h_rot))  # Moderate for medium samples
    
    # Adjust for data density and bunching
    if within_1sd < 20:  # Low density
        bandwidth *= 1.5
    elif bunching_score > 0.1:  # Evidence of bunching
        bandwidth *= 1.3
    
    # Kernel selection
    if n_samples < 1000:
        kernel = "triangular"  # Good default
    else:
        kernel = "epanechnikov"  # Optimal for larger samples
    
    return {
        "bandwidth": bandwidth,
        "kernel": kernel,
        "reasoning": "Fallback rule-based selection based on sample size and data density"
    }

def estimate(data, outcome, running_variable, cutoff_value, covariates=None):
    """
    RDD estimation with LLM-optimized hyperparameters.
    """
    
    # Get optimal hyperparameters using LLM
    hyperparams = _select_hyperparameters_llm(data, outcome, running_variable, cutoff_value)
    print(f"RDD Hyperparameters selected: {hyperparams.get('reasoning', 'No reasoning provided')}")
    
    # Run RDD with optimized hyperparameters
    rdd_obj = rdd.rdd(
        data,
        outcome_col=outcome,
        running_col=running_variable,
        cutpoint=cutoff_value,
        kernel=hyperparams["kernel"],
        bw=hyperparams["bandwidth"]
    )
    results = rdd_obj.fit()
    
    return {
        'estimate': results.params['treatment'],
        'std_error': results.std_errors['treatment'],
        'model': results,
        'hyperparameters': hyperparams
    }

def diagnose(data, running, treatment, cutoff, covariates, bandwidth=0.25):
    """
    Quick falsification tests for a *sharp* RDD.
    Return True only when all tests pass.
    """
    # a. Treatment must jump at the cutoff (sharp RDD) 
    df = data.copy()
    df["assigned"] = (df[running] >= cutoff).astype(int)
    jump = abs(df["assigned"].mean() - df[treatment].mean())     # ≈ sharpness threshold
    if jump < 0.80:
        return False
    
    # b. No bunching at cutoff (McCrary density proxy)
    z = (df[running] - cutoff) / df[running].std()
    p_below = ((z > -0.02) & (z < 0)).sum()
    p_above = ((z >= 0) & (z < 0.02)).sum()
    if abs(p_above - p_below) > 0.20 * max(p_below, p_above, 1):
        return False
    
    # c. Covariate continuity
    for c in covariates:
        left  = df.loc[df[running] <  cutoff, c]
        right = df.loc[df[running] >= cutoff, c]
        if stats.ttest_ind(left, right).pvalue < 0.05:
            return False
    
    # d. Enough mass near the cutoff
    bw = bandwidth * df[running].std()
    n_window = df.loc[abs(df[running] - cutoff) <= bw].shape[0]
    return n_window >= 50 



# import statsmodels.formula.api as smf

# def estimate(data,
#         outcome,
#         running_variable,
#         cutoff,
#         bandwidth=None
# ):
#     """
#     Method:
#     1. Construct relative running variable: x = running_variable - cutoff.
#     2. Define treatment indicator D = (x >= 0).
#     3. Optionally subset data to |x| <= bandwidth.
#     4. Fit OLS: outcome ~ D + sum_{p=1}^order x^p + sum_{p=1}^order D * x^p + covariates.
#     5. The coefficient on D is the local treatment effect at the cutoff.
#     """
#     df = data.copy()
#     df['__x__'] = df[running_variable] - cutoff
#     df['__D__'] = (df['__x__'] >= 0).astype(int)

#     # Apply bandwidth if specified
#     if bandwidth is not None:
#         df = df.loc[df['__x__'].abs() <= bandwidth].copy()

#     # Build formula components
#     terms = ['__D__']
#     # Polynomial terms for x and interaction
#     for p in range(1, order + 1):
#         df[f'__x{p}__'] = df['__x__'] ** p
#         df[f'__D_x{p}__'] = df['__D__'] * df[f'__x{p}__']
#         terms.append(f'__x{p}__')
#         terms.append(f'__D_x{p}__')

#     # Include covariates if provided
#     if covariates:
#         terms.extend(covariates)

#     # Construct formula
#     formula = f"{outcome} ~ " + " + ".join(terms)

#     # Fit the model
#     model = smf.ols(formula, data=df).fit()

#     # Extract treatment effect estimate and std error
#     estimate = model.params['__D__']
#     std_err = model.bse['__D__']

#     return {'estimate': estimate, 'std_error': std_err, 'model': model}

