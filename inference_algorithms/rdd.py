from rdd import RDD
from scipy import stats

def estimate(data,
        outcome,
        running_variable,
        cutoff_value,
        bandwidth=0.25
):
    rdd_obj = RDD(
        data,
        outcome_col=outcome,
        running_col=running_variable,
        cutpoint=cutoff_value,
        kernel='triangular',
        bw=bandwidth
    )
    results = rdd_obj.fit()
    return {
        'estimate': results.params['treatment'],
        'std_error': results.std_errors['treatment'],
        'model': results
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

