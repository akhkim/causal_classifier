import statsmodels.formula.api as smf

def rdd(data,
        outcome,
        running_variable,
        cutoff,
        covariates=None,
        order=1,
        bandwidth=None
):
    """
    Method:
    1. Construct relative running variable: x = running_variable - cutoff.
    2. Define treatment indicator D = (x >= 0).
    3. Optionally subset data to |x| <= bandwidth.
    4. Fit OLS: outcome ~ D + sum_{p=1}^order x^p + sum_{p=1}^order D * x^p + covariates.
    5. The coefficient on D is the local treatment effect at the cutoff.
    """
    df = data.copy()
    df['__x__'] = df[running_variable] - cutoff
    df['__D__'] = (df['__x__'] >= 0).astype(int)

    # Apply bandwidth if specified
    if bandwidth is not None:
        df = df.loc[df['__x__'].abs() <= bandwidth].copy()

    # Build formula components
    terms = ['__D__']
    # Polynomial terms for x and interaction
    for p in range(1, order + 1):
        df[f'__x{p}__'] = df['__x__'] ** p
        df[f'__D_x{p}__'] = df['__D__'] * df[f'__x{p}__']
        terms.append(f'__x{p}__')
        terms.append(f'__D_x{p}__')

    # Include covariates if provided
    if covariates:
        terms.extend(covariates)

    # Construct formula
    formula = f"{outcome} ~ " + " + ".join(terms)

    # Fit the model
    model = smf.ols(formula, data=df).fit()

    # Extract treatment effect estimate and std error
    estimate = model.params['__D__']
    std_err = model.bse['__D__']

    return {'estimate': estimate, 'std_error': std_err, 'model': model}




# from rdd import RDD

# def rdd(data, outcome, running_variable, cutoff, order=1, bandwidth=None):
#     rdd_obj = RDD(
#         data,
#         outcome_col=outcome,
#         running_col=running_variable,
#         cutpoint=cutoff,
#         kernel='triangular',
#         bw=bandwidth
#     )
#     results = rdd_obj.fit()
#     return {
#         'estimate': results.params['treatment'],
#         'std_error': results.std_errors['treatment'],
#         'model': results
#     }