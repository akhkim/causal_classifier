import statsmodels.formula.api as smf

def estimate(data,
                              treatment,
                              outcome,
                              time_variable,
                              covariates=None):
    """
    - Assumes treatment is a between-unit grouping (same for both periods).
    - Infers pre vs. post-period by sorting the two unique values of `time_variable`.
    """
    # Check exactly two time periods
    unique_times = data[time_variable].unique()
    if len(unique_times) != 2:
        raise ValueError(f"Expected exactly 2 unique values in '{time_variable}', got {len(unique_times)}.")
    sorted_times = sorted(unique_times)
    post_time = sorted_times[1]

    # Create indicators in a temp copy of data
    df = data.copy()
    df['__post__'] = (df[time_variable] == post_time).astype(int)
    df['__treat__'] = treatment
    df['__y__'] = outcome
    df['__did__'] = df['__treat__'] * df['__post__']

    # Build regression formula
    formula = "__y__ ~ __treat__ + __post__ + __did__"
    if covariates:
        formula += " + " + " + ".join(covariates)

    # Fit OLS model
    model = smf.ols(formula, data=df).fit()
    estimate = model.params['__did__']
    std_err = model.bse['__did__']

    return {'estimate': estimate, 'std_error': std_err, 'model': model}




# from linearmodels.panel import PanelOLS

# def difference_in_differences(df, entity, time, treatment, outcome, covariates=None):
#     # set multi-index
#     panel = df.set_index([entity, time])
#     exog = [treatment]
#     if covariates:
#         exog += covariates
#     exog = sm.add_constant(panel[exog])
#     endog = panel[outcome]
#     mod = PanelOLS(endog, exog, entity_effects=True, time_effects=True).fit()
#     return {
#         'estimate': mod.params[treatment],
#         'std_error': mod.std_errors[treatment],
#         'model': mod
#     }