import statsmodels.api as sm

def iv(
    data, 
    treatment, 
    outcome, 
    instruments, 
    covariates=None
):
    df = data.copy()

    # First-stage: treatment ~ instruments + covariates
    exog_fs = instruments.copy()
    if covariates:
        exog_fs += covariates
    X1 = sm.add_constant(df[exog_fs])
    y1 = df[treatment]
    first_stage = sm.OLS(y1, X1).fit()
    df['__treatment_hat__'] = first_stage.predict(X1)

    # Second-stage: outcome ~ treatment_hat + covariates
    exog_ss = ['__treatment_hat__']
    if covariates:
        exog_ss += covariates
    X2 = sm.add_constant(df[exog_ss])
    y2 = df[outcome]
    second_stage = sm.OLS(y2, X2).fit()

    # Extract estimate and standard error for treatment effect
    estimate = second_stage.params['__treatment_hat__']
    std_error = second_stage.bse['__treatment_hat__']

    # Cleanup
    df.drop(columns=['__treatment_hat__'], inplace=True)

    return {
        'estimate': estimate,
        'std_error': std_error,
        'first_stage': first_stage,
        'second_stage': second_stage
    }




# from linearmodels.iv import IV2SLS

# def iv(data, treatment, outcome, instruments, covariates=None):
#     exog = covariates if covariates else []
#     # build formula: outcome ~ exog + [treatment ~ instruments]
#     inst = ' + '.join(instruments)
#     ex = ' + '.join(exog) if exog else '1'
#     formula = f"{outcome} ~ {ex} + [{treatment} ~ {inst}]"
#     iv_res = IV2SLS.from_formula(formula, data).fit(cov_type='robust')
#     return {
#         'estimate': iv_res.params[treatment],
#         'std_error': iv_res.std_errors[treatment],
#         'model': iv_res
#     }